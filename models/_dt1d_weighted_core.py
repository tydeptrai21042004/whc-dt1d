"""Private weighted axial core used by DT1D-Adapter and its ablations.

The public proposal is :class:`models.dt1d_adapter.DT1DAdapter`. This private
module exposes the configurable implementation only so reviewer ablations can
change one component at a time without creating additional proposal methods.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._dt1d_base import _BaseDT1DAdapter, _parse_offsets


_LAMBDA_MODES = {"learned", "fixed", "off"}
_LAMBDA_SCOPES = {"block", "axis"}
_SHIFT_NORMALIZATIONS = {"mean", "paper"}


class _DT1DWeightedCore(_BaseDT1DAdapter):
    """Configurable weighted axial core.

    ``shift_lambda_init`` is specified in the effective lambda domain.  For the
    learned mode it is converted to the unconstrained theta parameter through
    atanh(lambda/lambda_max).  The default is exactly zero.

    The canonical proposal learns both the bounded axial shift coefficient and
    the outer residual gate.  Reviewer ablations can fix or disable either
    quantity without creating additional proposal methods.
    """

    method_name = "_DT1DWeightedCore"
    proposal_name = "internal_weighted_core"
    fixed_final_offsets = (1, 2, 4)

    def __init__(
        self,
        C: int,
        *,
        axis: str = "hw",
        alpha_group: int = 16,
        residual_scale: float = 1.0,
        gate_init: float = 0.01,
        gate_mode: str = "fixed",
        padding_mode: str = "replicate",
        contrast_split: int = 8,
        detail_basis: str = "orth",
        detail_components: str = "offset4",
        active_offsets: str | Sequence[int] | None = "1,2,4",
        use_pointwise: bool = False,
        pointwise_ratio: int = 32,
        pointwise_groups: int = 4,
        use_bn: bool = False,
        cache_kernel: bool = False,
        project_l1: bool = True,
        joint_l1_cap: float = 1.0,
        shift_p: int = 2,
        shift_lambda_mode: str = "learned",
        shift_lambda_scope: str = "block",
        shift_lambda_init: float = 0.0,
        shift_lambda_max: float = 0.5,
        shift_normalization: str = "mean",
    ) -> None:
        detail_components = str(detail_components).lower()
        if detail_components not in {"offset4", "none"}:
            raise ValueError(
                "The weighted DT1D core supports detail_components='offset4' "
                "(final) or 'none' (ablation)."
            )

        offsets = _parse_offsets(active_offsets)
        p = int(shift_p)
        if p <= 0:
            raise ValueError(f"shift_p must be a positive integer, got {shift_p!r}")

        lambda_mode = str(shift_lambda_mode).lower()
        lambda_scope = str(shift_lambda_scope).lower()
        shift_norm = str(shift_normalization).lower()
        if lambda_mode not in _LAMBDA_MODES:
            raise ValueError(f"shift_lambda_mode must be one of {_LAMBDA_MODES}")
        if lambda_scope not in _LAMBDA_SCOPES:
            raise ValueError(f"shift_lambda_scope must be one of {_LAMBDA_SCOPES}")
        if shift_norm not in _SHIFT_NORMALIZATIONS:
            raise ValueError(
                f"shift_normalization must be one of {_SHIFT_NORMALIZATIONS}"
            )
        lambda_max = float(shift_lambda_max)
        if not math.isfinite(lambda_max) or lambda_max <= 0:
            raise ValueError("shift_lambda_max must be finite and > 0")
        lambda_init = float(shift_lambda_init)
        if not math.isfinite(lambda_init):
            raise ValueError("shift_lambda_init must be finite")
        if abs(lambda_init) > lambda_max + 1e-12:
            raise ValueError(
                f"|shift_lambda_init| must be <= shift_lambda_max ({lambda_max}), "
                f"got {lambda_init}"
            )
        if lambda_mode == "off" and abs(lambda_init) > 1e-12:
            raise ValueError("shift_lambda_init must be 0 when shift_lambda_mode='off'")

        super().__init__(
            C=C,
            axis=axis,
            alpha_group=alpha_group,
            residual_scale=residual_scale,
            gate_init=gate_init,
            gate_mode=gate_mode,
            padding_mode=padding_mode,
            contrast_split=contrast_split,
            detail_basis=detail_basis,
            detail_components=detail_components,
            active_offsets=offsets,
            use_pointwise=use_pointwise,
            pointwise_ratio=pointwise_ratio,
            pointwise_groups=pointwise_groups,
            use_bn=use_bn,
            cache_kernel=cache_kernel,
            # Projection is deliberately applied AFTER weighted shifting.
            project_l1=False,
        )

        self.project_l1 = bool(project_l1)
        joint_l1_cap = float(joint_l1_cap)
        if not math.isfinite(joint_l1_cap) or joint_l1_cap <= 0:
            raise ValueError("joint_l1_cap must be a finite positive value")
        self.joint_l1_cap = joint_l1_cap
        self.shift_p = p
        self.shift_lambda_mode = lambda_mode
        self.shift_lambda_scope = lambda_scope
        self.shift_lambda_max = lambda_max
        self.shift_normalization = shift_norm

        scalar_count = 1 if lambda_scope == "block" else self.num_axes
        if lambda_mode == "learned":
            ratio = max(-0.999999, min(0.999999, lambda_init / lambda_max))
            theta_init = math.atanh(ratio)
            self.shift_theta = nn.Parameter(torch.full((scalar_count,), theta_init))
        elif lambda_mode == "fixed":
            self.register_buffer(
                "shift_lambda_fixed",
                torch.full((scalar_count,), lambda_init, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.register_buffer(
                "shift_lambda_fixed",
                torch.zeros(scalar_count, dtype=torch.float32),
                persistent=True,
            )

        self.base_radius = self._infer_base_radius()
        self.base_kernel_size = 2 * self.base_radius + 1
        self.weighting_active = not (
            self.shift_lambda_mode == "off"
            or (
                self.shift_lambda_mode == "fixed"
                and abs(float(shift_lambda_init)) <= 1e-12
            )
        )
        self.effective_radius = self.base_radius + (self.shift_p if self.weighting_active else 0)
        self.effective_kernel_size = 2 * self.effective_radius + 1

        self.is_dt1d_weighted_core = True
        self.implementation = "dt1d_fused_weighted_axial"

    def _infer_base_radius(self) -> int:
        radius = max(self.active_offsets) if self.active_offsets else 0
        if self.detail_components == "offset4":
            radius = max(radius, 4)
        return int(radius)

    def shift_lambda(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return effective lambda values with shape (num_axes,)."""
        if self.shift_lambda_mode == "learned":
            value = self.shift_lambda_max * torch.tanh(
                self.shift_theta.to(device=device, dtype=dtype)
            )
        else:
            value = self.shift_lambda_fixed.to(device=device, dtype=dtype)
        if self.shift_lambda_scope == "block":
            value = value.expand(self.num_axes)
        return value

    def _crop_base_kernel(self, full17: torch.Tensor) -> torch.Tensor:
        center = 8
        r = self.base_radius
        cropped = full17[..., center - r : center + r + 1]
        if cropped.shape[-1] != self.base_kernel_size:
            raise RuntimeError(f"Unexpected base kernel shape: {tuple(cropped.shape)}")
        return cropped

    @staticmethod
    def _shift_pair(base: torch.Tensor, p: int, scale: float) -> torch.Tensor:
        """Return scale*(S_-p base + S_+p base) on an expanded support."""
        k = int(base.shape[-1])
        out = base.new_zeros(*base.shape[:-1], k + 2 * p)
        out[..., :k] += base
        out[..., 2 * p : 2 * p + k] += base
        return out * float(scale)

    def build_unprojected_kernels(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build fused weighted kernels before the final joint-L1 projection."""
        full17 = super().build_kernels(device, dtype, project=False).squeeze(2)
        base = self._crop_base_kernel(full17)

        if not self.weighting_active:
            return base.unsqueeze(2)

        p = self.shift_p
        centered = F.pad(base, (p, p))
        shift_scale = 0.5 if self.shift_normalization == "mean" else 1.0
        weighted = self._shift_pair(base, p, shift_scale)
        lam = self.shift_lambda(device, dtype).view(self.num_axes, 1, 1)
        mixed = (1.0 - lam) * centered + lam * weighted
        return mixed.unsqueeze(2)

    def build_kernels(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        project: bool | None = None,
    ) -> torch.Tensor:
        """Build final fused kernels with dynamic K9/K11/K13/... support."""
        if project is None:
            project = self.project_l1
        kernel = self.build_unprojected_kernels(device, dtype)
        if project:
            # Generalized joint mass control. For the canonical cap=1 this is
            # the standard projection. A cap tau>1 preserves the
            # finite convolution-domain bound while allowing a stronger
            # task-specific axial correction:
            #   ||k_h||_1 + ||k_w||_1 <= tau.
            joint_l1 = kernel.abs().sum(dim=-1).sum(dim=0).squeeze(-1)
            cap = kernel.new_tensor(self.joint_l1_cap)
            scale = torch.maximum(joint_l1 / cap, torch.ones_like(joint_l1))
            kernel = kernel / scale.view(1, self.C, 1, 1)
        if int(kernel.shape[-1]) != self.effective_kernel_size:
            raise RuntimeError(
                f"Expected effective kernel size {self.effective_kernel_size}, "
                f"got {tuple(kernel.shape)}"
            )
        return kernel

    def parameter_count_breakdown(self) -> Dict[str, int]:
        result = super().parameter_count_breakdown()
        weighted = self.shift_theta.numel() if hasattr(self, "shift_theta") else 0
        result["shift_weight"] = int(weighted)
        result["total"] = int(result["total"] + weighted)
        result["base_kernel_size"] = int(self.base_kernel_size)
        result["effective_kernel_size"] = int(self.effective_kernel_size)
        result["joint_l1_cap"] = float(self.joint_l1_cap)
        return result

    @property
    def convolution_calls_per_forward(self) -> int:
        # Weighting is fused into each axial kernel; no extra convolution branch.
        return self.num_axes

    def extra_repr(self) -> str:
        return (
            f"C={self.C}, axis={self.axis}, group={self.alpha_group}, "
            f"offsets={self.active_offsets}, baseK={self.base_kernel_size}, "
            f"p={self.shift_p}, effectiveK={self.effective_kernel_size}, "
            f"lambda_mode={self.shift_lambda_mode}, scope={self.shift_lambda_scope}, "
            f"lambda_max={self.shift_lambda_max}, shift_norm={self.shift_normalization}, "
            f"project_l1={self.project_l1}, l1_cap={self.joint_l1_cap}, "
            f"detail={self.detail_components}, "
            f"gate_mode={self.gate_mode}, padding={self.padding_mode}"
        )


__all__ = ["_DT1DWeightedCore"]
