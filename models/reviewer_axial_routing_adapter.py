"""Reviewer-only axial routing ablation.

This module is *not* the DT1D proposal.  It exists only to reproduce the
component questions raised during review: direct versus shifted symmetric
filtering, shared versus per-channel coefficients, fixed averaging versus
learned axis--scale routing, and one versus multiple dilation branches.

The canonical proposal remains :class:`models.dt1d_adapter.DT1DAdapter`, which
uses the frozen R124-P2-G16-Axis-LearnedGate realization; this module remains only as a reviewer routing control.
"""
from __future__ import annotations

import math
from math import gcd
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_dilations(value: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        values = [int(v.strip()) for v in value.replace(";", ",").split(",") if v.strip()]
    else:
        values = [int(v) for v in value]
    result = tuple(dict.fromkeys(values))
    if not result or any(v <= 0 for v in result):
        raise ValueError("dilations must contain unique positive integers")
    return result


class ReviewerAxialRoutingAdapter(nn.Module):
    """Explicit branch implementation used only for reviewer ablations.

    Args:
        C: Feature-channel count.
        axis: ``h``, ``w``, or ``hw``.
        order: Symmetric source-kernel order ``M``.
        dilations: Branch dilation factors.
        group_size: Channels sharing one coefficient sequence. Set to ``1``
            for unshared/per-channel coefficients.
        shifted: If true, apply ``w * (delta_-1 + delta_+1)``. If false,
            apply the direct even-symmetric source kernel ``w``.
        routing: ``fixed_average`` or ``learned_softmax`` over all axis-scale
            branches.
    """

    is_reviewer_ablation = True

    def __init__(
        self,
        C: int,
        *,
        axis: str = "hw",
        order: int = 2,
        dilations: str | Sequence[int] = "1,2,4",
        group_size: int = 16,
        shifted: bool = True,
        routing: str = "learned_softmax",
        temperature: float = 1.0,
        residual_scale: float = 1.0,
        gate_init: float = 0.01,
        gate_mode: str = "learned",
        padding_mode: str = "replicate",
        normalize_l1: bool = True,
        use_pointwise: bool = False,
        pointwise_ratio: int = 32,
        pointwise_groups: int = 4,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        if C <= 0:
            raise ValueError("C must be positive")
        if axis not in {"h", "w", "hw"}:
            raise ValueError("axis must be h, w, or hw")
        if order < 0:
            raise ValueError("order must be non-negative")
        if routing not in {"fixed_average", "learned_softmax"}:
            raise ValueError("routing must be fixed_average or learned_softmax")
        if gate_mode not in {"learned", "fixed"}:
            raise ValueError("gate_mode must be learned or fixed")
        if padding_mode not in {"reflect", "replicate", "zeros", "constant"}:
            raise ValueError(f"unsupported padding_mode={padding_mode!r}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.C = int(C)
        self.axis = axis
        self.axis_names = tuple(name for name in ("h", "w") if name in axis)
        self.order = int(order)
        self.dilations = _parse_dilations(dilations)
        self.group_size = max(1, int(group_size))
        self.num_groups = math.ceil(self.C / self.group_size)
        self.shifted = bool(shifted)
        self.routing = routing
        self.temperature = float(temperature)
        self.residual_scale = float(residual_scale)
        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.normalize_l1 = bool(normalize_l1)
        self.use_pointwise = bool(use_pointwise)

        # One even source sequence per axis, scale, and channel-sharing group.
        self.alpha = nn.Parameter(
            torch.zeros(len(self.axis_names), len(self.dilations), self.num_groups, self.order + 1)
        )
        with torch.no_grad():
            self.alpha[..., 0].fill_(1.0)

        branch_count = len(self.axis_names) * len(self.dilations)
        if routing == "learned_softmax":
            self.route_logits = nn.Parameter(torch.zeros(branch_count))
        else:
            self.register_buffer("route_logits", torch.zeros(branch_count), persistent=True)

        if gate_mode == "learned":
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.register_buffer("gate", torch.tensor(float(gate_init)), persistent=True)
        self.gate_mode = gate_mode

        if self.use_pointwise:
            hidden = max(1, self.C // max(1, int(pointwise_ratio)))
            groups = gcd(max(1, int(pointwise_groups)), self.C)
            groups = gcd(groups, hidden) or 1
            self.pointwise = nn.Sequential(
                nn.Conv2d(self.C, hidden, 1, groups=groups, bias=False),
                nn.BatchNorm2d(hidden) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, self.C, 1, groups=groups, bias=False),
                nn.BatchNorm2d(self.C) if use_bn else nn.Identity(),
            )
        else:
            self.pointwise = nn.Identity()

    def _group_index(self, device: torch.device) -> torch.Tensor:
        return (torch.arange(self.C, device=device) // self.group_size).clamp_max(self.num_groups - 1)

    def _build_branch_kernel(
        self, axis_index: int, scale_index: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        group_index = self._group_index(device)
        alpha = self.alpha[axis_index, scale_index].to(device=device, dtype=dtype)[group_index]
        source_length = 2 * self.order + 1
        source = torch.zeros(self.C, source_length, device=device, dtype=dtype)
        source[:, self.order] = alpha[:, 0]
        for offset in range(1, self.order + 1):
            source[:, self.order - offset] = alpha[:, offset]
            source[:, self.order + offset] = alpha[:, offset]

        if self.shifted:
            # k = w * (delta_-1 + delta_+1), giving effective length 2M+3.
            kernel = F.pad(source, (1, 1))
            kernel = kernel[:, :-2] + kernel[:, 2:]
            # The expression above has source length. Pad the two outer shifted taps.
            left = source[:, -1:].clone()
            right = source[:, :1].clone()
            kernel = torch.cat((left, kernel, right), dim=1)
        else:
            kernel = source

        if self.normalize_l1:
            denom = kernel.abs().sum(dim=1, keepdim=True).clamp_min(1.0)
            kernel = kernel / denom
        return kernel

    def _pad(self, x: torch.Tensor, radius: int, axis_name: str, dilation: int) -> torch.Tensor:
        amount = radius * dilation
        pad_h = amount if axis_name == "h" else 0
        pad_w = amount if axis_name == "w" else 0
        if self.padding_mode == "constant":
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
        mode = self.padding_mode
        if mode == "reflect":
            h, w = x.shape[-2:]
            if (pad_h >= h and pad_h) or (pad_w >= w and pad_w):
                mode = "replicate"
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode)

    def _apply_branch(self, x: torch.Tensor, axis_name: str, dilation: int, kernel: torch.Tensor) -> torch.Tensor:
        length = int(kernel.shape[-1])
        radius = length // 2
        padded = self._pad(x, radius, axis_name, dilation)
        if axis_name == "h":
            weight = kernel.view(self.C, 1, length, 1)
            return F.conv2d(padded, weight, dilation=(dilation, 1), groups=self.C)
        weight = kernel.view(self.C, 1, 1, length)
        return F.conv2d(padded, weight, dilation=(1, dilation), groups=self.C)

    @property
    def convolution_calls_per_forward(self) -> int:
        return len(self.axis_names) * len(self.dilations)

    def routing_weights(self) -> torch.Tensor:
        count = self.convolution_calls_per_forward
        if self.routing == "fixed_average":
            return torch.full(
                (count,), 1.0 / count, device=self.route_logits.device, dtype=self.route_logits.dtype
            )
        return torch.softmax(self.route_logits / self.temperature, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.C:
            raise ValueError(f"expected BCHW with C={self.C}, got {tuple(x.shape)}")
        responses = []
        for axis_index, axis_name in enumerate(self.axis_names):
            for scale_index, dilation in enumerate(self.dilations):
                kernel = self._build_branch_kernel(axis_index, scale_index, x.device, x.dtype)
                responses.append(self._apply_branch(x, axis_name, dilation, kernel))
        weights = self.routing_weights().to(device=x.device, dtype=x.dtype)
        response = sum(weight * branch for weight, branch in zip(weights, responses))
        response = self.pointwise(response)
        return x + self.residual_scale * self.gate.to(device=x.device, dtype=x.dtype) * response

    def extra_repr(self) -> str:
        return (
            f"C={self.C}, axis={self.axis}, order={self.order}, dilations={self.dilations}, "
            f"group_size={self.group_size}, shifted={self.shifted}, routing={self.routing}"
        )


__all__ = ["ReviewerAxialRoutingAdapter"]
