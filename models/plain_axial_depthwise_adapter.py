"""Reviewer baseline for the earlier plain axial depthwise 1D convolution.

This module is intentionally separate from DT1D.  It supplies the direct
side-by-side baseline and the staged axial-kernel ablations requested by the
reviewers without retaining any earlier DT1D implementation.
"""
from __future__ import annotations

from math import gcd
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_KERNEL_MODES = {"unrestricted", "direct_symmetric", "reduced_symmetric"}


class PlainAxialDepthwiseAdapter(nn.Module):
    """Axial depthwise 1D-convolution baseline with matched execution shape.

    ``unrestricted`` learns 17 taps per channel and axis.
    ``direct_symmetric`` learns 9 independent taps and mirrors the side taps.
    ``reduced_symmetric`` learns 5 taps at offsets 0, ±1, ±2, ±4, ±8.
    """

    def __init__(
        self,
        C: int,
        *,
        axis: str = "hw",
        kernel_mode: str = "unrestricted",
        residual_scale: float = 1.0,
        gate_init: float = 0.01,
        gate_mode: str = "learned",
        padding_mode: str = "replicate",
        use_pointwise: bool = False,
        pointwise_ratio: int = 32,
        pointwise_groups: int = 4,
        use_bn: bool = False,
        project_l1: bool = False,
    ) -> None:
        super().__init__()
        if C <= 0:
            raise ValueError("C must be positive")
        if axis not in {"h", "w", "hw"}:
            raise ValueError("axis must be h, w, or hw")
        if kernel_mode not in _KERNEL_MODES:
            raise ValueError(f"kernel_mode must be one of {_KERNEL_MODES}")
        if gate_mode not in {"learned", "fixed"}:
            raise ValueError("gate_mode must be learned or fixed")
        if padding_mode not in {"reflect", "replicate", "zeros", "constant"}:
            raise ValueError("unsupported padding mode")

        self.C = int(C)
        self.axis = axis
        self.axis_names = tuple(a for a in ("h", "w") if a in axis)
        self.num_axes = len(self.axis_names)
        self.kernel_mode = kernel_mode
        self.residual_scale = float(residual_scale)
        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.gate_mode = gate_mode
        self.project_l1 = bool(project_l1)
        self.use_pointwise = bool(use_pointwise)

        ncoef = {"unrestricted": 17, "direct_symmetric": 9, "reduced_symmetric": 5}[kernel_mode]
        self.coefficients = nn.Parameter(torch.zeros(self.num_axes, self.C, ncoef))
        with torch.no_grad():
            center = 8 if kernel_mode == "unrestricted" else 0
            self.coefficients[..., center].fill_(1.0 / max(1, self.num_axes))

        if gate_mode == "learned":
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.register_buffer("gate", torch.tensor(float(gate_init)), persistent=True)

        if self.use_pointwise:
            hidden = max(1, self.C // max(1, int(pointwise_ratio)))
            groups = max(1, min(int(pointwise_groups), self.C, hidden))
            groups = gcd(groups, self.C)
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

        self.is_axial_baseline = True

    def build_kernels(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coef = self.coefficients.to(device=device, dtype=dtype)
        if self.kernel_mode == "unrestricted":
            kernel = coef
        elif self.kernel_mode == "direct_symmetric":
            kernel = torch.zeros(self.num_axes, self.C, 17, device=device, dtype=dtype)
            kernel[..., 8] = coef[..., 0]
            for offset in range(1, 9):
                kernel[..., 8 - offset] = coef[..., offset]
                kernel[..., 8 + offset] = coef[..., offset]
        else:
            kernel = torch.zeros(self.num_axes, self.C, 17, device=device, dtype=dtype)
            kernel[..., 8] = coef[..., 0]
            for index, offset in enumerate((1, 2, 4, 8), start=1):
                kernel[..., 8 - offset] = coef[..., index]
                kernel[..., 8 + offset] = coef[..., index]
        if self.project_l1:
            joint_l1 = kernel.abs().sum(dim=-1).sum(dim=0)
            scale = torch.maximum(joint_l1, torch.ones_like(joint_l1))
            kernel = kernel / scale.view(1, self.C, 1)
        return kernel.unsqueeze(2)

    def _pad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if self.padding_mode == "constant":
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
        mode = self.padding_mode
        if mode == "reflect":
            h, w = x.shape[-2:]
            if (pad_h >= h and pad_h > 0) or (pad_w >= w and pad_w > 0):
                mode = "replicate"
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode)

    def _conv_axis(self, x: torch.Tensor, axis_name: str, kernel: torch.Tensor) -> torch.Tensor:
        if axis_name == "h":
            x = self._pad(x, 8, 0)
            return F.conv2d(x, kernel.view(self.C, 1, 17, 1), groups=self.C)
        x = self._pad(x, 0, 8)
        return F.conv2d(x, kernel.view(self.C, 1, 1, 17), groups=self.C)

    @property
    def convolution_calls_per_forward(self) -> int:
        return self.num_axes

    def parameter_count_breakdown(self) -> Dict[str, int]:
        spatial = self.coefficients.numel()
        gate = self.gate.numel() if isinstance(self.gate, nn.Parameter) else 0
        pointwise = sum(p.numel() for p in self.pointwise.parameters())
        return {
            "spatial": int(spatial),
            "learned_gate": int(gate),
            "pointwise": int(pointwise),
            "total": int(spatial + gate + pointwise),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.C:
            raise ValueError(f"Expected BCHW input with C={self.C}, got {tuple(x.shape)}")
        kernels = self.build_kernels(x.device, x.dtype)
        response = torch.zeros_like(x)
        for axis_index, axis_name in enumerate(self.axis_names):
            response = response + self._conv_axis(x, axis_name, kernels[axis_index])
        response = self.pointwise(response)
        return x + self.residual_scale * self.gate.to(dtype=x.dtype, device=x.device) * response


__all__ = ["PlainAxialDepthwiseAdapter"]
