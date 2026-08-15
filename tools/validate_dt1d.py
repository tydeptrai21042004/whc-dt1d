#!/usr/bin/env python3
"""Structural/numerical validation for the single canonical DT1D-Adapter."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from models.dt1d_adapter import DT1DAdapter


def adapter_params(module):
    return sum(p.numel() for p in module.parameters())


def weighted_convolution_error(trials: int = 100) -> float:
    torch.manual_seed(123)
    worst = 0.0
    for _ in range(trials):
        k = torch.randn(9, dtype=torch.float64)
        lam = float(torch.rand((), dtype=torch.float64) - 0.5)
        centered = F.pad(k, (2, 2))
        shifted = torch.zeros(13, dtype=torch.float64)
        shifted[:9] += k
        shifted[4:13] += k
        direct = (1.0 - lam) * centered + 0.5 * lam * shifted
        w = torch.zeros(5, dtype=torch.float64)
        w[0] = lam / 2
        w[2] = 1.0 - lam
        w[4] = lam / 2
        explicit = F.conv1d(k.view(1, 1, -1), w.flip(0).view(1, 1, -1), padding=4).view(-1)
        worst = max(worst, float((direct - explicit).abs().max()))
    return worst


def spectral_error(trials: int = 100) -> float:
    torch.manual_seed(456)
    worst = 0.0
    nfft = 256
    for _ in range(trials):
        k = torch.randn(9, dtype=torch.float64)
        lam = float(torch.rand((), dtype=torch.float64) - 0.5)
        centered = F.pad(k, (2, 2))
        shifted = torch.zeros(13, dtype=torch.float64)
        shifted[:9] += k
        shifted[4:13] += k
        kw = (1 - lam) * centered + 0.5 * lam * shifted
        K = torch.fft.fft(F.pad(k, (0, nfft - 9)))
        KW = torch.fft.fft(F.pad(kw, (0, nfft - 13)))
        omega = 2 * math.pi * torch.arange(nfft, dtype=torch.float64) / nfft
        predicted = torch.exp(-2j * omega) * (1 - lam + lam * torch.cos(2 * omega)) * K
        worst = max(worst, float((KW - predicted).abs().max()))
    return worst


def resnet18_adapter_total() -> int:
    channels = [64, 64, 128, 128, 256, 256, 512, 512]
    return sum(adapter_params(DT1DAdapter(c)) for c in channels)


def main() -> int:
    torch.manual_seed(0)
    m = DT1DAdapter(64)
    with torch.no_grad():
        m.quotient_beta.normal_(0, 2)
        m.detail_eta.normal_(0, 2)
        m.shift_theta.copy_(torch.tensor([0.4, -0.3]))
    kernels = m.build_kernels(torch.device("cpu"), torch.float32)
    joint = kernels.abs().sum(-1).sum(0).squeeze(-1)
    x = torch.randn(2, 64, 16, 16)
    y = m(x)
    y.square().mean().backward()
    final = resnet18_adapter_total()
    result = {
        "proposal": "DT1D-Adapter",
        "architecture": m.architecture_name,
        "fixed_architecture": {
            "axes": "hw",
            "group_size": 16,
            "base_offsets": [1, 2, 4],
            "detail": "psi4",
            "shift_p": 2,
            "lambda_scope": "axis",
            "lambda_mode": "learned",
            "lambda_init": 0.0,
            "lambda_max": 0.5,
            "joint_l1_cap": 1.0,
            "gate_mode": "learned",
            "gate_init": 0.01,
            "pointwise": False,
            "padding": "replicate",
            "effective_kernel": 13,
            "conv_calls": 2,
        },
        "numerical": {
            "joint_l1_max": float(joint.detach().max()),
            "lambda_grad_l1": float(m.shift_theta.grad.abs().sum()),
            "gate_grad_abs": float(m.gate.grad.abs()),
            "weighted_convolution_max_error": weighted_convolution_error(),
            "spectral_identity_max_error": spectral_error(),
        },
        "parameter_check": {
            "c64": adapter_params(DT1DAdapter(64)),
            "resnet18_dt1d_adapter": final,
        },
    }
    if result["numerical"]["joint_l1_max"] > 1.00001:
        raise SystemExit("Joint L1 cap validation failed")
    if result["numerical"]["weighted_convolution_max_error"] > 1e-12:
        raise SystemExit("Weighted convolution identity failed")
    if result["numerical"]["spectral_identity_max_error"] > 1e-11:
        raise SystemExit("Spectral identity failed")
    if result["numerical"]["lambda_grad_l1"] <= 0 or result["numerical"]["gate_grad_abs"] <= 0:
        raise SystemExit("Learnable shift/gate did not receive gradients")
    out = ROOT / "reproducibility" / "dt1d_structural_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
