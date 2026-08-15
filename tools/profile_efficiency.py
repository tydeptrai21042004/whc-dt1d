# tools/profile_efficiency.py
"""Utility functions for FLOPs, parameter, latency, and memory profiling."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Dict, Optional

import torch

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:  # optional dependency
    FlopCountAnalysis = None

try:
    from thop import profile as thop_profile
except Exception:  # optional dependency
    thop_profile = None


def count_params(model: torch.nn.Module) -> Dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_param_ratio": float(trainable / max(total, 1)),
    }


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def _functional_adapter_macs(model: torch.nn.Module, sample: torch.Tensor) -> int:
    """Count MACs from the axial F.conv2d calls inside adapter modules.

    THOP/module-hook counters do not see functional convolutions. This helper
    measures only those axial calls, so ordinary Conv2d/Linear operations can
    be counted separately without double-counting pointwise modules.
    """
    total = 0
    handles = []

    def hook(module, inputs, _output):
        nonlocal total
        x = inputs[0]
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            return
        n, c, h, w = map(int, x.shape)
        cls = module.__class__.__name__
        if getattr(module, "is_dt1d_adapter", False) or cls == "PlainAxialDepthwiseAdapter":
            axes = len(getattr(module, "axis_names", ()))
            kernel_len = int(getattr(module, "effective_kernel_size", 17))
            total += n * c * h * w * kernel_len * axes
        elif cls == "ReviewerAxialRoutingAdapter":
            axes = len(getattr(module, "axis_names", ()))
            dilations = len(getattr(module, "dilations", ()))
            order = int(getattr(module, "order", 0))
            shifted = bool(getattr(module, "shifted", False))
            kernel_len = 2 * order + 3 if shifted else 2 * order + 1
            total += n * c * h * w * kernel_len * axes * dilations

    for module in model.modules():
        if (
            getattr(module, "is_dt1d_adapter", False)
            or module.__class__.__name__ in {"PlainAxialDepthwiseAdapter", "ReviewerAxialRoutingAdapter"}
        ):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            _ = model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return int(total)


def _native_module_macs(model: torch.nn.Module, sample: torch.Tensor) -> int:
    """Dependency-free Conv2d/Linear MAC counter used as a final fallback."""
    total = 0
    handles = []

    def conv_hook(module: torch.nn.Conv2d, _inputs, output):
        nonlocal total
        if not isinstance(output, torch.Tensor):
            return
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total += int(output.numel()) * int(kernel_ops)

    def linear_hook(module: torch.nn.Linear, _inputs, output):
        nonlocal total
        if isinstance(output, torch.Tensor):
            total += int(output.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            _ = model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return int(total)


@torch.no_grad()
def measure_compute(model: torch.nn.Module, device: torch.device, input_size: int = 224) -> Dict[str, Optional[float | int | str]]:
    """Measure compute with explicit, reproducible fallback conventions.

    Priority: fvcore FLOPs; THOP module MACs plus analytic axial-functional
    MACs; dependency-free Conv2d/Linear hooks plus the same analytic axial
    functional MACs. For MAC-based paths we report FLOPs = 2 * MACs and save
    ``compute_source`` so the manuscript can state the convention.
    """
    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)

    if FlopCountAnalysis is not None:
        try:
            flops = int(FlopCountAnalysis(model, x).total())
            model.train(was_training)
            return {
                "flops": flops,
                "flops_g": float(flops / 1e9),
                "macs": None,
                "macs_g": None,
                "compute_source": "fvcore",
            }
        except Exception as exc:
            print(f"[Warn] fvcore FLOPs computation failed: {exc}")

    functional_macs = _functional_adapter_macs(model, x)

    if thop_profile is not None:
        try:
            module_macs, _params = thop_profile(model, inputs=(x,), verbose=False)
            macs = int(module_macs) + int(functional_macs)
            flops = int(2 * macs)
            model.train(was_training)
            return {
                "flops": flops,
                "flops_g": float(flops / 1e9),
                "macs": macs,
                "macs_g": float(macs / 1e9),
                "functional_adapter_macs": int(functional_macs),
                "compute_source": "thop_plus_axial_functional_2x_macs",
            }
        except Exception as exc:
            print(f"[Warn] THOP MAC/FLOPs fallback failed: {exc}")

    module_macs = _native_module_macs(model, x)
    macs = int(module_macs) + int(functional_macs)
    flops = int(2 * macs)
    model.train(was_training)
    return {
        "flops": flops,
        "flops_g": float(flops / 1e9),
        "macs": macs,
        "macs_g": float(macs / 1e9),
        "functional_adapter_macs": int(functional_macs),
        "compute_source": "native_conv_linear_plus_axial_functional_2x_macs",
    }


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 32,
    warmup: int = 20,
    iters: int = 100,
    use_amp: bool = False,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    for _ in range(max(0, warmup)):
        with _amp_context(device, use_amp):
            _ = model(x)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(max(1, iters)):
        with _amp_context(device, use_amp):
            _ = model(x)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    ms_per_batch = elapsed * 1000.0 / max(1, iters)
    ms_per_image = ms_per_batch / batch_size
    model.train(was_training)
    return {
        "latency_ms_per_batch": float(ms_per_batch),
        "latency_ms_per_image": float(ms_per_image),
        "fps": float(1000.0 / max(ms_per_image, 1e-12)),
    }


@torch.no_grad()
def measure_peak_inference_memory(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 32,
    use_amp: bool = False,
) -> Optional[float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    was_training = model.training
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with _amp_context(device, use_amp):
        _ = model(x)
    torch.cuda.synchronize(device)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    model.train(was_training)
    return float(peak_mb)


def profile_model(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 32,
    use_amp: bool = False,
    warmup: int = 20,
    iters: int = 100,
) -> Dict:
    profile = count_params(model)
    profile.update(measure_compute(model, device, input_size=input_size))
    profile.update(
        measure_latency(
            model,
            device,
            input_size=input_size,
            batch_size=batch_size,
            warmup=warmup,
            iters=iters,
            use_amp=use_amp,
        )
    )
    profile["peak_inference_memory_mb"] = measure_peak_inference_memory(
        model,
        device,
        input_size=input_size,
        batch_size=batch_size,
        use_amp=use_amp,
    )
    profile["profile_batch_size"] = int(batch_size)
    profile["input_size"] = int(input_size)
    return profile


def save_profile(profile: Dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, default=str)
