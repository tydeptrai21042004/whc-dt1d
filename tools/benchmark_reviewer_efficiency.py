#!/usr/bin/env python3
"""Matched efficiency benchmark for the revised DT1D reviewer response.

The script builds complete torchvision backbones through the same code path as
training and exports parameters, task-specific storage, FLOPs when fvcore is
available, median/p95 latency, throughput, and peak inference memory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import main as training



METHOD_ARGS = {
    "dt1d": ["--tuning_method", "dt1d"],
    "routing_reference": [
        "--tuning_method", "reviewer_routing",
        "--reviewer_shifted", "True",
        "--reviewer_routing", "learned_softmax",
        "--reviewer_dilations", "1,2,4",
        "--reviewer_group_size", "16",
    ],
    "direct_symmetric": [
        "--tuning_method", "reviewer_routing",
        "--reviewer_shifted", "False",
        "--reviewer_routing", "learned_softmax",
        "--reviewer_dilations", "1,2,4",
        "--reviewer_group_size", "16",
    ],
    "plain_axial": ["--tuning_method", "plain_axial", "--axial_kernel_mode", "unrestricted"],
    "linear": ["--tuning_method", "linear"],
    "full": ["--tuning_method", "full"],
    "conv": ["--tuning_method", "conv", "--adapt_size", "4"],
    "residual": ["--tuning_method", "residual", "--ra_mode", "parallel", "--ra_reduction", "16"],
    "bam": ["--tuning_method", "bam", "--bam_reduction", "16", "--bam_insert", "stage"],
    "ssf": ["--tuning_method", "ssf"],
    "bitfit": ["--tuning_method", "bitfit"],
    "lora_conv": ["--tuning_method", "lora_conv", "--lora_r", "8", "--lora_alpha", "16"],
}


def parse_methods(text: str) -> list[str]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    unknown = [v for v in values if v not in METHOD_ARGS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown methods {unknown}; available={sorted(METHOD_ARGS)}")
    return values


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def task_state(model: torch.nn.Module, *, head: bool | None = None) -> dict[str, torch.Tensor]:
    result = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = training._is_head_param(name)
        if head is None or is_head == head:
            result[name] = parameter.detach().cpu()
    return result


def serialized_mb(state: dict[str, torch.Tensor]) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(state, handle.name)
        handle.flush()
        return Path(handle.name).stat().st_size / (1024**2)


def build(method: str, ns: argparse.Namespace, device: torch.device):
    parser = training.get_args_parser()
    argv = [
        "--backbone", ns.backbone,
        "--weights", "none",
        "--pretrained", "False",
        "--nb_classes", str(ns.num_classes),
        "--input_size", str(ns.input_size),
        "--device", str(device),
        "--use_amp", "False",
        "--profile_efficiency", "False",
        "--save_ckpt", "False",
        "--final_test", "False",
        "--batch_size", str(ns.batch_size),
        "--num_workers", "0",
        "--dt_padding", "replicate",
        "--dt_axis", "hw",
        "--dt_alpha_group", "16",
        "--dt_detail_basis", "orth",
        "--dt_detail_components", "offset4",
        "--dt_active_offsets", "1,2,4,8",
        *METHOD_ARGS[method],
    ]
    if ns.cache_dt1d and method == "dt1d":
        argv.extend(["--dt_cache_kernel", "True"])
    args = training.canonicalize_args(parser.parse_args(argv))
    model, ids = training.build_model_for_experiment(args)
    model = training.set_trainability_policy(model, args, ids).to(device).eval()
    if ns.cache_dt1d:
        for module in model.modules():
            if hasattr(module, "prepare_for_inference"):
                if hasattr(module, "cache_kernel"):
                    module.cache_kernel = True
                module.prepare_for_inference(device=device, dtype=torch.float32)
    return model


def benchmark(model, x, warmup: int, iterations: int, device: torch.device) -> tuple[list[float], float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(iterations):
            sync(device)
            start = time.perf_counter()
            model(x)
            sync(device)
            times.append((time.perf_counter() - start) * 1000.0)
        peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    return times, peak


def flop_count(model, x) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(model, x).total() / 1e9)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=parse_methods, default=parse_methods("dt1d,routing_reference,direct_symmetric,plain_axial,linear,full,conv,residual,bam,ssf,bitfit,lora_conv"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dt1d", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reviewer_efficiency")
    ns = parser.parse_args()

    device = torch.device(ns.device if not ns.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    ns.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ns.methods:
        model = build(method, ns, device)
        x = torch.randn(ns.batch_size, 3, ns.input_size, ns.input_size, device=device)
        times, peak = benchmark(model, x, ns.warmup, ns.iterations, device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        head_state = task_state(model, head=True)
        adapter_state = task_state(model, head=False)
        sorted_times = sorted(times)
        # Nearest-rank percentile: p95 is never below the median merely because
        # the quick smoke benchmark uses only a few iterations.
        p95_index = min(len(sorted_times) - 1, max(0, math.ceil(0.95 * len(sorted_times)) - 1))
        median_batch = statistics.median(times)
        row = {
            "method": method,
            "device": str(device),
            "backbone": ns.backbone,
            "input_size": ns.input_size,
            "batch_size": ns.batch_size,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "adapter_parameters": sum(v.numel() for v in adapter_state.values()),
            "head_parameters": sum(v.numel() for v in head_state.values()),
            "adapter_storage_fp32_mb": sum(v.numel() for v in adapter_state.values()) * 4 / (1024**2),
            "adapter_storage_fp16_mb": sum(v.numel() for v in adapter_state.values()) * 2 / (1024**2),
            "serialized_adapter_mb": serialized_mb(adapter_state),
            "serialized_head_mb": serialized_mb(head_state),
            "flops_g": flop_count(model, x),
            "latency_median_ms_per_batch": median_batch,
            "latency_p95_ms_per_batch": sorted_times[p95_index],
            "latency_median_ms_per_image": median_batch / ns.batch_size,
            "throughput_images_per_second": 1000.0 * ns.batch_size / median_batch,
            "peak_inference_memory_mb": peak,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))
        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (ns.output / "efficiency.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (ns.output / "efficiency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {ns.output / 'efficiency.json'} and {ns.output / 'efficiency.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
