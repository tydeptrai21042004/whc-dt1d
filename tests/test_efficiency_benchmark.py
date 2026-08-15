from __future__ import annotations

import argparse
import math
import statistics

import torch

from tools import benchmark_reviewer_efficiency as benchmark


def test_cached_dt1d_benchmark_build_uses_cache():
    ns = argparse.Namespace(
        backbone="resnet18",
        num_classes=5,
        input_size=64,
        batch_size=1,
        cache_dt1d=True,
    )
    model = benchmark.build("dt1d", ns, torch.device("cpu"))
    adapters = [module for module in model.modules() if getattr(module, "is_dt1d_adapter", False)]
    assert adapters
    assert all(module.cache_kernel for module in adapters)
    assert all(module._cached_kernels.numel() > 0 for module in adapters)


def test_nearest_rank_p95_is_not_below_median():
    values = sorted([7.0, 12.0])
    index = min(len(values) - 1, max(0, math.ceil(0.95 * len(values)) - 1))
    assert values[index] >= statistics.median(values)


def test_efficiency_method_registry_includes_reviewer_baselines():
    parsed = benchmark.parse_methods("dt1d,routing_reference,plain_axial,conv,residual,bam,ssf,lora_conv,bitfit,linear,full")
    assert parsed == ["dt1d", "routing_reference", "plain_axial", "conv", "residual", "bam", "ssf", "lora_conv", "bitfit", "linear", "full"]
