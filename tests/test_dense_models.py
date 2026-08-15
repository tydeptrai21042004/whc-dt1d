from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch

from dense_prediction.models import build_dense_model, configure_dense_trainability
from models.dt1d_adapter import DT1DAdapter


def make_args(pipeline, task, method, num_classes, vit_variant="tiny"):
    return SimpleNamespace(
        pipeline=pipeline, task=task, tuning_method=method, pretrained=False,
        num_classes=num_classes, input_size=64, adapter_stages="4,7,13,16",
        adapter_reduction=16, vit_variant=vit_variant, detector_variant="tiny",
        dt_axis="hw", dt_gate_mode="learned", dt_padding="replicate",
        dt_use_pointwise=False, cache_dt1d=True,
        axial_kernel_mode="unrestricted", axial_project_l1=False,
        lora_rank=4, lora_alpha=1.0, lora_target="1x1",
    )


@pytest.mark.parametrize("method", ["dt1d", "plain_axial", "full", "linear", "bitfit", "conv_adapter", "residual_adapter", "ssf", "bam", "lora_conv"])
def test_deeplab_dense_methods_build_and_select_trainable_parameters(method):
    args = make_args("deeplab_mobilenet_v3", "binary_segmentation", method, 1)
    model = build_dense_model(args)
    counts = configure_dense_trainability(model, args)
    assert counts["total_params"] > 0 and counts["trainable_params"] > 0
    if method == "dt1d":
        assert sum(isinstance(module, DT1DAdapter) for module in model.modules()) == 4
        model.eval()
        output = model(torch.randn(1, 3, 64, 64))["out"]
        assert output.shape == (1, 1, 64, 64)


@pytest.mark.parametrize("method", ["dt1d", "full", "linear", "bitfit"])
def test_vit_dense_reported_methods_build(method):
    args = make_args("vit_b16_dense", "binary_segmentation", method, 1)
    model = build_dense_model(args)
    counts = configure_dense_trainability(model, args)
    output = model(torch.randn(1, 3, 64, 64))["out"]
    assert output.shape == (1, 1, 64, 64)
    assert counts["trainable_params"] > 0


@pytest.mark.parametrize("method", ["dt1d", "full", "head_only", "conv_adapter", "bam"])
def test_fasterrcnn_reported_methods_build(method):
    args = make_args("fasterrcnn_mobilenet_v3_fpn", "detection", method, 2)
    model = build_dense_model(args)
    counts = configure_dense_trainability(model, args)
    assert counts["trainable_params"] > 0
    if method == "dt1d":
        model.eval()
        outputs = model([torch.rand(3, 64, 64)])
        assert isinstance(outputs, list)
        assert {"boxes", "labels", "scores"}.issubset(outputs[0])
