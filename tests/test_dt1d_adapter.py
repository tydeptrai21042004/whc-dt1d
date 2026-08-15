from __future__ import annotations

import torch

from models.dt1d_adapter import DT1DAdapter
from models.dt1d_ablation_adapter import DT1DAblationAdapter


def test_canonical_dt1d_is_frozen_final_architecture():
    model = DT1DAdapter(C=64)
    assert model.method_name == "DT1D-Adapter"
    assert model.proposal_name == "DT1D-Adapter"
    assert model.architecture_name == "R124-P2-G16-Axis-LearnedGate"
    assert model.axis == "hw"
    assert model.alpha_group == 16
    assert model.active_offsets == (1, 2, 4)
    assert model.detail_components == "offset4"
    assert model.shift_p == 2
    assert model.shift_lambda_mode == "learned"
    assert model.shift_lambda_scope == "axis"
    assert model.shift_lambda_max == 0.5
    assert model.project_l1 is True
    assert model.joint_l1_cap == 1.0
    assert model.gate_mode == "learned"
    assert model.use_pointwise is False
    assert model.padding_mode == "replicate"
    assert model.effective_kernel_size == 13
    assert model.convolution_calls_per_forward == 2


def test_canonical_forward_backward_and_shift_gate_gradients():
    torch.manual_seed(7)
    model = DT1DAdapter(C=16).train()
    x = torch.randn(2, 16, 17, 19, requires_grad=True)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.shift_theta.grad is not None
    assert torch.isfinite(model.shift_theta.grad).all()
    assert model.gate.grad is not None
    assert torch.isfinite(model.gate.grad).all()


def test_joint_l1_projection_caps_both_axes_together():
    model = DT1DAdapter(C=16)
    with torch.no_grad():
        model.quotient_beta.normal_(0, 3)
        model.detail_eta.normal_(0, 3)
    kernels = model.build_kernels(torch.device("cpu"), torch.float32)
    joint = kernels.squeeze(2).abs().sum(dim=-1).sum(dim=0)
    assert torch.all(joint <= 1.000001)


def test_ablation_class_can_disable_projection_and_weighted_shift():
    model = DT1DAblationAdapter(
        C=16,
        project_l1=False,
        shift_lambda_mode="off",
        shift_lambda_init=0.0,
    )
    assert model.is_reviewer_ablation
    assert model.project_l1 is False
    assert model.weighting_active is False
    assert model.effective_kernel_size == model.base_kernel_size == 9


def test_ablation_gate_off_is_identity_safe():
    model = DT1DAblationAdapter(C=16, gate_mode="fixed", gate_init=0.0)
    x = torch.randn(2, 16, 11, 13)
    assert torch.allclose(model(x), x, atol=1e-7)


def test_cached_and_uncached_outputs_match():
    model = DT1DAdapter(C=16, cache_kernel=True).eval()
    x = torch.randn(2, 16, 17, 17)
    uncached = model(x)
    model.prepare_for_inference(x.device, x.dtype)
    cached = model(x)
    assert torch.allclose(uncached, cached, atol=1e-7)
    model.train()
    assert model._cached_kernels.numel() == 0


def test_resnet18_adapter_parameter_count_matches_validation_target():
    channels = [64, 64, 128, 128, 256, 256, 512, 512]
    canonical = sum(sum(p.numel() for p in DT1DAdapter(c).parameters()) for c in channels)
    assert canonical == 1224
