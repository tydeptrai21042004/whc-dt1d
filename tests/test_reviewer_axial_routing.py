from __future__ import annotations

import torch

from models.reviewer_axial_routing_adapter import ReviewerAxialRoutingAdapter


def test_reviewer_routing_variants_shape_backward_and_branch_count():
    variants = [
        dict(shifted=False, routing="fixed_average", dilations="1,2,4", group_size=16),
        dict(shifted=True, routing="fixed_average", dilations="1,2,4", group_size=16),
        dict(shifted=True, routing="fixed_average", dilations="1,2,4", group_size=1),
        dict(shifted=True, routing="learned_softmax", dilations="1,2,4", group_size=16),
        dict(shifted=True, routing="fixed_average", dilations="1", group_size=16),
    ]
    for kwargs in variants:
        module = ReviewerAxialRoutingAdapter(16, axis="hw", order=2, **kwargs)
        x = torch.randn(2, 16, 17, 19, requires_grad=True)
        y = module(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        y.square().mean().backward()
        assert any(p.grad is not None for p in module.parameters() if p.requires_grad)
        assert module.convolution_calls_per_forward == 2 * len(module.dilations)


def test_fixed_and_learned_routing_weights_are_valid():
    fixed = ReviewerAxialRoutingAdapter(8, routing="fixed_average", dilations="1,2")
    learned = ReviewerAxialRoutingAdapter(8, routing="learned_softmax", dilations="1,2")
    for module in (fixed, learned):
        weights = module.routing_weights()
        assert weights.numel() == module.convolution_calls_per_forward
        assert torch.all(weights >= 0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))
    assert not isinstance(fixed.route_logits, torch.nn.Parameter)
    assert isinstance(learned.route_logits, torch.nn.Parameter)


def test_shifted_and_direct_kernel_support_differ():
    direct = ReviewerAxialRoutingAdapter(4, order=2, shifted=False, dilations="1")
    shifted = ReviewerAxialRoutingAdapter(4, order=2, shifted=True, dilations="1")
    kd = direct._build_branch_kernel(0, 0, torch.device("cpu"), torch.float32)
    ks = shifted._build_branch_kernel(0, 0, torch.device("cpu"), torch.float32)
    assert kd.shape[-1] == 5
    assert ks.shape[-1] == 7
