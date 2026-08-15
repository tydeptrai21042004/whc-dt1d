from __future__ import annotations

import copy

import torch
from torchvision.models import resnet18

import main as training


def _args(method: str):
    args = training.get_args_parser().parse_args([])
    args.backbone = "resnet18"
    args.tuning_method = method
    args.freeze_backbone = True
    args.dt_cache_kernel = False
    if method == "plain_axial":
        args.axial_kernel_mode = "unrestricted"
        args.axial_project_l1 = False
    if method == "reviewer_routing":
        args.reviewer_shifted = True
        args.reviewer_routing = "learned_softmax"
        args.reviewer_dilations = "1,2,4"
        args.reviewer_group_size = 16
    return args


def _build(method: str):
    args = _args(method)
    model = resnet18(weights=None)
    model, adapter_ids = training._add_adapters(model, args)
    model = training.set_trainability_policy(model, args, extra_adapter_param_ids=adapter_ids)
    return model, args


def _adapter_named(model):
    return [(name, p) for name, p in model.named_parameters() if "pet_adapter" in name and p.requires_grad]


def test_hook_based_dt_plain_and_reviewer_adapters_affect_backbone_and_receive_gradients():
    torch.manual_seed(7)
    x = torch.randn(2, 3, 32, 32)
    for method in ("dt1d", "dt1d_ablation", "plain_axial", "reviewer_routing"):
        model, _args_obj = _build(method)
        params = _adapter_named(model)
        assert params, f"{method}: no trainable pet_adapter tensors"

        # Perturbing an adapter parameter must change the full-network output.
        model.eval()
        with torch.no_grad():
            y0 = model(x).clone()
            params[0][1].add_(0.25)
            y1 = model(x).clone()
            params[0][1].sub_(0.25)
        assert not torch.equal(y0, y1), f"{method}: adapter perturbation does not affect backbone output"

        # Every trainable adapter tensor must be in the autograd graph; at least
        # one must receive a non-zero gradient for a generic loss.
        model.train()
        model.zero_grad(set_to_none=True)
        y = model(x)
        y.float().square().mean().backward()
        missing = [name for name, p in params if p.grad is None]
        assert not missing, f"{method}: disconnected adapter tensors: {missing}"
        assert all(torch.isfinite(p.grad).all() for _, p in params)
        assert any(p.grad.abs().sum().item() > 0.0 for _, p in params)
