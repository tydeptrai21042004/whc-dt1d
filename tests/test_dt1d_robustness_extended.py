from __future__ import annotations

import io
import math

import pytest
import torch
import torch.nn.functional as F

from models._dt1d_base import _BaseDT1DAdapter, _parse_offsets
from models.dt1d_adapter import DT1DAdapter
from models.dt1d_ablation_adapter import DT1DAblationAdapter
from proposal_contract import load_spec, proposal_fingerprint


@pytest.mark.parametrize("bad", ["1,2,4,8,99", [1, 2, 4, 99], "99"])
def test_invalid_offsets_fail_loudly_instead_of_being_silently_dropped(bad):
    with pytest.raises(ValueError):
        _parse_offsets(bad)


@pytest.mark.parametrize("channels", [1, 2, 15, 16, 17, 24, 31, 32, 64, 768])
def test_canonical_parameter_count_formula_for_edge_channel_counts(channels):
    m = DT1DAdapter(channels)
    groups = math.ceil(channels / 16)
    # 2 axes * groups * (4 shared coefficients + 1 detail amplitude)
    # + 2 axis lambdas + 1 learned residual gate.
    expected = 10 * groups + 3
    assert sum(p.numel() for p in m.parameters()) == expected
    assert m.parameter_count_breakdown()["total"] == expected


def test_singleton_group_disables_zero_mean_detail_without_nan():
    m = DT1DAdapter(1)
    assert m.valid_contrast_group.tolist() == [0.0]
    assert torch.count_nonzero(m.channel_contrast).item() == 0
    with torch.no_grad():
        m.detail_eta.fill_(1000.0)
    k = m.build_kernels(torch.device("cpu"), torch.float32)
    assert torch.isfinite(k).all()


@pytest.mark.parametrize("channels", [17, 24, 31, 33])
def test_partial_remainder_groups_keep_balanced_zero_mean_contrast(channels):
    m = DT1DAdapter(channels)
    for group in range(m.num_alpha_groups):
        start = group * m.alpha_group
        end = min(channels, start + m.alpha_group)
        chunk = m.channel_contrast[start:end]
        if len(chunk) >= 2:
            assert m.valid_contrast_group[group].item() == 1.0
            assert abs(float(chunk.sum())) < 1e-6
            assert torch.count_nonzero(chunk).item() == len(chunk)
        else:
            assert m.valid_contrast_group[group].item() == 0.0


def test_zero_lambda_embeds_k9_exactly_in_center_of_k13():
    m = DT1DAdapter(16)
    base17 = m.build_unprojected_kernels(torch.device("cpu"), torch.float32).squeeze(2)
    # Learned lambda initializes to exactly zero, so only the centered K9 support is nonzero.
    assert base17.shape[-1] == 13
    assert torch.count_nonzero(base17[..., :2]).item() == 0
    assert torch.count_nonzero(base17[..., -2:]).item() == 0


def test_weighted_shift_preserves_dc_sum_before_projection():
    torch.manual_seed(101)
    m = DT1DAblationAdapter(C=16, project_l1=False)
    with torch.no_grad():
        m.quotient_beta.normal_(0.0, 0.2)
        m.detail_eta.normal_(0.0, 0.2)
        m.shift_theta[:] = torch.tensor([0.8, -0.7])
    # Recover the cropped K9 base from the internal pre-weighting construction.
    full17 = _BaseDT1DAdapter.build_kernels(m, torch.device("cpu"), torch.float32, project=False).squeeze(2)
    base = m._crop_base_kernel(full17)
    shifted = m.build_unprojected_kernels(torch.device("cpu"), torch.float32).squeeze(2)
    assert torch.allclose(base.sum(-1), shifted.sum(-1), atol=1e-6, rtol=1e-6)


def test_extreme_trainable_values_still_produce_finite_projected_kernels():
    m = DT1DAdapter(33)
    with torch.no_grad():
        m.quotient_beta.uniform_(-1e4, 1e4)
        m.detail_eta.uniform_(-1e4, 1e4)
        m.shift_theta[:] = torch.tensor([1e4, -1e4])
        m.gate.fill_(0.01)
    k = m.build_kernels(torch.device("cpu"), torch.float32)
    mass = k.squeeze(2).abs().sum(-1).sum(0)
    assert torch.isfinite(k).all()
    assert torch.all(mass <= 1.00001)


def test_lambda_bound_holds_for_extreme_theta_values():
    m = DT1DAdapter(16)
    with torch.no_grad():
        m.shift_theta[:] = torch.tensor([1e9, -1e9])
    lam = m.shift_lambda(torch.device("cpu"), torch.float32)
    assert torch.all(torch.isfinite(lam))
    assert lam[0].item() <= 0.5
    assert lam[1].item() >= -0.5
    assert lam[0].item() > 0.499
    assert lam[1].item() < -0.499


@pytest.mark.parametrize("shape", [(1, 16, 1, 1), (1, 16, 1, 3), (1, 16, 3, 1), (2, 16, 2, 2)])
def test_tiny_spatial_inputs_are_supported_with_replicate_padding(shape):
    m = DT1DAdapter(16)
    x = torch.randn(*shape)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_forward_rejects_wrong_rank_and_channel_count_with_clear_errors():
    m = DT1DAdapter(16)
    with pytest.raises(ValueError, match="BCHW"):
        m(torch.randn(2, 16, 7))
    with pytest.raises(ValueError, match="Channel mismatch"):
        m(torch.randn(2, 15, 7, 7))


def test_float64_forward_backward_is_finite():
    torch.manual_seed(102)
    m = DT1DAdapter(16).double().train()
    x = torch.randn(1, 16, 5, 6, dtype=torch.float64, requires_grad=True)
    y = m(x)
    loss = y.square().mean()
    loss.backward()
    assert y.dtype == torch.float64
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_exactly_two_depthwise_conv_calls_in_canonical_forward(monkeypatch):
    m = DT1DAdapter(8)
    x = torch.randn(1, 8, 7, 9)
    real_conv2d = F.conv2d
    calls = []

    def counted(*args, **kwargs):
        calls.append((args[1].shape, kwargs.get("groups")))
        return real_conv2d(*args, **kwargs)

    monkeypatch.setattr(F, "conv2d", counted)
    y = m(x)
    assert y.shape == x.shape
    assert len(calls) == 2
    assert all(groups == 8 for _, groups in calls)


def test_checkpoint_load_invalidates_precomputed_inference_cache():
    torch.manual_seed(103)
    cached = DT1DAdapter(16, cache_kernel=True).eval()
    x = torch.randn(1, 16, 7, 7)
    _ = cached(x)
    assert cached._cached_kernels.numel() > 0

    fresh = DT1DAdapter(16).eval()
    with torch.no_grad():
        fresh.quotient_beta.normal_(0.0, 0.3)
        fresh.detail_eta.normal_(0.0, 0.2)
        fresh.shift_theta[:] = torch.tensor([0.4, -0.3])
        fresh.gate.fill_(0.07)
    expected = fresh(x)
    cached.load_state_dict(fresh.state_dict())
    assert cached._cached_kernels.numel() == 0
    got = cached(x)
    assert torch.allclose(got, expected, atol=1e-7, rtol=1e-6)


def test_state_dict_binary_roundtrip_preserves_output():
    torch.manual_seed(104)
    a = DT1DAdapter(16).eval()
    with torch.no_grad():
        for p in a.parameters():
            p.normal_(0.0, 0.05)
    x = torch.randn(1, 16, 5, 5)
    expected = a(x)
    stream = io.BytesIO()
    torch.save(a.state_dict(), stream)
    stream.seek(0)
    b = DT1DAdapter(16).eval()
    b.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
    assert torch.allclose(b(x), expected, atol=1e-7, rtol=1e-6)


def test_runtime_model_matches_machine_readable_proposal_contract():
    spec = load_spec()
    m = DT1DAdapter(32)
    assert spec["proposal"] == m.proposal_name
    assert spec["architecture"] == m.architecture_name
    assert tuple(spec["axes"]) == m.axis_names
    assert tuple(spec["active_offsets"]) == m.active_offsets
    assert spec["group_size"] == m.alpha_group
    assert spec["base_kernel_size"] == m.base_kernel_size
    assert spec["effective_kernel_size"] == m.effective_kernel_size
    assert spec["depthwise_convolution_calls"] == m.convolution_calls_per_forward
    assert proposal_fingerprint() == "c5a9104df8cb882b75d6176c3730a18d0dd61c2645f4000e654b62e561f14093"


@pytest.mark.parametrize("kwargs", [
    {"shift_lambda_init": float("nan")},
    {"shift_lambda_init": float("inf")},
    {"shift_lambda_max": float("nan")},
    {"shift_lambda_max": float("inf")},
])
def test_nonfinite_shift_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DT1DAblationAdapter(16, **kwargs)
