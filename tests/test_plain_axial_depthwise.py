from __future__ import annotations

import pytest
import torch

from models.plain_axial_depthwise_adapter import PlainAxialDepthwiseAdapter


@pytest.mark.parametrize(
    "mode,ncoef",
    [("unrestricted", 17), ("direct_symmetric", 9), ("reduced_symmetric", 5)],
)
def test_axial_ablation_modes(mode, ncoef):
    model = PlainAxialDepthwiseAdapter(C=8, axis="hw", kernel_mode=mode, gate_init=0.01)
    assert model.coefficients.shape == (2, 8, ncoef)
    x = torch.randn(2, 8, 15, 19, requires_grad=True)
    y = model(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None
    assert model.convolution_calls_per_forward == 2


def test_unrestricted_kernel_can_be_asymmetric():
    model = PlainAxialDepthwiseAdapter(C=2, kernel_mode="unrestricted")
    with torch.no_grad():
        model.coefficients.zero_()
        model.coefficients[0, 0, 1] = 1.0
    kernel = model.build_kernels(torch.device("cpu"), torch.float32).squeeze(2)
    assert not torch.allclose(kernel[0, 0], kernel[0, 0].flip(0))


def test_direct_and_reduced_kernels_are_symmetric():
    for mode in ("direct_symmetric", "reduced_symmetric"):
        model = PlainAxialDepthwiseAdapter(C=4, kernel_mode=mode)
        with torch.no_grad():
            model.coefficients.normal_()
        kernel = model.build_kernels(torch.device("cpu"), torch.float32).squeeze(2)
        assert torch.allclose(kernel, kernel.flip(-1))
