"""Mixed-precision regression coverage for the revised DT1D proposal."""
import pytest
import torch

from models.dt1d_adapter import DT1DAdapter


def make_model():
    return DT1DAdapter(C=16)


@pytest.mark.parametrize("target_dtype", [torch.float16, torch.bfloat16])
def test_kernel_builder_respects_requested_dtype(target_dtype):
    model = make_model()
    kernel = model.build_kernels(torch.device("cpu"), target_dtype)
    assert kernel.dtype == target_dtype
    assert torch.isfinite(kernel.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_autocast_forward_and_backward():
    model = make_model().cuda().train()
    x = torch.randn(2, 16, 19, 23, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(x).float().square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
