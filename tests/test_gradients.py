from __future__ import annotations

import pytest
import torch

from local_conv_attention import (
    LocalAttention2d,
    LocalTransformerBlock2d,
    ReferenceLocalAttention2d,
    ReferenceLocalTransformerBlock2d,
    ShiftLocalAttention2d,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _grad_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float64:
        return 1e-10, 1e-8
    # Gradients go through softmax and LayerNorm, so float32 gets a slightly looser
    # tolerance than the forward checks while still remaining strict.
    return 2e-5, 2e-4


def _assert_grad_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    atol, rtol = _grad_tolerances(actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_operator_gradients_match_references(device: str, dtype: torch.dtype, seed: int) -> None:
    torch.manual_seed(seed)
    q1 = torch.randn(2, 8, 5, 5, device=device, dtype=dtype, requires_grad=True)
    k1 = torch.randn(2, 8, 5, 5, device=device, dtype=dtype, requires_grad=True)
    v1 = torch.randn(2, 8, 5, 5, device=device, dtype=dtype, requires_grad=True)

    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)
    q3 = q1.detach().clone().requires_grad_(True)
    k3 = k1.detach().clone().requires_grad_(True)
    v3 = v1.detach().clone().requires_grad_(True)

    optimized = LocalAttention2d(num_heads=2, window_size=3).to(device=device, dtype=dtype)
    shift_ref = ShiftLocalAttention2d(num_heads=2, window_size=3).to(device=device, dtype=dtype)
    unfold_ref = ReferenceLocalAttention2d(num_heads=2, window_size=3).to(device=device, dtype=dtype)

    target = torch.randn(2, 8, 5, 5, device=device, dtype=dtype)
    loss_optimized = torch.nn.functional.mse_loss(optimized(q1, k1, v1), target)
    loss_shift = torch.nn.functional.mse_loss(shift_ref(q2, k2, v2), target)
    loss_unfold = torch.nn.functional.mse_loss(unfold_ref(q3, k3, v3), target)
    loss_optimized.backward()
    loss_shift.backward()
    loss_unfold.backward()

    _assert_grad_close(q1.grad, q2.grad)
    _assert_grad_close(q1.grad, q3.grad)
    _assert_grad_close(k1.grad, k2.grad)
    _assert_grad_close(k1.grad, k3.grad)
    _assert_grad_close(v1.grad, v2.grad)
    _assert_grad_close(v1.grad, v3.grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [0, 1])
def test_block_gradients_match_reference(device: str, dtype: torch.dtype, seed: int) -> None:
    torch.manual_seed(seed)
    block = LocalTransformerBlock2d(dim=8, num_heads=2, window_size=3).to(device=device, dtype=dtype)
    reference = ReferenceLocalTransformerBlock2d(dim=8, num_heads=2, window_size=3).to(
        device=device, dtype=dtype
    )
    reference.load_state_dict(block.state_dict())

    x1 = torch.randn(2, 8, 4, 4, device=device, dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    target = torch.randn(2, 8, 4, 4, device=device, dtype=dtype)

    loss_block = torch.nn.functional.mse_loss(block(x1), target)
    loss_ref = torch.nn.functional.mse_loss(reference(x2), target)
    loss_block.backward()
    loss_ref.backward()

    _assert_grad_close(x1.grad, x2.grad)
    for (name_a, param_a), (name_b, param_b) in zip(block.named_parameters(), reference.named_parameters()):
        assert name_a == name_b
        _assert_grad_close(param_a.grad, param_b.grad)
