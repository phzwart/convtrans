from __future__ import annotations

import pytest
import torch

from local_conv_attention import LocalTransformerBlock2d, ReferenceLocalTransformerBlock2d
from local_conv_attention.utils import attention_tolerances


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    atol, rtol = attention_tolerances(actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dim,num_heads,window_size,height,width", [(8, 2, 3, 4, 4), (16, 4, 5, 5, 5)])
def test_block_matches_reference_with_shared_weights(
    device: str,
    dtype: torch.dtype,
    seed: int,
    dim: int,
    num_heads: int,
    window_size: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(seed)
    block = LocalTransformerBlock2d(dim=dim, num_heads=num_heads, window_size=window_size).to(
        device=device, dtype=dtype
    )
    reference = ReferenceLocalTransformerBlock2d(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
    ).to(device=device, dtype=dtype)
    reference.load_state_dict(block.state_dict())

    x = torch.randn(2, dim, height, width, device=device, dtype=dtype)
    _assert_close(block(x), reference(x))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimized_block_matches_shift_reference(
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    optimized = LocalTransformerBlock2d(
        dim=8,
        num_heads=2,
        window_size=3,
        implementation="optimized",
    ).to(device=device, dtype=dtype)
    shift_reference = LocalTransformerBlock2d(
        dim=8,
        num_heads=2,
        window_size=3,
        implementation="shift",
    ).to(device=device, dtype=dtype)
    shift_reference.load_state_dict(optimized.state_dict())

    x = torch.randn(2, 8, 4, 4, device=device, dtype=dtype)
    _assert_close(optimized(x), shift_reference(x))


@pytest.mark.parametrize("device", DEVICES)
def test_block_state_dicts_are_compatible(device: str) -> None:
    block = LocalTransformerBlock2d(dim=12, num_heads=3, window_size=3).to(device=device)
    reference = ReferenceLocalTransformerBlock2d(dim=12, num_heads=3, window_size=3).to(device=device)
    missing, unexpected = reference.load_state_dict(block.state_dict(), strict=False)
    assert not missing
    assert not unexpected
