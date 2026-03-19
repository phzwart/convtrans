from __future__ import annotations

import pytest
import torch

from local_conv_attention import (
    FlattenedMaskedLocalAttention2d,
    LocalAttention2d,
    ReferenceLocalAttention2d,
    ShiftLocalAttention2d,
)
from local_conv_attention.utils import attention_tolerances


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

OPERATOR_CASES = [
    (1, 1, 5, 5, 1, 8, 1),
    (2, 2, 7, 9, 3, 4, 2),
    (1, 4, 11, 11, 5, 4, 1),
]

FLAT_CASES = [
    (1, 2, 4, 4, 3, 4),
    (2, 1, 3, 5, 1, 8),
    (1, 4, 5, 5, 3, 2),
]


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    atol, rtol = attention_tolerances(actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("batch,num_heads,height,width,window_size,head_dim,dilation", OPERATOR_CASES)
def test_optimized_matches_shift_and_unfold_references(
    device: str,
    dtype: torch.dtype,
    seed: int,
    batch: int,
    num_heads: int,
    height: int,
    width: int,
    window_size: int,
    head_dim: int,
    dilation: int,
) -> None:
    torch.manual_seed(seed)
    channels = num_heads * head_dim
    q = torch.randn(batch, channels, height, width, device=device, dtype=dtype)
    k = torch.randn(batch, channels, height, width, device=device, dtype=dtype)
    v = torch.randn(batch, channels, height, width, device=device, dtype=dtype)

    optimized = LocalAttention2d(num_heads=num_heads, window_size=window_size, dilation=dilation)
    shift_ref = ShiftLocalAttention2d(num_heads=num_heads, window_size=window_size, dilation=dilation)
    unfold_ref = ReferenceLocalAttention2d(num_heads=num_heads, window_size=window_size, dilation=dilation)

    out_optimized, attn_optimized = optimized(q, k, v, return_attention=True)
    out_shift, attn_shift = shift_ref(q, k, v, return_attention=True)
    out_unfold, attn_unfold = unfold_ref(q, k, v, return_attention=True)

    _assert_close(out_optimized, out_shift)
    _assert_close(attn_optimized, attn_shift)
    _assert_close(out_optimized, out_unfold)
    _assert_close(attn_optimized, attn_unfold)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("batch,num_heads,height,width,window_size,head_dim", FLAT_CASES)
def test_fixed_shift_matches_flattened_masked_attention(
    device: str,
    dtype: torch.dtype,
    seed: int,
    batch: int,
    num_heads: int,
    height: int,
    width: int,
    window_size: int,
    head_dim: int,
) -> None:
    torch.manual_seed(seed)
    channels = num_heads * head_dim
    q = torch.randn(batch, channels, height, width, device=device, dtype=dtype)
    k = torch.randn(batch, channels, height, width, device=device, dtype=dtype)
    v = torch.randn(batch, channels, height, width, device=device, dtype=dtype)

    fixed = LocalAttention2d(num_heads=num_heads, window_size=window_size)
    flat = FlattenedMaskedLocalAttention2d(num_heads=num_heads, window_size=window_size)

    out_fixed = fixed(q, k, v)
    out_flat = flat(q, k, v)

    _assert_close(out_fixed, out_flat)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_window_size_one_returns_value_map(device: str, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 8, 5, 5, device=device, dtype=dtype)
    k = torch.randn(2, 8, 5, 5, device=device, dtype=dtype)
    v = torch.randn(2, 8, 5, 5, device=device, dtype=dtype)

    out = LocalAttention2d(num_heads=2, window_size=1)(q, k, v)
    _assert_close(out, v)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_queries_and_keys_reduce_to_local_average(device: str, dtype: torch.dtype) -> None:
    q = torch.ones(1, 4, 4, 4, device=device, dtype=dtype)
    k = torch.ones_like(q)
    v = torch.arange(1, 1 + 4 * 4 * 4, device=device, dtype=dtype).view(1, 4, 4, 4)

    out = LocalAttention2d(num_heads=2, window_size=3)(q, k, v)

    # For uniform scores, each valid neighbor gets equal weight.
    expected_corner = v[:, :, 0:2, 0:2].mean(dim=(-2, -1))
    _assert_close(out[:, :, 1, 1], v[:, :, 0:3, 0:3].mean(dim=(-2, -1)))
    _assert_close(out[:, :, 0, 0], expected_corner)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_edge_masking_matches_manual_corner_average(device: str, dtype: torch.dtype) -> None:
    q = torch.ones(1, 2, 3, 3, device=device, dtype=dtype)
    k = torch.ones_like(q)
    v = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]]],
        device=device,
        dtype=dtype,
    )

    out = LocalAttention2d(num_heads=1, window_size=3)(q, k, v)
    expected = torch.tensor([[3.0, 7.0]], device=device, dtype=dtype)
    _assert_close(out[0, :, 0, 0], expected[0])
