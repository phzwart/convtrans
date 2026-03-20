from __future__ import annotations

import torch

from local_conv_attention.masks import local_validity_mask
from local_conv_attention.ops import NeighborhoodShift2d


def test_neighborhood_shift_explicit_values() -> None:
    x = torch.arange(1, 10, dtype=torch.float64).view(1, 1, 1, 3, 3)
    shifted, mask = NeighborhoodShift2d(window_size=3)(x, return_mask=True)

    expected_center = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 4.0, 5.0],
        ],
        dtype=torch.float64,
    )
    expected_middle = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=torch.float64,
    )

    assert shifted.shape == (1, 1, 9, 1, 3, 3)
    torch.testing.assert_close(shifted[0, 0, 0, 0], expected_center)
    torch.testing.assert_close(shifted[0, 0, 4, 0], expected_middle)
    assert mask.dtype == torch.bool


def test_local_validity_mask_matches_known_corner_pattern() -> None:
    mask = local_validity_mask(3, 3, window_size=3)
    top_left = mask[:, 0, 0]
    expected = torch.tensor(
        [False, False, False, False, True, True, False, True, True],
        dtype=torch.bool,
    )
    torch.testing.assert_close(top_left, expected)


def test_window_size_one_is_identity_shift() -> None:
    x = torch.randn(2, 3, 4, 5, 6)
    shifted, mask = NeighborhoodShift2d(window_size=1)(x, return_mask=True)
    torch.testing.assert_close(shifted[:, :, 0], x)
    assert mask.all()


def test_reflect_padding_corner_neighbor_not_zero() -> None:
    """Reflect/replicate fill: top-left stencil includes mirrored edge, not zeros."""
    x = torch.arange(1, 10, dtype=torch.float64).view(1, 1, 1, 3, 3)
    shifted, mask = NeighborhoodShift2d(window_size=3, boundary_pad="reflect")(x, return_mask=True)
    assert mask.all()
    # With reflect, offset (-1,-1) at corner should see reflected content, not 0.
    assert shifted[0, 0, 0, 0, 0, 0].item() != 0.0


def test_conv_shift_bank_matches_neighborhood_shift_reflect() -> None:
    from local_conv_attention.ops import ConvShiftBank2d

    torch.manual_seed(0)
    x4 = torch.randn(2, 8, 11, 13)
    # NeighborhoodShift2d uses [B, heads, dim, H, W]; use heads=1 for same layout as NCHW.
    x5 = x4.unsqueeze(1)
    ws, dil = 5, 2
    a, ma = ConvShiftBank2d(window_size=ws, dilation=dil, boundary_pad="reflect")(x4, return_mask=True)
    b, mb = NeighborhoodShift2d(window_size=ws, dilation=dil, boundary_pad="reflect")(x5, return_mask=True)
    # b is [B, heads, K, dim, H, W] — merge heads*dim like NCHW for comparison.
    b_nchw = b.permute(0, 1, 3, 2, 4, 5).reshape(
        b.size(0), b.size(1) * b.size(3), b.size(2), b.size(4), b.size(5)
    )
    torch.testing.assert_close(a, b_nchw)
    torch.testing.assert_close(ma, mb.squeeze(3))
