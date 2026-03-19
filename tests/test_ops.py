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
