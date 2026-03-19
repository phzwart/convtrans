from __future__ import annotations

import matplotlib
import torch

from local_conv_attention.visualization import (
    combine_upsampled_heatmaps,
    upsample_region_heatmap,
    visualize_explanation,
    visualize_signed_explanation,
)


matplotlib.use("Agg")


def test_heatmap_upsampling_and_combination() -> None:
    heatmap = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    upsampled = upsample_region_heatmap(heatmap, scale_factor=2, output_shape=(4, 4))
    assert tuple(upsampled.shape) == (4, 4)
    combined = combine_upsampled_heatmaps({2: heatmap, 4: torch.ones(1, 1)}, output_shape=(4, 4))
    assert tuple(combined.shape) == (4, 4)
    assert torch.isfinite(combined).all()


def test_visualization_helpers_render_without_crashing() -> None:
    image = torch.rand(1, 16, 16)
    explanation = {
        "per_scale_heatmaps": {
            1: torch.rand(8, 8),
            2: torch.rand(4, 4),
        },
        "combined_heatmap": torch.rand(16, 16),
        "target_xy": (4, 5),
    }
    fig = visualize_explanation(image, explanation, target_xy=(4, 5))
    signed_fig = visualize_signed_explanation(
        image,
        positive_heatmap=torch.rand(16, 16),
        negative_heatmap=torch.rand(16, 16),
        target_xy=(4, 5),
    )
    assert fig is not None
    assert signed_fig is not None
