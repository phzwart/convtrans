from __future__ import annotations

import torch

from local_conv_attention import HEAExplainer, HEAUNet, HEAUNetModelConfig


def test_ablate_region_changes_target_score() -> None:
    torch.manual_seed(4)
    config = HEAUNetModelConfig(
        in_channels=1,
        num_classes=1,
        base_channels=4,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    config.attention.heads = 2
    config.attention.head_dim = 4
    model = HEAUNet(config).eval()
    explainer = HEAExplainer(model)
    x = torch.randn(1, 1, 32, 32)

    explanation = explainer.explain_pixel(
        x,
        target_xy=(8, 8),
        mode="gated_magnitude",
    )
    scale_heatmap = explanation["per_scale_heatmaps"][1]
    top_coord = torch.nonzero(scale_heatmap == scale_heatmap.max(), as_tuple=False)[0]

    result = explainer.ablate_region(
        x,
        memory_scale=1,
        coarse_coord=(int(top_coord[0]), int(top_coord[1])),
        target_xy=(8, 8),
    )

    assert torch.isfinite(result["delta"])
    assert result["delta"].abs().item() > 0.0
