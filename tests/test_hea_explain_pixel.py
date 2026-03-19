from __future__ import annotations

import torch

from local_conv_attention import HEAExplainer, HEAUNet, HEAUNetModelConfig


def _make_model(*, window_size: int = 3) -> HEAUNet:
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
    config.semantic_memory.window_sizes = [window_size, window_size, window_size]
    config.hea.per_scale_window_sizes = [window_size, window_size, window_size]
    config.bottleneck_window_size = window_size
    config.validate()
    return HEAUNet(config).eval()


def test_explain_pixel_returns_expected_structure() -> None:
    torch.manual_seed(0)
    model = _make_model(window_size=3)
    explainer = HEAExplainer(model)
    x = torch.randn(1, 1, 32, 32)

    result = explainer.explain_pixel(
        x,
        target_xy=(10, 12),
        mode="gated_magnitude",
    )

    assert result["target_xy"] == (10, 12)
    assert tuple(result["combined_heatmap"].shape) == (32, 32)
    assert result["gate_value"] is not None
    assert set(result["per_scale_heatmaps"]) == {1, 2, 3}
    for heatmap in result["per_scale_heatmaps"].values():
        assert torch.isfinite(heatmap).all()
        assert heatmap.ndim == 2


def test_explain_pixel_coordinate_mapping_for_unit_window() -> None:
    torch.manual_seed(1)
    model = _make_model(window_size=1)
    explainer = HEAExplainer(model)
    x = torch.randn(1, 1, 32, 32)

    result = explainer.explain_pixel(
        x,
        target_xy=(9, 13),
        mode="attention",
    )

    scale1_heatmap = result["per_scale_heatmaps"][1]
    nonzero = torch.nonzero(scale1_heatmap > 0, as_tuple=False)
    assert nonzero.shape[0] == 1
    assert tuple(nonzero[0].tolist()) == (9 // 2, 13 // 2)
