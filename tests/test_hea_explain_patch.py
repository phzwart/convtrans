from __future__ import annotations

import torch

from local_conv_attention import HEAExplainer, HEAUNet, HEAUNetModelConfig


def _make_model() -> HEAUNet:
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
    config.validate()
    return HEAUNet(config).eval()


def test_explain_patch_returns_finite_aggregates() -> None:
    torch.manual_seed(2)
    model = _make_model()
    explainer = HEAExplainer(model)
    x = torch.randn(1, 1, 32, 32)

    result = explainer.explain_patch(
        x,
        center_xy=(12, 12),
        patch_radius=1,
        mode="gated_magnitude",
        score_norm="l2",
    )

    assert result["target_patch"] == ((11, 14), (11, 14))
    assert torch.isfinite(result["target_logit"])
    for score_tensor in result["gated_scores"].values():
        assert torch.isfinite(score_tensor).all()
        assert (score_tensor >= 0).all()


def test_explain_patch_signed_logit_returns_signed_maps() -> None:
    torch.manual_seed(3)
    model = _make_model()
    with torch.no_grad():
        channels = model.segmentation_head.weight.size(1)
        pattern = torch.tensor(
            [1.0 if index % 2 == 0 else -1.0 for index in range(channels)],
            dtype=model.segmentation_head.weight.dtype,
        )
        model.segmentation_head.weight[0, :, 0, 0] = pattern
        model.segmentation_head.bias.zero_()

    explainer = HEAExplainer(model)
    x = torch.randn(1, 1, 32, 32)
    result = explainer.explain_patch(
        x,
        target_patch=((8, 12), (8, 12)),
        mode="signed_logit",
        target_channel=0,
    )

    assert result["signed_scores"] is not None
    assert torch.isfinite(result["combined_heatmap"]).all()
    assert result["positive_heatmap"] is not None
    assert result["negative_heatmap"] is not None
    assert result["positive_heatmap"].shape == (32, 32)
    assert result["negative_heatmap"].shape == (32, 32)
