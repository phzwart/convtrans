from __future__ import annotations

import torch

from local_conv_attention import HEABackbone, HEAUNetModelConfig


def test_hea_backbone_returns_dense_latent_feature() -> None:
    config = HEAUNetModelConfig(
        name="hea_dense_lejepa",
        in_channels=1,
        base_channels=4,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    config.latent.source = "top"
    backbone = HEABackbone(config)
    x = torch.randn(2, 1, 32, 32)

    outputs = backbone.forward_features(x)

    assert outputs["top_feature"].shape == (2, 4, 32, 32)
    assert outputs["latent_feature"].shape == (2, 4, 32, 32)
    assert len(outputs["encoder_features"]) == 4
    assert set(outputs["decoder_features"]) == {0, 1, 2}


def test_hea_backbone_can_select_bottleneck_latent_source() -> None:
    config = HEAUNetModelConfig(
        name="hea_dense_lejepa",
        in_channels=1,
        base_channels=4,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    config.latent.source = "bottleneck"
    backbone = HEABackbone(config)
    x = torch.randn(1, 1, 32, 32)

    outputs = backbone.forward_features(x)

    assert outputs["latent_feature"].shape == (1, 32, 4, 4)
