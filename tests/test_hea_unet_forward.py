from __future__ import annotations

import torch

from local_conv_attention import HEAUNet, HEAUNetModelConfig


def test_hea_unet_forward_shape() -> None:
    config = HEAUNetModelConfig(
        in_channels=1,
        num_classes=3,
        base_channels=8,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    model = HEAUNet(config)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 3, 64, 64)


def test_hea_unet_progressive_forward_shape() -> None:
    config = HEAUNetModelConfig(
        in_channels=3,
        num_classes=2,
        base_channels=8,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    config.attention.elevator_mode = "progressive"
    config.hea.enabled_decoder_stages = [0, 1]
    model = HEAUNet(config)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 2, 64, 64)
