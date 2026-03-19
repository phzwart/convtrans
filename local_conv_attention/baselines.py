"""Comparison baseline segmentation models."""

from __future__ import annotations

from torch import Tensor, nn

from .config import HEAUNetModelConfig
from .decoder import HEADecoderStage
from .encoder import ConvStem2d, HEAEncoderStage, ResidualConvBlock2d


class BasicUNet(nn.Module):
    """A plain convolutional U-Net baseline for segmentation comparison."""

    def __init__(self, config: HEAUNetModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        channels = [config.base_channels * mult for mult in config.channel_multipliers]
        self.channels = channels
        self.stem = ConvStem2d(
            config.in_channels,
            channels[0],
            norm=config.norm,
            act=config.act,
        )

        self.encoder_stages = nn.ModuleList(
            [
                HEAEncoderStage(
                    channels[0],
                    channels[0],
                    depth=config.encoder_depths[0],
                    downsample=False,
                    norm=config.norm,
                    act=config.act,
                )
            ]
        )
        for scale in range(1, len(channels)):
            self.encoder_stages.append(
                HEAEncoderStage(
                    channels[scale - 1],
                    channels[scale],
                    depth=config.encoder_depths[scale],
                    downsample=True,
                    norm=config.norm,
                    act=config.act,
                )
            )

        self.bottleneck = nn.Sequential(
            *[
                ResidualConvBlock2d(
                    channels[-1],
                    channels[-1],
                    norm=config.norm,
                    act=config.act,
                )
                for _ in range(config.bottleneck_depth)
            ]
        )

        self.decoder_stages = nn.ModuleDict()
        for target_scale in reversed(range(len(channels) - 1)):
            decoder_depth_index = len(channels) - 2 - target_scale
            self.decoder_stages[str(target_scale)] = HEADecoderStage(
                in_channels=channels[target_scale + 1],
                out_channels=channels[target_scale],
                skip_channels=channels[target_scale],
                depth=config.decoder_depths[decoder_depth_index],
                use_raw_skip=True,
                norm=config.norm,
                act=config.act,
            )

        self.segmentation_head = nn.Conv2d(channels[0], config.num_classes, kernel_size=1)

    def forward_features(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = self.stem(x)
        encoder_features = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)

        x = self.bottleneck(encoder_features[-1])
        for target_scale in reversed(range(len(self.channels) - 1)):
            x = self.decoder_stages[str(target_scale)](x, encoder_features[target_scale])
        return x, encoder_features

    def forward(self, x: Tensor) -> Tensor:
        features, _ = self.forward_features(x)
        return self.segmentation_head(features)
