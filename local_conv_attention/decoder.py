"""Decoder-side building blocks for HEA-UNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoder import ActKind, NormKind, ResidualConvBlock2d


class HEADecoderStage(nn.Module):
    """A simple upsample-refine decoder stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        skip_channels: int,
        depth: int,
        use_raw_skip: bool,
        norm: NormKind = "batchnorm",
        act: ActKind = "gelu",
    ) -> None:
        super().__init__()
        self.use_raw_skip = use_raw_skip
        self.up_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        blocks = []
        current_in = out_channels + skip_channels if use_raw_skip else out_channels
        for _ in range(depth):
            blocks.append(
                ResidualConvBlock2d(
                    current_in,
                    out_channels,
                    norm=norm,
                    act=act,
                )
            )
            current_in = out_channels
        self.refine = nn.Sequential(*blocks)

    def forward(self, x: Tensor, reference: Tensor) -> Tensor:
        x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_proj(x)
        if self.use_raw_skip:
            x = torch.cat([x, reference], dim=1)
        return self.refine(x)
