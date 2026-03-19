"""Encoder-side building blocks for HEA-UNet."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


NormKind = Literal["batchnorm", "groupnorm", "instancenorm", "none"]
ActKind = Literal["gelu", "relu", "silu"]


def make_norm2d(kind: NormKind, channels: int) -> nn.Module:
    """Create a configurable 2D normalization layer."""
    if kind == "batchnorm":
        return nn.BatchNorm2d(channels)
    if kind == "groupnorm":
        groups = 8
        while channels % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    if kind == "instancenorm":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm kind: {kind}")


def make_activation(kind: ActKind) -> nn.Module:
    """Create a configurable activation module."""
    if kind == "gelu":
        return nn.GELU()
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation kind: {kind}")


class ConvNormAct2d(nn.Module):
    """A small Conv-Norm-Activation helper block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        norm: NormKind = "batchnorm",
        act: ActKind = "gelu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=norm == "none",
        )
        self.norm = make_norm2d(norm, out_channels)
        self.act = make_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualConvBlock2d(nn.Module):
    """A simple residual conv block for encoder and decoder stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: NormKind = "batchnorm",
        act: ActKind = "gelu",
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct2d(in_channels, out_channels, norm=norm, act=act)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=norm == "none"),
            make_norm2d(norm, out_channels),
        )
        self.act = make_activation(act)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class ConvStem2d(nn.Module):
    """Initial convolutional stem."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: NormKind = "batchnorm",
        act: ActKind = "gelu",
    ) -> None:
        super().__init__()
        self.proj = ConvNormAct2d(in_channels, out_channels, norm=norm, act=act)
        self.refine = ResidualConvBlock2d(out_channels, out_channels, norm=norm, act=act)

    def forward(self, x: Tensor) -> Tensor:
        return self.refine(self.proj(x))


class HEAEncoderStage(nn.Module):
    """Configurable encoder stage with optional stride-2 downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int,
        downsample: bool,
        norm: NormKind = "batchnorm",
        act: ActKind = "gelu",
    ) -> None:
        super().__init__()
        self.downsample = (
            ConvNormAct2d(in_channels, out_channels, stride=2, norm=norm, act=act)
            if downsample
            else nn.Identity()
        )

        blocks = []
        current_in = out_channels if downsample else in_channels
        current_out = out_channels
        for _ in range(depth):
            blocks.append(
                ResidualConvBlock2d(
                    current_in,
                    current_out,
                    norm=norm,
                    act=act,
                )
            )
            current_in = current_out
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        return self.blocks(x)
