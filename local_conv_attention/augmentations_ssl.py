"""Alignment-preserving augmentations for dense SSL views."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DenseViewCorruptionConfig


class DenseSSLViewCorruptor(nn.Module):
    """Apply photometric and corruption-only transforms while preserving geometry."""

    def __init__(self, config: DenseViewCorruptionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

    def _intensity_jitter(self, x: Tensor) -> Tensor:
        scale = 1.0 + 0.2 * torch.randn(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)
        bias = 0.1 * torch.randn(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)
        return x * scale + bias

    def _blur(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def _gaussian_noise(self, x: Tensor) -> Tensor:
        return x + 0.05 * torch.randn_like(x)

    def _random_block_mask(self, x: Tensor) -> Tensor:
        if self.config.block_mask_ratio <= 0.0:
            return x
        batch, _, height, width = x.shape
        block_h = max(1, int(round(height * self.config.block_mask_ratio)))
        block_w = max(1, int(round(width * self.config.block_mask_ratio)))
        mask = torch.ones(batch, 1, height, width, device=x.device, dtype=x.dtype)
        for batch_index in range(batch):
            for _ in range(self.config.block_mask_num_blocks):
                y0 = int(torch.randint(0, max(1, height - block_h + 1), (1,), device=x.device))
                x0 = int(torch.randint(0, max(1, width - block_w + 1), (1,), device=x.device))
                mask[batch_index, :, y0 : y0 + block_h, x0 : x0 + block_w] = 0.0
        return x * mask

    def forward(self, x: Tensor) -> Tensor:
        if self.config.intensity_jitter:
            x = self._intensity_jitter(x)
        if self.config.blur:
            x = self._blur(x)
        if self.config.gaussian_noise:
            x = self._gaussian_noise(x)
        if self.config.random_block_mask:
            x = self._random_block_mask(x)
        return x
