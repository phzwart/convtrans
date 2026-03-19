"""Aligned multi-view generation for dense LeJEPA training."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .augmentations_ssl import DenseSSLViewCorruptor
from .config import DenseLeJEPAObjectiveConfig


class DenseAlignedViewGenerator(nn.Module):
    """Generate multiple aligned views with shared geometry."""

    def __init__(self, config: DenseLeJEPAObjectiveConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.corruptor = DenseSSLViewCorruptor(config.views.corruption)

    def _shared_crop(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ratio = self.config.views.shared_crop_ratio
        if ratio >= 1.0:
            valid_mask = torch.ones(x.size(0), 1, x.size(-2), x.size(-1), device=x.device, dtype=x.dtype)
            return x, valid_mask

        height, width = x.shape[-2:]
        crop_h = max(1, int(round(height * ratio)))
        crop_w = max(1, int(round(width * ratio)))
        y0 = int(torch.randint(0, max(1, height - crop_h + 1), (1,), device=x.device))
        x0 = int(torch.randint(0, max(1, width - crop_w + 1), (1,), device=x.device))
        cropped = x[..., y0 : y0 + crop_h, x0 : x0 + crop_w]
        valid_mask = torch.ones(
            x.size(0),
            1,
            crop_h,
            crop_w,
            device=x.device,
            dtype=x.dtype,
        )
        return cropped, valid_mask

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        if x.dim() != 4:
            raise ValueError(f"Expected inputs with shape [B, C, H, W], got {tuple(x.shape)}.")

        valid_mask = None
        if self.config.views.mode == "aligned_shared_crop":
            x, valid_mask = self._shared_crop(x)
        elif self.config.views.mode != "aligned_same_geometry":
            raise ValueError(f"Unsupported view mode {self.config.views.mode!r}.")

        views = [self.corruptor(x.clone()) for _ in range(self.config.num_views)]
        return torch.stack(views, dim=1), valid_mask
