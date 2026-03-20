"""Aligned multi-view generation for dense LeJEPA training."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .augmentations_ssl import DenseSSLViewCorruptor
from .config import DenseLeJEPAObjectiveConfig, DenseViewConfig
from .synthetic_data import rotate_tensor_nchw


def sample_pre_corrupt_rotation_deg(config: DenseViewConfig, device: torch.device) -> float:
    """Return θ (degrees CCW) for one view’s rotate → corrupt → derotate pipeline."""
    if config.pre_corrupt_rotation_quarter_turns:
        k = int(torch.randint(0, 4, (1,), device=device).item())
        return 90.0 * float(k)
    lo, hi = config.pre_corrupt_rotation_deg
    lo_f, hi_f = float(lo), float(hi)
    span = hi_f - lo_f
    if span > 0.0:
        u = torch.rand((), device=device, dtype=torch.float32)
        return lo_f + span * float(u.item())
    return lo_f


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

        views: list[Tensor] = []
        pad = self.config.views.pre_corrupt_rotation_padding
        for _ in range(self.config.num_views):
            xv = x.clone()
            if self.config.views.pre_corrupt_rotation:
                theta = sample_pre_corrupt_rotation_deg(self.config.views, xv.device)
                if abs(theta) > 1e-8:
                    xv = rotate_tensor_nchw(xv, theta, padding_mode=pad)
                    xv = self.corruptor(xv)
                    xv = rotate_tensor_nchw(xv, -theta, padding_mode=pad)
                else:
                    # No extra RNG vs ``pre_corrupt_rotation=False`` when range is (0, 0).
                    xv = self.corruptor(xv)
            else:
                xv = self.corruptor(xv)
            views.append(xv)
        return torch.stack(views, dim=1), valid_mask
