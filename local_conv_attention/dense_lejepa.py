"""Dense LeJEPA-style self-supervised adaptor for the HEA backbone.

This module learns dense latent fields from multiple aligned views of the same
image. It is symmetric and teacher-free: there is no EMA target network, no
predictor head, and no stop-gradient branch. Training minimizes a dense
invariance loss across aligned views plus SIGReg over the resulting latent
vectors.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import HEABackbone
from .config import HEAUNetModelConfig
from .encoder import make_activation
from .losses_ssl import dense_invariance_loss
from .sigreg import SIGRegLoss
from .views import DenseAlignedViewGenerator


class DenseLatentProjector2d(nn.Module):
    """Project dense HEA features into dense latent vectors."""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        *,
        depth: int = 1,
        activation: str = "gelu",
        normalize_latents: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("DenseLatentProjector2d depth must be at least 1.")
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(depth - 1):
            layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=1))
            layers.append(make_activation(activation))
        layers.append(nn.Conv2d(current_channels, latent_dim, kernel_size=1))
        self.proj = nn.Sequential(*layers)
        self.normalize_latents = normalize_latents

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        if self.normalize_latents:
            x = F.normalize(x, dim=1)
        return x


class DenseLeJEPAModel(nn.Module):
    """Teacher-free dense latent pretraining model for the HEA backbone."""

    def __init__(self, config: HEAUNetModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = HEABackbone(config)
        self.view_generator = DenseAlignedViewGenerator(config.lejepa)
        latent_in_channels = self._latent_in_channels()
        self.projector = DenseLatentProjector2d(
            latent_in_channels,
            config.latent.latent_dim,
            depth=config.latent.projector_depth,
            activation=config.act,
            normalize_latents=config.latent.normalize_latents,
        )
        self.sigreg = SIGRegLoss(
            num_slices=config.lejepa.sigreg.num_slices,
            num_knots=config.lejepa.sigreg.num_knots,
            t_max=config.lejepa.sigreg.t_max,
        )

    def _latent_in_channels(self) -> int:
        source = self.config.latent.source
        if source == "top":
            return self.backbone.channels[0]
        if source == "bottleneck":
            return self.backbone.channels[-1]
        if source.startswith("encoder_"):
            return self.backbone.channels[int(source.split("_", maxsplit=1)[1])]
        if source.startswith("decoder_"):
            return self.backbone.channels[int(source.split("_", maxsplit=1)[1])]
        raise ValueError(f"Unsupported latent source {source!r}.")

    @staticmethod
    def _downsample_valid_mask(valid_mask: Tensor | None, shape: tuple[int, int]) -> Tensor | None:
        if valid_mask is None:
            return None
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        return F.interpolate(valid_mask.float(), size=shape, mode="nearest")

    @staticmethod
    def _flatten_latents(latents: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if valid_mask is None:
            return latents.permute(0, 2, 3, 1).reshape(-1, latents.size(1))
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        flat_latents = latents.permute(0, 2, 3, 1).reshape(-1, latents.size(1))
        flat_mask = valid_mask.reshape(-1) > 0
        return flat_latents[flat_mask]

    def _sigreg_loss(self, latents: Tensor, valid_mask: Tensor | None) -> Tensor:
        if not self.config.lejepa.sigreg.enabled:
            return latents.new_zeros(())
        if self.config.lejepa.sigreg.per_view:
            losses = []
            for view_index in range(latents.size(1)):
                embeddings = self._flatten_latents(latents[:, view_index], valid_mask)
                if embeddings.size(0) < 2:
                    losses.append(latents.new_zeros(()))
                    continue
                losses.append(self.sigreg(embeddings))
            return torch.stack(losses).mean()
        embeddings = latents.permute(0, 1, 3, 4, 2).reshape(-1, latents.size(2))
        if valid_mask is not None:
            flat_mask = valid_mask.reshape(-1) > 0
            flat_mask = flat_mask.repeat_interleave(latents.size(1))
            embeddings = embeddings[flat_mask]
        if embeddings.size(0) < 2:
            return latents.new_zeros(())
        return self.sigreg(embeddings)

    def _prepare_views(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        if x.dim() == 5:
            return x, None
        return self.view_generator(x)

    def forward(
        self,
        x: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        views, generated_valid_mask = self._prepare_views(x)
        if valid_mask is None:
            valid_mask = generated_valid_mask

        batch, num_views, channels, height, width = views.shape
        flat_views = views.reshape(batch * num_views, channels, height, width)
        backbone_features = self.backbone(flat_views)
        projected = self.projector(backbone_features)
        latents = projected.reshape(
            batch,
            num_views,
            projected.size(1),
            projected.size(2),
            projected.size(3),
        )
        latent_valid_mask = self._downsample_valid_mask(valid_mask, latents.shape[-2:])
        loss_valid_mask = latent_valid_mask if self.config.lejepa.invariance.loss_on_valid_only else None

        inv_loss = dense_invariance_loss(latents, valid_mask=loss_valid_mask)
        sigreg_loss = self._sigreg_loss(latents, loss_valid_mask)
        lambda_sigreg = self.config.lejepa.lambda_sigreg
        loss = (1.0 - lambda_sigreg) * inv_loss + lambda_sigreg * sigreg_loss
        return {
            "latents": latents,
            "inv_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
            "loss": loss,
            "valid_mask": latent_valid_mask,
        }
