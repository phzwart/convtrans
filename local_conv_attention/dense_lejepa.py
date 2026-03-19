"""Dense LeJEPA-style self-supervised adaptor for the HEA backbone.

This module learns dense latent fields from multiple aligned views of the same
image. It is symmetric and teacher-free: there is no EMA target network, no
predictor head, and no stop-gradient branch. Training minimizes a dense
invariance loss across aligned views plus SIGReg over the resulting latent
vectors. Multiple backbone hooks (multi-scale) can be trained jointly.
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


def _latent_source_module_key(name: str) -> str:
    return name.replace(".", "_dot_")


def _in_channels_for_latent_source(channels: list[int], source: str) -> int:
    if source == "top":
        return channels[0]
    if source == "bottleneck":
        return channels[-1]
    if source.startswith("encoder_"):
        return channels[int(source.split("_", maxsplit=1)[1])]
    if source.startswith("decoder_"):
        return channels[int(source.split("_", maxsplit=1)[1])]
    raise ValueError(f"Unsupported latent source {source!r}.")


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
        self._latent_sources = config.latent.resolved_sources()
        self.projectors = nn.ModuleDict(
            {
                _latent_source_module_key(s): DenseLatentProjector2d(
                    _in_channels_for_latent_source(self.backbone.channels, s),
                    config.latent.latent_dim,
                    depth=config.latent.projector_depth,
                    activation=config.act,
                    normalize_latents=config.latent.normalize_latents,
                )
                for s in self._latent_sources
            }
        )
        # First hook’s projector (tests / demos that touch ``model.projector``).
        self.projector = self.projectors[_latent_source_module_key(self._latent_sources[0])]
        self.sigreg = SIGRegLoss(
            num_slices=config.lejepa.sigreg.num_slices,
            num_knots=config.lejepa.sigreg.num_knots,
            t_max=config.lejepa.sigreg.t_max,
        )

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

    def _latents_from_feature_pack(
        self,
        pack: dict[str, Any],
        *,
        batch: int,
        num_views: int,
    ) -> dict[str, Tensor]:
        top = pack["top_feature"]
        encoder_features = pack["encoder_features"]
        decoder_features = pack["decoder_features"]
        latents_by_source: dict[str, Tensor] = {}
        for source in self._latent_sources:
            feat = self.backbone.resolve_latent_tensor(
                source,
                top_feature=top,
                encoder_features=encoder_features,
                decoder_features=decoder_features,
            )
            proj = self.projectors[_latent_source_module_key(source)](feat)
            _, dim, h, w = proj.shape
            latents_by_source[source] = proj.view(batch, num_views, dim, h, w)
        return latents_by_source

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
        if self.config.lejepa.sequential_view_forward:
            accum: dict[str, list[Tensor]] = {s: [] for s in self._latent_sources}
            for view_index in range(num_views):
                chunk = views[:, view_index].contiguous()
                pack = self.backbone.forward_features(chunk)
                partial = self._latents_from_feature_pack(pack, batch=batch, num_views=1)
                for s in self._latent_sources:
                    accum[s].append(partial[s])
            latents_by_source = {s: torch.cat(accum[s], dim=1) for s in self._latent_sources}
        else:
            flat_views = views.reshape(batch * num_views, channels, height, width)
            pack = self.backbone.forward_features(flat_views)
            latents_by_source = self._latents_from_feature_pack(
                pack,
                batch=batch,
                num_views=num_views,
            )

        primary = self._latent_sources[0]
        latents = latents_by_source[primary]
        latent_valid_mask = self._downsample_valid_mask(valid_mask, latents.shape[-2:])

        inv_total = latents.new_zeros(())
        sig_total = latents.new_zeros(())
        n_hooks = len(self._latent_sources)
        for source in self._latent_sources:
            lat = latents_by_source[source]
            mask_s = self._downsample_valid_mask(valid_mask, lat.shape[-2:])
            loss_valid_mask = mask_s if self.config.lejepa.invariance.loss_on_valid_only else None
            inv_total = inv_total + dense_invariance_loss(lat, valid_mask=loss_valid_mask)
            sig_total = sig_total + self._sigreg_loss(lat, loss_valid_mask)

        inv_loss = inv_total / n_hooks
        sigreg_loss = sig_total / n_hooks
        lambda_sigreg = self.config.lejepa.lambda_sigreg
        loss = (1.0 - lambda_sigreg) * inv_loss + lambda_sigreg * sigreg_loss
        return {
            "latents": latents,
            "latents_by_source": latents_by_source,
            "inv_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
            "loss": loss,
            "valid_mask": latent_valid_mask,
        }
