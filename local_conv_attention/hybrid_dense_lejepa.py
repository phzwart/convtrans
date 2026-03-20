"""Dense LeJEPA-style SSL for the hybrid conv + local-attention encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import HEAUNetModelConfig
from .dense_lejepa import DenseLatentProjector2d, _latent_source_module_key
from .hybrid_encoder import HybridConvAttentionEncoder
from .losses_ssl import dense_invariance_loss
from .sigreg import SIGRegLoss
from .views import DenseAlignedViewGenerator

HYBRID_LATENT_HOOK = "encoder_out"


class HybridDenseLeJEPAModel(nn.Module):
    """Teacher-free dense latent pretraining with :class:`HybridConvAttentionEncoder`.

    Same objective as :class:`DenseLeJEPAModel` (dense invariance + SIGReg), but the backbone
    is a single-scale encoder. Latent hooks must be ``encoder_out`` (the output of the last
    attention block, before any classification head).

    Configure via :class:`~local_conv_attention.config.HEAUNetModelConfig` with
    ``name="hybrid_dense_lejepa"`` and a non-null ``hybrid_encoder`` field.
    """

    def __init__(self, config: HEAUNetModelConfig) -> None:
        super().__init__()
        if config.name != "hybrid_dense_lejepa":
            raise ValueError(
                f"HybridDenseLeJEPAModel expects config.name='hybrid_dense_lejepa', got {config.name!r}."
            )
        if config.hybrid_encoder is None:
            raise ValueError("hybrid_dense_lejepa requires config.hybrid_encoder.")
        config.validate()
        self.config = config
        self.backbone = HybridConvAttentionEncoder(config.hybrid_encoder)
        self.view_generator = DenseAlignedViewGenerator(config.lejepa)
        self._latent_sources = config.latent.resolved_sources()
        feature_channels = config.hybrid_encoder.block.channels
        self.projectors = nn.ModuleDict(
            {
                _latent_source_module_key(s): DenseLatentProjector2d(
                    feature_channels,
                    config.latent.latent_dim,
                    depth=config.latent.projector_depth,
                    kernel_size=config.latent.projector_kernel_size,
                    activation=config.act,
                    normalize_latents=config.latent.normalize_latents,
                )
                for s in self._latent_sources
            }
        )
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

    def _encode_flat_views(self, flat_views: Tensor) -> Tensor:
        feats = self.backbone.forward_features(flat_views)
        return self.backbone.head(feats)

    def _latents_from_features(
        self,
        feat: Tensor,
        *,
        batch: int,
        num_views: int,
        sources_subset: list[str] | None = None,
    ) -> dict[str, Tensor]:
        names = sources_subset if sources_subset is not None else self._latent_sources
        latents_by_source: dict[str, Tensor] = {}
        for source in names:
            if source != HYBRID_LATENT_HOOK:
                raise ValueError(
                    f"Hybrid encoder LeJEPA only supports latent hook {HYBRID_LATENT_HOOK!r}, got {source!r}."
                )
            proj = self.projectors[_latent_source_module_key(source)](feat)
            _, dim, h, w = proj.shape
            latents_by_source[source] = proj.view(batch, num_views, dim, h, w)
        return latents_by_source

    def _sources_for_step(self, rotate_latent_index: int | None) -> list[str]:
        if self.config.latent.step_mode == "joint":
            return list(self._latent_sources)
        idx = 0 if rotate_latent_index is None else int(rotate_latent_index) % len(self._latent_sources)
        return [self._latent_sources[idx]]

    def forward(
        self,
        x: Tensor,
        *,
        valid_mask: Tensor | None = None,
        rotate_latent_index: int | None = None,
    ) -> dict[str, Any]:
        views, generated_valid_mask = self._prepare_views(x)
        if valid_mask is None:
            valid_mask = generated_valid_mask

        batch, num_views, channels, height, width = views.shape
        active_sources = self._sources_for_step(rotate_latent_index)

        if self.config.lejepa.sequential_view_forward:
            accum: dict[str, list[Tensor]] = {s: [] for s in active_sources}
            for view_index in range(num_views):
                chunk = views[:, view_index].contiguous()
                feat = self._encode_flat_views(chunk)
                partial = self._latents_from_features(
                    feat, batch=batch, num_views=1, sources_subset=active_sources
                )
                for s in active_sources:
                    accum[s].append(partial[s])
            latents_by_source = {s: torch.cat(accum[s], dim=1) for s in active_sources}
        else:
            flat_views = views.reshape(batch * num_views, channels, height, width)
            feat = self._encode_flat_views(flat_views)
            latents_by_source = self._latents_from_features(
                feat,
                batch=batch,
                num_views=num_views,
                sources_subset=active_sources,
            )

        primary = active_sources[0]
        latents = latents_by_source[primary]
        latent_valid_mask = self._downsample_valid_mask(valid_mask, latents.shape[-2:])

        inv_total = latents.new_zeros(())
        sig_total = latents.new_zeros(())
        n_terms = len(active_sources)
        for source in active_sources:
            lat = latents_by_source[source]
            mask_s = self._downsample_valid_mask(valid_mask, lat.shape[-2:])
            loss_valid_mask = mask_s if self.config.lejepa.invariance.loss_on_valid_only else None
            inv_total = inv_total + dense_invariance_loss(lat, valid_mask=loss_valid_mask)
            sig_total = sig_total + self._sigreg_loss(lat, loss_valid_mask)

        inv_loss = inv_total / n_terms
        sigreg_loss = sig_total / n_terms
        lambda_sigreg = self.config.lejepa.lambda_sigreg
        loss = (1.0 - lambda_sigreg) * inv_loss + lambda_sigreg * sigreg_loss
        return {
            "latents": latents,
            "latents_by_source": latents_by_source,
            "active_latent_source": primary,
            "inv_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
            "loss": loss,
            "valid_mask": latent_valid_mask,
        }
