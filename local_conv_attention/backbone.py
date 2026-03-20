"""Reusable HEA backbone without task heads."""

from __future__ import annotations

from typing import Any, Sequence

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .block import LocalTransformerBlock2d
from .config import HEAUNetModelConfig
from .decoder import HEADecoderStage
from .encoder import ConvStem2d, HEAEncoderStage
from .hea import HEAFusionBlock2d, SemanticMemoryBlock2d


class HEABackbone(nn.Module):
    """Reusable HEA U-Net trunk that returns dense spatial features."""

    def __init__(self, config: HEAUNetModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self._grad_ckpt = bool(config.backbone_gradient_checkpointing)
        # Primary hook for ``forward()`` / ``forward_features``[\"latent_feature\"]; multi-hook LeJEPA uses ``resolve_latent_tensor``.
        self.latent_source = config.latent.resolved_sources()[0]

        channels = [config.base_channels * mult for mult in config.channel_multipliers]
        self.channels = channels
        self.stem = ConvStem2d(
            config.in_channels,
            channels[0],
            norm=config.norm,
            act=config.act,
        )

        self.encoder_stages = nn.ModuleList()
        self.encoder_stages.append(
            HEAEncoderStage(
                channels[0],
                channels[0],
                depth=config.encoder_depths[0],
                downsample=False,
                norm=config.norm,
                act=config.act,
            )
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
                LocalTransformerBlock2d(
                    dim=channels[-1],
                    num_heads=config.attention.heads,
                    window_size=config.bottleneck_window_size,
                    dilation=config.bottleneck_dilation,
                    implementation=config.attention.operator_backend,
                    boundary_pad=config.attention.local_attention_boundary_pad,
                )
                for _ in range(config.bottleneck_depth)
            ]
        )

        memory_cfg = config.semantic_memory
        self.memory_scale_settings = {
            scale: {
                "depth": depth,
                "window": window,
                "dilation": dilation,
            }
            for scale, depth, window, dilation in zip(
                memory_cfg.enabled_scales,
                memory_cfg.block_depths,
                memory_cfg.window_sizes,
                memory_cfg.dilations,
            )
        }
        self.semantic_memory_blocks = nn.ModuleDict(
            {
                str(scale): SemanticMemoryBlock2d(
                    dim=channels[scale],
                    depth=self.memory_scale_settings[scale]["depth"],
                    num_heads=config.attention.heads,
                    window_size=self.memory_scale_settings[scale]["window"],
                    dilation=self.memory_scale_settings[scale]["dilation"],
                    implementation=config.attention.operator_backend,
                    boundary_pad=config.attention.local_attention_boundary_pad,
                    use_local_transformer_block=memory_cfg.use_local_transformer_block,
                    norm=config.norm,
                    act=config.act,
                )
                for scale in memory_cfg.enabled_scales
            }
        )

        hea_window_by_scale = {
            scale: window
            for scale, window in zip(
                memory_cfg.enabled_scales,
                config.hea.per_scale_window_sizes,
            )
        }
        hea_dilation_by_scale = {
            scale: dilation
            for scale, dilation in zip(
                memory_cfg.enabled_scales,
                config.hea.per_scale_dilations,
            )
        }

        self.progressive_blocks = nn.ModuleDict()
        if config.attention.elevator_mode == "progressive":
            sorted_scales = sorted(memory_cfg.enabled_scales, reverse=True)
            for target_scale in sorted(sorted_scales[1:]):
                source_scales = sorted(scale for scale in sorted_scales if scale > target_scale)
                self.progressive_blocks[str(target_scale)] = HEAFusionBlock2d(
                    query_dim=channels[target_scale],
                    memory_dims=[channels[scale] for scale in source_scales],
                    scale_factors=[2 ** (scale - target_scale) for scale in source_scales],
                    num_heads=config.attention.heads,
                    head_dim=config.attention.head_dim,
                    window_sizes=[hea_window_by_scale[scale] for scale in source_scales],
                    dilations=[hea_dilation_by_scale[scale] for scale in source_scales],
                    implementation=config.attention.operator_backend,
                    boundary_pad=config.attention.local_attention_boundary_pad,
                    fusion_mode=config.attention.fusion_mode,
                    residual_fusion=config.hea.fusion,
                    qkv_bias=config.attention.qkv_bias,
                    project_context=config.hea.project_context,
                    joint_scale_projection=config.hea.joint_scale_projection,
                )

        self.decoder_stages = nn.ModuleDict()
        self.decoder_hea_blocks = nn.ModuleDict()
        for target_scale in reversed(range(len(channels) - 1)):
            in_channels = channels[target_scale + 1]
            out_channels = channels[target_scale]
            decoder_depth_index = len(channels) - 2 - target_scale
            self.decoder_stages[str(target_scale)] = HEADecoderStage(
                in_channels=in_channels,
                out_channels=out_channels,
                skip_channels=channels[target_scale],
                depth=config.decoder_depths[decoder_depth_index],
                use_raw_skip=config.use_raw_skips,
                norm=config.norm,
                act=config.act,
            )

            if target_scale in config.hea.enabled_decoder_stages:
                source_scales = [
                    scale for scale in memory_cfg.enabled_scales if scale > target_scale
                ]
                if source_scales:
                    self.decoder_hea_blocks[str(target_scale)] = HEAFusionBlock2d(
                        query_dim=channels[target_scale],
                        memory_dims=[channels[scale] for scale in source_scales],
                        scale_factors=[2 ** (scale - target_scale) for scale in source_scales],
                        num_heads=config.attention.heads,
                        head_dim=config.attention.head_dim,
                        window_sizes=[hea_window_by_scale[scale] for scale in source_scales],
                        dilations=[hea_dilation_by_scale[scale] for scale in source_scales],
                        implementation=config.attention.operator_backend,
                        boundary_pad=config.attention.local_attention_boundary_pad,
                        fusion_mode=config.attention.fusion_mode,
                        residual_fusion=config.hea.fusion,
                        qkv_bias=config.attention.qkv_bias,
                        project_context=config.hea.project_context,
                        joint_scale_projection=config.hea.joint_scale_projection,
                    )

    def _build_semantic_memories(self, encoder_features: Sequence[Tensor]) -> dict[int, Tensor]:
        memories: dict[int, Tensor] = {}
        for scale, block in self.semantic_memory_blocks.items():
            scale_index = int(scale)
            memories[scale_index] = block(encoder_features[scale_index])
        return memories

    def _progressive_elevator(self, memories: dict[int, Tensor]) -> dict[int, Tensor]:
        if self.config.attention.elevator_mode != "progressive":
            return memories

        enriched = dict(memories)
        for target_scale in sorted(self.progressive_blocks.keys(), key=int, reverse=True):
            scale_index = int(target_scale)
            source_scales = sorted([scale for scale in enriched if scale > scale_index])
            enriched[scale_index] = self.progressive_blocks[target_scale](
                enriched[scale_index],
                [enriched[scale] for scale in source_scales],
            )
        return enriched

    def resolve_latent_tensor(
        self,
        source: str,
        *,
        top_feature: Tensor,
        encoder_features: Sequence[Tensor],
        decoder_features: dict[int, Tensor],
    ) -> Tensor:
        if source == "top":
            return top_feature
        if source == "bottleneck":
            return encoder_features[-1]
        if source.startswith("encoder_"):
            scale = int(source.split("_", maxsplit=1)[1])
            return encoder_features[scale]
        if source.startswith("decoder_"):
            scale = int(source.split("_", maxsplit=1)[1])
            return decoder_features[scale]
        raise ValueError(f"Unsupported latent source {source!r}.")

    def _resolve_latent_feature(
        self,
        *,
        top_feature: Tensor,
        encoder_features: Sequence[Tensor],
        decoder_features: dict[int, Tensor],
    ) -> Tensor:
        return self.resolve_latent_tensor(
            self.latent_source,
            top_feature=top_feature,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
        )

    def encode_features(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        encoder_features = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)
        encoder_features[-1] = self.bottleneck(encoder_features[-1])
        return encoder_features

    def decode_with_memories(
        self,
        encoder_features: Sequence[Tensor],
        memories: dict[int, Tensor],
        *,
        debug_stage: int | None = None,
        target_slice: tuple[tuple[int, int], tuple[int, int]] | None = None,
        batch_index: int = 0,
    ) -> tuple[Tensor, dict[str, Any] | None, dict[int, Tensor]]:
        stage_debug: dict[str, Any] | None = None
        decoder_features: dict[int, Tensor] = {}
        x = encoder_features[-1]
        for target_scale in reversed(range(len(self.channels) - 1)):
            reference = encoder_features[target_scale]
            x = self.decoder_stages[str(target_scale)](x, reference)
            if str(target_scale) in self.decoder_hea_blocks:
                block = self.decoder_hea_blocks[str(target_scale)]
                source_scales = sorted([scale for scale in memories if scale > target_scale])
                if debug_stage is not None and debug_stage == target_scale:
                    block_result = block(
                        x,
                        [memories[scale] for scale in source_scales],
                        return_debug=True,
                        target_slice=target_slice,
                        batch_index=batch_index,
                    )
                    x, block_debug = block_result
                    stage_debug = {
                        **block_debug,
                        "decoder_stage": target_scale,
                        "source_scales": source_scales,
                    }
                else:
                    x = block(x, [memories[scale] for scale in source_scales])
            decoder_features[target_scale] = x
        return x, stage_debug, decoder_features

    def _decode_after_encode(self, *encoder_features: Tensor) -> tuple[Tensor, dict[int, Tensor], dict[int, Tensor]]:
        enc_list = list(encoder_features)
        memories = self._progressive_elevator(self._build_semantic_memories(enc_list))
        top_feature, _, decoder_features = self.decode_with_memories(enc_list, memories)
        return top_feature, memories, decoder_features

    def forward_features(self, x: Tensor) -> dict[str, Any]:
        if self._grad_ckpt:
            encoder_features = list(
                checkpoint(
                    lambda inp: tuple(self.encode_features(inp)),
                    x,
                    use_reentrant=False,
                )
            )
            top_feature, memories, decoder_features = checkpoint(
                lambda *enc: self._decode_after_encode(*enc),
                *encoder_features,
                use_reentrant=False,
            )
        else:
            encoder_features = self.encode_features(x)
            memories = self._progressive_elevator(self._build_semantic_memories(encoder_features))
            top_feature, _, decoder_features = self.decode_with_memories(encoder_features, memories)

        latent_feature = self._resolve_latent_feature(
            top_feature=top_feature,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
        )
        return {
            "top_feature": top_feature,
            "latent_feature": latent_feature,
            "memories": memories,
            "encoder_features": encoder_features,
            "decoder_features": decoder_features,
        }

    def forward_with_stage_debug(
        self,
        x: Tensor,
        *,
        stage: int = 0,
        target_slice: tuple[tuple[int, int], tuple[int, int]] | None = None,
        batch_index: int = 0,
    ) -> dict[str, Any]:
        if self._grad_ckpt:
            raise RuntimeError(
                "forward_with_stage_debug does not support backbone_gradient_checkpointing; "
                "set backbone_gradient_checkpointing=False for debug forwards."
            )
        encoder_features = self.encode_features(x)
        memories = self._progressive_elevator(self._build_semantic_memories(encoder_features))
        top_feature, stage_debug, decoder_features = self.decode_with_memories(
            encoder_features,
            memories,
            debug_stage=stage,
            target_slice=target_slice,
            batch_index=batch_index,
        )
        if stage_debug is None:
            raise ValueError(f"No HEA decoder block is available at stage {stage}.")
        latent_feature = self._resolve_latent_feature(
            top_feature=top_feature,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
        )
        return {
            "top_feature": top_feature,
            "latent_feature": latent_feature,
            "memories": memories,
            "encoder_features": list(encoder_features),
            "decoder_features": decoder_features,
            "stage_debug": stage_debug,
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)["latent_feature"]
