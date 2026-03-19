"""Hierarchical Elevator Attention U-Net architectures."""

from __future__ import annotations

from typing import Any, Sequence

from torch import Tensor, nn

from .backbone import HEABackbone
from .baselines import BasicUNet
from .config import HEAUNetModelConfig, InstancePostprocessConfig
from .instance_head import BottomUpInstanceHead2d
from .postprocess import decode_instances


class HEAUNet(nn.Module):
    """Configurable segmentation model with Hierarchical Elevator Attention fusion."""

    def __init__(self, config: HEAUNetModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = HEABackbone(config)
        self.channels = self.backbone.channels
        self.segmentation_head = nn.Conv2d(self.channels[0], config.num_classes, kernel_size=1)

    def _build_semantic_memories(self, encoder_features: Sequence[Tensor]) -> dict[int, Tensor]:
        return self.backbone._build_semantic_memories(encoder_features)

    def _progressive_elevator(self, memories: dict[int, Tensor]) -> dict[int, Tensor]:
        return self.backbone._progressive_elevator(memories)

    def encode_features(self, x: Tensor) -> list[Tensor]:
        return self.backbone.encode_features(x)

    def decode_with_memories(
        self,
        encoder_features: Sequence[Tensor],
        memories: dict[int, Tensor],
        *,
        debug_stage: int | None = None,
        target_slice: tuple[tuple[int, int], tuple[int, int]] | None = None,
        batch_index: int = 0,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        top_feature, stage_debug, _ = self.backbone.decode_with_memories(
            encoder_features,
            memories,
            debug_stage=debug_stage,
            target_slice=target_slice,
            batch_index=batch_index,
        )
        return top_feature, stage_debug

    def forward_features(self, x: Tensor) -> tuple[Tensor, dict[int, Tensor], list[Tensor]]:
        features = self.backbone.forward_features(x)
        return features["top_feature"], features["memories"], features["encoder_features"]

    def forward_with_stage_debug(
        self,
        x: Tensor,
        *,
        stage: int = 0,
        target_slice: tuple[tuple[int, int], tuple[int, int]] | None = None,
        batch_index: int = 0,
    ) -> dict[str, Any]:
        features = self.backbone.forward_with_stage_debug(
            x,
            stage=stage,
            target_slice=target_slice,
            batch_index=batch_index,
        )
        output = self.segmentation_head(features["top_feature"])
        return {
            "output": output,
            "features": features["top_feature"],
            "memories": features["memories"],
            "encoder_features": features["encoder_features"],
            "stage_debug": features["stage_debug"],
        }

    def forward(self, x: Tensor) -> Tensor:
        features, _, _ = self.forward_features(x)
        return self.segmentation_head(features)


class HEAUNetInstanceModel(nn.Module):
    """Bottom-up instance segmentation model built on a configurable U-Net trunk."""

    def __init__(
        self,
        config: HEAUNetModelConfig,
        *,
        postprocess_config: InstancePostprocessConfig | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.postprocess_config = postprocess_config

        trunk_config = config.as_trunk_config()
        if trunk_config.name == "hea_unet":
            self.trunk: nn.Module = HEAUNet(trunk_config)
        elif trunk_config.name == "basic_unet":
            self.trunk = BasicUNet(trunk_config)
        else:
            raise ValueError(f"Unsupported instance trunk: {trunk_config.name}")

        self.instance_head = BottomUpInstanceHead2d(
            in_channels=self.trunk.channels[0],
            config=config.instance_head,
            norm=config.norm,
            act=config.act,
            num_classes=config.num_classes,
        )

    def forward(
        self,
        x: Tensor,
        *,
        return_features: bool = False,
        postprocess: bool = False,
    ) -> dict[str, Tensor]:
        trunk_outputs = self.trunk.forward_features(x)
        h_star = trunk_outputs[0]
        outputs = self.instance_head(h_star)
        if return_features:
            outputs["H_star"] = h_star
            if len(trunk_outputs) > 1:
                outputs["trunk_debug"] = trunk_outputs[1:]
        if postprocess:
            if self.postprocess_config is None:
                raise ValueError("postprocess=True requires a postprocess config.")
            outputs.update(decode_instances(outputs, self.postprocess_config))
        return outputs

    def decode(
        self,
        predictions: dict[str, Tensor],
        *,
        postprocess_config: InstancePostprocessConfig | None = None,
    ) -> dict[str, Tensor]:
        config = postprocess_config or self.postprocess_config
        if config is None:
            raise ValueError("A postprocess config is required to decode instances.")
        return decode_instances(predictions, config)
