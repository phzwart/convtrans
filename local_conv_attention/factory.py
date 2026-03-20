"""Factory helpers for segmentation and instance models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from .baselines import BasicUNet
from .config import (
    BottomUpInstanceLossConfig,
    HEAExperimentConfig,
    HEAUNetModelConfig,
    experiment_config_from_dict,
    load_experiment_config,
)
from .dense_lejepa import DenseLeJEPAModel
from .hybrid_dense_lejepa import HybridDenseLeJEPAModel
from .losses import BottomUpInstanceLoss
from .swin import SwinUNet
from .unet import HEAUNet, HEAUNetInstanceModel


def build_model(config: HEAUNetModelConfig | HEAExperimentConfig | dict[str, Any]) -> nn.Module:
    """Instantiate a configured segmentation model from a dataclass config or plain dict."""
    postprocess_config = None
    if isinstance(config, dict):
        config = experiment_config_from_dict(config)
    if isinstance(config, HEAExperimentConfig):
        postprocess_config = config.postprocess
        config = config.model
    if config.name == "hea_unet":
        return HEAUNet(config)
    if config.name == "basic_unet":
        return BasicUNet(config)
    if config.name == "swin_unet":
        return SwinUNet(config)
    if config.name == "hea_dense_lejepa":
        return DenseLeJEPAModel(config)
    if config.name == "hybrid_dense_lejepa":
        return HybridDenseLeJEPAModel(config)
    if config.name in {"hea_unet_instance", "basic_unet_instance"}:
        return HEAUNetInstanceModel(config, postprocess_config=postprocess_config)
    raise ValueError(f"Unsupported model name: {config.name}")


def build_model_from_yaml(path: str | Path) -> nn.Module:
    """Instantiate a configured segmentation model directly from a YAML config path."""
    return build_model(load_experiment_config(path))


def build_instance_loss(config: BottomUpInstanceLossConfig | HEAExperimentConfig) -> BottomUpInstanceLoss:
    """Instantiate the bottom-up instance loss from config."""
    if isinstance(config, HEAExperimentConfig):
        config = config.loss
    return BottomUpInstanceLoss(config)
