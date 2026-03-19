from __future__ import annotations

from pathlib import Path

import pytest
import torch

from local_conv_attention import BasicUNet, SwinUNet, build_model_from_yaml
from local_conv_attention.config import HEAUNetModelConfig


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "model_cls,model_name",
    [
        (BasicUNet, "basic_unet"),
        (SwinUNet, "swin_unet"),
    ],
)
def test_baseline_model_forward_shapes(model_cls: type[torch.nn.Module], model_name: str) -> None:
    config = HEAUNetModelConfig(
        name=model_name,
        in_channels=3,
        num_classes=2,
        base_channels=8,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
        swin_stage_heads=[2, 4, 4, 8],
    )
    model = model_cls(config)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 2, 64, 64)


@pytest.mark.parametrize(
    "config_name,expected_type",
    [
        ("basic_unet_small.yaml", "BasicUNet"),
        ("swin_unet_small.yaml", "SwinUNet"),
    ],
)
def test_build_baseline_model_from_yaml(config_name: str, expected_type: str) -> None:
    model = build_model_from_yaml(CONFIG_DIR / config_name)
    assert model.__class__.__name__ == expected_type
