from __future__ import annotations

from pathlib import Path

from local_conv_attention import (
    HEAExperimentConfig,
    HEAUNetModelConfig,
    build_model,
    build_model_from_yaml,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_instance_model_builds_from_yaml() -> None:
    model = build_model_from_yaml(CONFIG_DIR / "hea_unet_instance_default.yaml")
    assert model.__class__.__name__ == "HEAUNetInstanceModel"


def test_instance_model_builds_from_python_config() -> None:
    experiment = HEAExperimentConfig(
        model=HEAUNetModelConfig(
            name="basic_unet_instance",
            in_channels=1,
            num_classes=2,
            base_channels=8,
            channel_multipliers=[1, 2, 4, 8],
            encoder_depths=[1, 1, 1, 1],
            decoder_depths=[1, 1, 1],
        )
    )
    experiment.model.trunk.type = "basic_unet"
    model = build_model(experiment)
    assert model.config.name == "basic_unet_instance"


def test_instance_trunk_overrides_apply() -> None:
    experiment = HEAExperimentConfig(
        model=HEAUNetModelConfig(
            name="hea_unet_instance",
            base_channels=8,
            channel_multipliers=[1, 2, 4, 8],
            encoder_depths=[1, 1, 1, 1],
            decoder_depths=[1, 1, 1],
        )
    )
    experiment.model.trunk.hea_enabled = False
    experiment.model.trunk.elevator_mode = "progressive"
    experiment.validate()
    assert experiment.model.hea.enabled_decoder_stages == []
    assert experiment.model.attention.elevator_mode == "progressive"
