from __future__ import annotations

import torch

from local_conv_attention import (
    BottomUpInstanceLoss,
    HEAExperimentConfig,
    HEAUNetInstanceModel,
    HEAUNetModelConfig,
    build_instance_targets,
)


def test_hea_unet_instance_forward_shapes() -> None:
    config = HEAUNetModelConfig(
        name="hea_unet_instance",
        in_channels=1,
        num_classes=2,
        base_channels=8,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
    )
    model = HEAUNetInstanceModel(config)
    outputs = model(torch.randn(2, 1, 64, 64), return_features=True)
    assert outputs["foreground_logits"].shape == (2, 1, 64, 64)
    assert outputs["center_logits"].shape == (2, 1, 64, 64)
    assert outputs["offsets"].shape == (2, 2, 64, 64)
    assert outputs["H_star"].shape == (2, 8, 64, 64)


def test_basic_unet_instance_forward_shapes() -> None:
    config = HEAUNetModelConfig(
        name="basic_unet_instance",
        in_channels=3,
        num_classes=2,
        base_channels=8,
        channel_multipliers=[1, 2, 4, 8],
        encoder_depths=[1, 1, 1, 1],
        decoder_depths=[1, 1, 1],
        use_raw_skips=True,
    )
    model = HEAUNetInstanceModel(config)
    outputs = model(torch.randn(1, 3, 48, 48))
    assert outputs["foreground_logits"].shape == (1, 1, 48, 48)


def test_instance_model_backward_end_to_end() -> None:
    experiment = HEAExperimentConfig(
        model=HEAUNetModelConfig(
            name="hea_unet_instance",
            in_channels=1,
            num_classes=2,
            base_channels=8,
            channel_multipliers=[1, 2, 4, 8],
            encoder_depths=[1, 1, 1, 1],
            decoder_depths=[1, 1, 1],
        )
    )
    model = HEAUNetInstanceModel(experiment.model)
    criterion = BottomUpInstanceLoss(experiment.loss)
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    labels = torch.zeros(2, 32, 32, dtype=torch.long)
    labels[0, 4:12, 4:12] = 1
    labels[1, 16:24, 12:20] = 1
    targets = build_instance_targets(labels, experiment.targets)

    loss = criterion(model(x), targets)["loss"]
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)
