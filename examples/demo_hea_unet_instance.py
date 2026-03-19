from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_conv_attention import (
    BottomUpInstanceLoss,
    HEAExperimentConfig,
    HEAUNetInstanceModel,
    HEAUNetModelConfig,
    build_instance_targets,
)


def main() -> None:
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
    model = HEAUNetInstanceModel(experiment.model, postprocess_config=experiment.postprocess)
    criterion = BottomUpInstanceLoss(experiment.loss)

    x = torch.randn(2, 1, 64, 64)
    instance_labels = torch.zeros(2, 64, 64, dtype=torch.long)
    instance_labels[0, 12:24, 10:22] = 1
    instance_labels[0, 34:50, 38:54] = 2
    instance_labels[1, 20:40, 20:40] = 1

    targets = build_instance_targets(instance_labels, experiment.targets)
    predictions = model(x, return_features=True)
    loss_dict = criterion(predictions, targets)

    print("foreground:", tuple(predictions["foreground_logits"].shape))
    print("center:", tuple(predictions["center_logits"].shape))
    print("offsets:", tuple(predictions["offsets"].shape))
    print("H_star:", tuple(predictions["H_star"].shape))
    print("loss:", float(loss_dict["loss"].item()))


if __name__ == "__main__":
    main()
