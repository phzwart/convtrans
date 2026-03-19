"""Losses for bottom-up instance segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import BottomUpInstanceLossConfig


def _dice_loss(logits: Tensor, target: Tensor, smooth: float) -> Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, target.ndim))
    intersection = (probs * target).sum(dim=dims)
    total = probs.sum(dim=dims) + target.sum(dim=dims)
    score = (2 * intersection + smooth) / (total + smooth)
    return 1.0 - score.mean()


class BottomUpInstanceLoss(nn.Module):
    """A readable multi-task loss for dense instance segmentation outputs."""

    def __init__(self, config: BottomUpInstanceLossConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

    def _foreground_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        if self.config.foreground.type == "bce":
            return bce
        return bce + _dice_loss(logits, target, self.config.foreground.dice_smooth)

    def _center_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        if self.config.center.type == "bce":
            return F.binary_cross_entropy_with_logits(logits, target)
        return F.mse_loss(torch.sigmoid(logits), target)

    def _offset_loss(self, offsets: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        if self.config.offset.type == "smooth_l1":
            per_pixel = F.smooth_l1_loss(
                offsets,
                target,
                reduction="none",
                beta=self.config.offset.beta,
            )
        else:
            per_pixel = F.l1_loss(offsets, target, reduction="none")
        weighted = per_pixel * weight
        normalizer = weight.sum() * offsets.size(1)
        if normalizer <= 0:
            return offsets.sum() * 0.0
        return weighted.sum() / normalizer

    def _semantic_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            logits,
            target.long(),
            ignore_index=self.config.semantic.ignore_index,
        )

    def forward(self, predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        center_logits = (
            predictions["center_logits"]
            if "center_logits" in predictions
            else predictions["center_heatmap_logits"]
        )
        foreground_loss = self._foreground_loss(
            predictions["foreground_logits"],
            targets["foreground_target"],
        )
        center_loss = self._center_loss(center_logits, targets["center_target"])
        offset_weight = targets.get("offset_weight", targets["valid_mask"])
        offset_loss = self._offset_loss(
            predictions["offsets"],
            targets["offset_target"],
            offset_weight,
        )

        total = (
            self.config.foreground.weight * foreground_loss
            + self.config.center.weight * center_loss
            + self.config.offset.weight * offset_loss
        )
        output = {
            "loss": total,
            "foreground_loss": foreground_loss,
            "center_loss": center_loss,
            "offset_loss": offset_loss,
        }

        if "semantic_logits" in predictions and "semantic_target" in targets:
            semantic_loss = self._semantic_loss(predictions["semantic_logits"], targets["semantic_target"])
            total = total + self.config.semantic.weight * semantic_loss
            output["semantic_loss"] = semantic_loss
            output["loss"] = total
        return output
