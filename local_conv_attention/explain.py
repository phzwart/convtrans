"""Region-level explanation utilities for HEA and HEA-UNet models.

These tools explain HEA's contextual retrieval from low-resolution semantic
memories into a chosen high-resolution decoder stage. They expose where the
HEA block looked, how much each retrieved region contributed after fusion, and
optionally how those contributions align with a simple top-stage segmentation
logit. They are not a full causal explanation of the entire model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import Tensor

from .unet import HEAUNet, HEAUNetInstanceModel
from .visualization import combine_upsampled_heatmaps, visualize_explanation


ExplainMode = Literal["attention", "gated_magnitude", "signed_logit"]


@dataclass
class _ResolvedTarget:
    target_slice: tuple[tuple[int, int], tuple[int, int]]
    target_xy: tuple[int, int] | None


def _normalize_target_slice(
    height: int,
    width: int,
    *,
    target_xy: tuple[int, int] | None = None,
    target_patch: tuple[tuple[int, int], tuple[int, int]] | None = None,
    center_xy: tuple[int, int] | None = None,
    patch_radius: int | None = None,
) -> _ResolvedTarget:
    """Resolve a target pixel or patch into a clamped spatial slice."""
    if target_patch is not None:
        (y0, y1), (x0, x1) = target_patch
        y0 = max(0, min(height, y0))
        y1 = max(y0, min(height, y1))
        x0 = max(0, min(width, x0))
        x1 = max(x0, min(width, x1))
        center = ((y0 + y1 - 1) // 2, (x0 + x1 - 1) // 2)
        return _ResolvedTarget(target_slice=((y0, y1), (x0, x1)), target_xy=center)
    if target_xy is not None:
        y, x = target_xy
        y = max(0, min(height - 1, y))
        x = max(0, min(width - 1, x))
        return _ResolvedTarget(target_slice=((y, y + 1), (x, x + 1)), target_xy=(y, x))
    if center_xy is not None and patch_radius is not None:
        y, x = center_xy
        return _normalize_target_slice(
            height,
            width,
            target_patch=((y - patch_radius, y + patch_radius + 1), (x - patch_radius, x + patch_radius + 1)),
        )
    raise ValueError("Provide target_xy, target_patch, or center_xy with patch_radius.")


def _aggregate_scores_to_heatmap(
    scores: Tensor,
    candidate_coords: Tensor,
    candidate_valid: Tensor,
    memory_shape: tuple[int, int],
) -> Tensor:
    """Scatter patch-local candidate scores back onto the coarse memory lattice."""
    heatmap = torch.zeros(memory_shape, device=scores.device, dtype=scores.dtype)
    coords = candidate_coords.reshape(-1, 2).long()
    values = scores.permute(1, 2, 0).reshape(-1)
    valid = candidate_valid.reshape(-1)
    if valid.any():
        coords = coords[valid]
        values = values[valid]
        heatmap.index_put_((coords[:, 0], coords[:, 1]), values, accumulate=True)
    return heatmap


def _extract_patch_score(
    output: Tensor,
    *,
    batch_index: int,
    target_slice: tuple[tuple[int, int], tuple[int, int]],
    channel: int,
) -> Tensor:
    """Return the mean logit over a selected pixel or patch."""
    (y0, y1), (x0, x1) = target_slice
    return output[batch_index, channel, y0:y1, x0:x1].mean()


class HEAExplainer:
    """Region-level explanation helper for HEA-UNet models."""

    def __init__(self, model: HEAUNet | HEAUNetInstanceModel) -> None:
        self.model = model.eval()

    def _resolve_stage(self, stage: int | str) -> int:
        if stage == "top":
            return 0
        if isinstance(stage, int):
            return stage
        raise ValueError(f"Unsupported stage selector {stage!r}.")

    def _resolve_trunk(self) -> HEAUNet:
        if isinstance(self.model, HEAUNet):
            return self.model
        if isinstance(self.model, HEAUNetInstanceModel) and isinstance(self.model.trunk, HEAUNet):
            return self.model.trunk
        raise TypeError("HEAExplainer requires an HEAUNet or HEAUNetInstanceModel with an HEA trunk.")

    def _forward_with_debug(
        self,
        x: Tensor,
        *,
        stage: int,
        target_slice: tuple[tuple[int, int], tuple[int, int]],
        batch_index: int,
        output_key: str | None,
    ) -> dict[str, Any]:
        trunk = self._resolve_trunk()
        result = trunk.forward_with_stage_debug(
            x,
            stage=stage,
            target_slice=target_slice,
            batch_index=batch_index,
        )
        if isinstance(self.model, HEAUNet):
            result["model_output"] = result["output"]
            return result
        output_dict = self.model.instance_head(result["features"])
        if output_key is not None and output_key not in output_dict:
            raise KeyError(f"Unknown instance output key {output_key!r}.")
        result["model_output"] = output_dict
        return result

    def _target_tensor(
        self,
        result: Mapping[str, Any],
        *,
        output_key: str | None,
    ) -> Tensor:
        model_output = result["model_output"]
        if isinstance(model_output, Tensor):
            return model_output
        if output_key is None:
            raise ValueError("Instance-style outputs require output_key to select a target tensor.")
        return model_output[output_key]

    def _signed_head_weight(
        self,
        *,
        stage: int,
        target_channel: int,
    ) -> Tensor:
        if not isinstance(self.model, HEAUNet):
            raise ValueError("signed_logit explanations are only supported for simple HEA-UNet segmentation heads.")
        if stage != 0:
            raise ValueError("signed_logit explanations are only supported for the top decoder HEA stage.")
        head = self.model.segmentation_head
        if head.kernel_size != (1, 1):
            raise ValueError("signed_logit explanations require a 1x1 segmentation head.")
        return head.weight[target_channel, :, 0, 0].detach()

    def _per_scale_scores(
        self,
        *,
        stage_debug: Mapping[str, Any],
        mode: ExplainMode,
        score_norm: Literal["l1", "l2"],
        signed_weight: Tensor | None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor] | None]:
        per_scale_heatmaps: dict[int, Tensor] = {}
        raw_scores: dict[int, Tensor] = {}
        signed_parts: dict[int, Tensor] | None = {} if mode == "signed_logit" else None

        for source_scale, scale_debug in zip(stage_debug["source_scales"], stage_debug["per_scale"]):
            if mode == "attention":
                score_tensor = scale_debug["attention"].mean(dim=0)
            elif mode == "gated_magnitude":
                fused = scale_debug["fused_candidates"]
                norm_value = 1 if score_norm == "l1" else 2
                score_tensor = torch.linalg.vector_norm(fused, ord=norm_value, dim=1)
            else:
                if signed_weight is None:
                    raise ValueError("signed_logit mode requires a target logit weight.")
                fused = scale_debug["fused_candidates"]
                score_tensor = (fused * signed_weight.view(1, -1, 1, 1)).sum(dim=1)

            heatmap = _aggregate_scores_to_heatmap(
                score_tensor,
                scale_debug["candidate_coords"],
                scale_debug["candidate_valid"],
                scale_debug["memory_shape"],
            )
            per_scale_heatmaps[source_scale] = heatmap.detach()
            raw_scores[source_scale] = score_tensor.detach()
            if signed_parts is not None:
                signed_parts[source_scale] = heatmap.detach()

        return per_scale_heatmaps, raw_scores, signed_parts

    def _explain(
        self,
        x: Tensor,
        *,
        target_xy: tuple[int, int] | None,
        target_patch: tuple[tuple[int, int], tuple[int, int]] | None,
        center_xy: tuple[int, int] | None,
        patch_radius: int | None,
        batch_index: int,
        stage: int | str,
        mode: ExplainMode,
        target_channel: int,
        output_key: str | None,
        score_norm: Literal["l1", "l2"],
    ) -> dict[str, Any]:
        stage_index = self._resolve_stage(stage)
        resolved = _normalize_target_slice(
            x.size(-2),
            x.size(-1),
            target_xy=target_xy,
            target_patch=target_patch,
            center_xy=center_xy,
            patch_radius=patch_radius,
        )
        result = self._forward_with_debug(
            x,
            stage=stage_index,
            target_slice=resolved.target_slice,
            batch_index=batch_index,
            output_key=output_key,
        )
        output_tensor = self._target_tensor(result, output_key=output_key)
        target_logit = _extract_patch_score(
            output_tensor,
            batch_index=batch_index,
            target_slice=resolved.target_slice,
            channel=target_channel,
        )
        signed_weight = None
        if mode == "signed_logit":
            signed_weight = self._signed_head_weight(stage=stage_index, target_channel=target_channel)

        stage_debug = result["stage_debug"]
        per_scale_heatmaps, raw_scores, signed_scores = self._per_scale_scores(
            stage_debug=stage_debug,
            mode=mode,
            score_norm=score_norm,
            signed_weight=signed_weight,
        )
        scale_factor_map = {
            scale: scale_debug["scale_factor"]
            for scale, scale_debug in zip(stage_debug["source_scales"], stage_debug["per_scale"])
        }
        combined = combine_upsampled_heatmaps(
            {
                scale_factor_map[scale]: heatmap
                for scale, heatmap in per_scale_heatmaps.items()
            },
            output_shape=x.shape[-2:],
        )

        positive_heatmap = None
        negative_heatmap = None
        if mode == "signed_logit":
            positive_parts = {
                scale: heatmap.clamp_min(0.0)
                for scale, heatmap in per_scale_heatmaps.items()
            }
            negative_parts = {
                scale: (-heatmap.clamp_max(0.0))
                for scale, heatmap in per_scale_heatmaps.items()
            }
            positive_heatmap = combine_upsampled_heatmaps(
                {
                    scale_factor_map[scale]: heatmap
                    for scale, heatmap in positive_parts.items()
                },
                output_shape=x.shape[-2:],
            )
            negative_heatmap = combine_upsampled_heatmaps(
                {
                    scale_factor_map[scale]: heatmap
                    for scale, heatmap in negative_parts.items()
                },
                output_shape=x.shape[-2:],
            )

        return {
            "target_xy": resolved.target_xy,
            "target_patch": resolved.target_slice,
            "per_scale_heatmaps": per_scale_heatmaps,
            "combined_heatmap": combined.detach(),
            "coarse_region_coords": {
                scale: scale_debug["candidate_coords"].detach()
                for scale, scale_debug in zip(stage_debug["source_scales"], stage_debug["per_scale"])
            },
            "attention_weights": {
                scale: scale_debug["attention"].detach()
                for scale, scale_debug in zip(stage_debug["source_scales"], stage_debug["per_scale"])
            },
            "gated_scores": raw_scores if mode == "gated_magnitude" else None,
            "signed_scores": signed_scores if mode == "signed_logit" else None,
            "gate_value": stage_debug["gate"],
            "target_logit": target_logit.detach(),
            "positive_heatmap": positive_heatmap.detach() if positive_heatmap is not None else None,
            "negative_heatmap": negative_heatmap.detach() if negative_heatmap is not None else None,
            "metadata": {
                "stage": stage_index,
                "mode": mode,
                "batch_index": batch_index,
                "target_channel": target_channel,
                "source_scales": list(stage_debug["source_scales"]),
                "scale_factors": scale_factor_map,
                "output_key": output_key,
                "score_norm": score_norm,
            },
            "stage_debug": stage_debug,
            "model_output": output_tensor.detach(),
        }

    def explain_pixel(
        self,
        x: Tensor,
        *,
        target_xy: tuple[int, int],
        batch_index: int = 0,
        stage: int | str = "top",
        mode: ExplainMode = "gated_magnitude",
        target_channel: int = 0,
        output_key: str | None = None,
        score_norm: Literal["l1", "l2"] = "l1",
    ) -> dict[str, Any]:
        """Explain a single output pixel."""
        return self._explain(
            x,
            target_xy=target_xy,
            target_patch=None,
            center_xy=None,
            patch_radius=None,
            batch_index=batch_index,
            stage=stage,
            mode=mode,
            target_channel=target_channel,
            output_key=output_key,
            score_norm=score_norm,
        )

    def explain_patch(
        self,
        x: Tensor,
        *,
        target_patch: tuple[tuple[int, int], tuple[int, int]] | None = None,
        center_xy: tuple[int, int] | None = None,
        patch_radius: int | None = None,
        batch_index: int = 0,
        stage: int | str = "top",
        mode: ExplainMode = "gated_magnitude",
        target_channel: int = 0,
        output_key: str | None = None,
        score_norm: Literal["l1", "l2"] = "l1",
    ) -> dict[str, Any]:
        """Explain a small output patch by aggregating per-pixel contributions."""
        return self._explain(
            x,
            target_xy=None,
            target_patch=target_patch,
            center_xy=center_xy,
            patch_radius=patch_radius,
            batch_index=batch_index,
            stage=stage,
            mode=mode,
            target_channel=target_channel,
            output_key=output_key,
            score_norm=score_norm,
        )

    def ablate_region(
        self,
        x: Tensor,
        *,
        memory_scale: int,
        coarse_coord: tuple[int, int],
        target_xy: tuple[int, int] | None = None,
        target_patch: tuple[tuple[int, int], tuple[int, int]] | None = None,
        center_xy: tuple[int, int] | None = None,
        patch_radius: int | None = None,
        stage: int | str = "top",
        batch_index: int = 0,
        target_channel: int = 0,
        output_key: str | None = None,
        replacement: float = 0.0,
    ) -> dict[str, Any]:
        """Ablate one selected semantic-memory region and measure target-score change."""
        trunk = self._resolve_trunk()
        resolved = _normalize_target_slice(
            x.size(-2),
            x.size(-1),
            target_xy=target_xy,
            target_patch=target_patch,
            center_xy=center_xy,
            patch_radius=patch_radius,
        )

        with torch.no_grad():
            encoder_features = trunk.encode_features(x)
            memories = trunk._progressive_elevator(trunk._build_semantic_memories(encoder_features))
            if memory_scale not in memories:
                raise ValueError(f"memory_scale {memory_scale} is not available. Found {sorted(memories)}.")

            baseline_features, _ = trunk.decode_with_memories(encoder_features, memories)
            if isinstance(self.model, HEAUNet):
                baseline_output: Tensor | Mapping[str, Tensor] = self.model.segmentation_head(baseline_features)
            else:
                baseline_output = self.model.instance_head(baseline_features)

            ablated_memories = dict(memories)
            scale_memory = memories[memory_scale].clone()
            y, x_coord = coarse_coord
            scale_memory[batch_index, :, y, x_coord] = replacement
            ablated_memories[memory_scale] = scale_memory
            ablated_features, _ = trunk.decode_with_memories(encoder_features, ablated_memories)
            if isinstance(self.model, HEAUNet):
                ablated_output: Tensor | Mapping[str, Tensor] = self.model.segmentation_head(ablated_features)
            else:
                ablated_output = self.model.instance_head(ablated_features)

        baseline_tensor = baseline_output if isinstance(baseline_output, Tensor) else baseline_output[output_key or "foreground_logits"]
        ablated_tensor = ablated_output if isinstance(ablated_output, Tensor) else ablated_output[output_key or "foreground_logits"]

        baseline_score = _extract_patch_score(
            baseline_tensor,
            batch_index=batch_index,
            target_slice=resolved.target_slice,
            channel=target_channel,
        )
        ablated_score = _extract_patch_score(
            ablated_tensor,
            batch_index=batch_index,
            target_slice=resolved.target_slice,
            channel=target_channel,
        )
        delta = baseline_score - ablated_score
        return {
            "target_xy": resolved.target_xy,
            "target_patch": resolved.target_slice,
            "memory_scale": memory_scale,
            "coarse_coord": coarse_coord,
            "baseline_score": baseline_score.detach(),
            "ablated_score": ablated_score.detach(),
            "delta": delta.detach(),
            "metadata": {
                "stage": self._resolve_stage(stage),
                "batch_index": batch_index,
                "target_channel": target_channel,
                "output_key": output_key,
                "replacement": replacement,
            },
        }

    def ablate_topk_regions(
        self,
        x: Tensor,
        *,
        memory_scale: int,
        top_k: int = 3,
        explanation: Mapping[str, Any] | None = None,
        target_xy: tuple[int, int] | None = None,
        target_patch: tuple[tuple[int, int], tuple[int, int]] | None = None,
        center_xy: tuple[int, int] | None = None,
        patch_radius: int | None = None,
        stage: int | str = "top",
        batch_index: int = 0,
        target_channel: int = 0,
        output_key: str | None = None,
        replacement: float = 0.0,
        ranking_mode: ExplainMode = "gated_magnitude",
    ) -> dict[str, Any]:
        """Ablate the top-k scored regions for one memory scale."""
        if explanation is None:
            if target_xy is not None:
                explanation = self.explain_pixel(
                    x,
                    target_xy=target_xy,
                    batch_index=batch_index,
                    stage=stage,
                    mode=ranking_mode,
                    target_channel=target_channel,
                    output_key=output_key,
                )
            else:
                explanation = self.explain_patch(
                    x,
                    target_patch=target_patch,
                    center_xy=center_xy,
                    patch_radius=patch_radius,
                    batch_index=batch_index,
                    stage=stage,
                    mode=ranking_mode,
                    target_channel=target_channel,
                    output_key=output_key,
                )

        heatmap = explanation["per_scale_heatmaps"][memory_scale]
        top_k = max(1, min(top_k, heatmap.numel()))
        top_indices = torch.topk(heatmap.reshape(-1), k=top_k).indices
        width = heatmap.size(1)
        regions = [(int(index // width), int(index % width)) for index in top_indices]
        results = [
            self.ablate_region(
                x,
                memory_scale=memory_scale,
                coarse_coord=coord,
                target_xy=target_xy,
                target_patch=target_patch,
                center_xy=center_xy,
                patch_radius=patch_radius,
                stage=stage,
                batch_index=batch_index,
                target_channel=target_channel,
                output_key=output_key,
                replacement=replacement,
            )
            for coord in regions
        ]
        return {
            "memory_scale": memory_scale,
            "regions": regions,
            "results": results,
            "ranking_mode": ranking_mode,
        }

    def visualize_explanation(
        self,
        image: Tensor,
        explanation: Mapping[str, Any],
    ) -> Any:
        """Render a standard region-level HEA explanation figure."""
        return visualize_explanation(
            image,
            explanation,
            target_xy=explanation.get("target_xy"),
        )
