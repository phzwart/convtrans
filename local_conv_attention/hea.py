"""Hierarchical Elevator Attention modules for segmentation models."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import AttentionImplementation
from .block import LocalTransformerBlock2d
from .encoder import ActKind, NormKind, make_activation, make_norm2d
from .ops import BoundaryPadMode, ConvShiftBank2d, NeighborhoodShift2d
from .utils import merge_heads, reshape_heads, scaled_dot_product_scale


CrossScaleFusionMode = Literal["per_scale", "joint_softmax"]
ResidualFusionMode = Literal["gated_residual", "additive", "concat_proj"]


def _align_to_query_grid(tensor: Tensor, out_h: int, out_w: int, scale_factor: int) -> Tensor:
    """Repeat coarse-lattice candidates onto the fine query grid via anchor indexing."""
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}.")
    src_h = tensor.size(-2)
    src_w = tensor.size(-1)
    y_idx = torch.div(
        torch.arange(out_h, device=tensor.device),
        scale_factor,
        rounding_mode="floor",
    ).clamp(max=src_h - 1)
    x_idx = torch.div(
        torch.arange(out_w, device=tensor.device),
        scale_factor,
        rounding_mode="floor",
    ).clamp(max=src_w - 1)
    return tensor.index_select(-2, y_idx).index_select(-1, x_idx)


def _normalize_target_slice(
    height: int,
    width: int,
    target_slice: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Clamp a requested spatial slice to valid image bounds."""
    if target_slice is None:
        return ((0, height), (0, width))
    (y0, y1), (x0, x1) = target_slice
    y0 = max(0, min(height, y0))
    y1 = max(y0, min(height, y1))
    x0 = max(0, min(width, x0))
    x1 = max(x0, min(width, x1))
    return ((y0, y1), (x0, x1))


def _candidate_coordinate_metadata(
    *,
    target_slice: tuple[tuple[int, int], tuple[int, int]],
    query_height: int,
    query_width: int,
    scale_factor: int,
    memory_height: int,
    memory_width: int,
    offsets: Sequence[tuple[int, int]],
    device: torch.device,
) -> dict[str, Tensor]:
    """Return anchor and candidate coarse coordinates for a selected fine-grid patch."""
    (y0, y1), (x0, x1) = target_slice
    patch_h = y1 - y0
    patch_w = x1 - x0

    y_idx = torch.div(
        torch.arange(query_height, device=device),
        scale_factor,
        rounding_mode="floor",
    ).clamp(max=memory_height - 1)
    x_idx = torch.div(
        torch.arange(query_width, device=device),
        scale_factor,
        rounding_mode="floor",
    ).clamp(max=memory_width - 1)

    anchor_y = y_idx[y0:y1].view(patch_h, 1).expand(patch_h, patch_w)
    anchor_x = x_idx[x0:x1].view(1, patch_w).expand(patch_h, patch_w)

    dy = torch.tensor([offset[0] for offset in offsets], device=device, dtype=torch.long)
    dx = torch.tensor([offset[1] for offset in offsets], device=device, dtype=torch.long)
    candidate_y = anchor_y.unsqueeze(-1) + dy.view(1, 1, -1)
    candidate_x = anchor_x.unsqueeze(-1) + dx.view(1, 1, -1)
    candidate_valid = (
        (candidate_y >= 0)
        & (candidate_y < memory_height)
        & (candidate_x >= 0)
        & (candidate_x < memory_width)
    )

    return {
        "anchor_coords": torch.stack([anchor_y, anchor_x], dim=-1),
        "candidate_coords": torch.stack([candidate_y, candidate_x], dim=-1),
        "candidate_valid": candidate_valid,
    }


class _CrossScaleNeighborhoodExtractor(nn.Module):
    """Extract coarse-lattice K/V neighborhoods and align them to a fine query grid."""

    def __init__(
        self,
        num_heads: int,
        window_size: int,
        dilation: int,
        *,
        implementation: AttentionImplementation,
        boundary_pad: BoundaryPadMode = "zeros",
    ) -> None:
        super().__init__()
        if implementation not in {"optimized", "shift"}:
            raise ValueError(
                "Cross-scale HEA only supports 'optimized' and 'shift' backends."
            )
        self.num_heads = num_heads
        self.implementation = implementation
        if implementation == "optimized":
            self.shift = ConvShiftBank2d(
                window_size=window_size,
                dilation=dilation,
                boundary_pad=boundary_pad,
            )
        else:
            self.shift = NeighborhoodShift2d(
                window_size=window_size,
                dilation=dilation,
                boundary_pad=boundary_pad,
            )

    def forward(
        self,
        k: Tensor,
        v: Tensor,
        *,
        query_height: int,
        query_width: int,
        scale_factor: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, channels, _, _ = k.shape
        head_dim = channels // self.num_heads

        if self.implementation == "optimized":
            kv_neighbors, valid_mask = self.shift(
                torch.cat([k.contiguous(), v.contiguous()], dim=0),
                return_mask=True,
            )
            k_neighbors, v_neighbors = kv_neighbors.split(batch, dim=0)
            k_neighbors = k_neighbors.contiguous().reshape(
                batch,
                self.num_heads,
                head_dim,
                -1,
                k.size(-2),
                k.size(-1),
            )
            v_neighbors = v_neighbors.contiguous().reshape(
                batch,
                self.num_heads,
                head_dim,
                -1,
                v.size(-2),
                v.size(-1),
            )
            k_neighbors = k_neighbors.permute(0, 1, 3, 2, 4, 5).contiguous()
            v_neighbors = v_neighbors.permute(0, 1, 3, 2, 4, 5).contiguous()
        else:
            k_neighbors, valid_mask = self.shift(reshape_heads(k, self.num_heads), return_mask=True)
            v_neighbors = self.shift(reshape_heads(v, self.num_heads))
            valid_mask = valid_mask.squeeze(3)

        k_neighbors = _align_to_query_grid(k_neighbors, query_height, query_width, scale_factor)
        v_neighbors = _align_to_query_grid(v_neighbors, query_height, query_width, scale_factor)
        valid_mask = _align_to_query_grid(valid_mask, query_height, query_width, scale_factor)
        return k_neighbors, v_neighbors, valid_mask


class HierarchicalElevatorAttention2d(nn.Module):
    """Cross-scale local attention from coarse semantic memories into a fine query map."""

    def __init__(
        self,
        query_dim: int,
        memory_dims: Sequence[int],
        scale_factors: Sequence[int],
        *,
        num_heads: int,
        head_dim: int,
        window_sizes: Sequence[int],
        dilations: Sequence[int],
        implementation: AttentionImplementation = "optimized",
        boundary_pad: BoundaryPadMode = "zeros",
        fusion_mode: CrossScaleFusionMode = "per_scale",
        qkv_bias: bool = True,
        joint_scale_projection: bool = True,
    ) -> None:
        super().__init__()
        if not memory_dims:
            raise ValueError("HierarchicalElevatorAttention2d requires at least one memory map.")
        expected = len(memory_dims)
        for name, values in {
            "scale_factors": scale_factors,
            "window_sizes": window_sizes,
            "dilations": dilations,
        }.items():
            if len(values) != expected:
                raise ValueError(f"{name} must have length {expected}, got {len(values)}.")

        self.query_dim = query_dim
        self.memory_dims = list(memory_dims)
        self.scale_factors = list(scale_factors)
        self.fusion_mode = fusion_mode
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.joint_scale_projection = joint_scale_projection

        self.query_proj = nn.Conv2d(query_dim, self.attn_dim, kernel_size=1, bias=qkv_bias)
        self.key_projs = nn.ModuleList(
            [nn.Conv2d(memory_dim, self.attn_dim, kernel_size=1, bias=qkv_bias) for memory_dim in memory_dims]
        )
        self.value_projs = nn.ModuleList(
            [nn.Conv2d(memory_dim, self.attn_dim, kernel_size=1, bias=qkv_bias) for memory_dim in memory_dims]
        )
        self.extractors = nn.ModuleList(
            [
                _CrossScaleNeighborhoodExtractor(
                    num_heads=num_heads,
                    window_size=window_size,
                    dilation=dilation,
                    implementation=implementation,
                    boundary_pad=boundary_pad,
                )
                for window_size, dilation in zip(window_sizes, dilations)
            ]
        )
        if fusion_mode == "per_scale" and len(memory_dims) > 1 and joint_scale_projection:
            self.scale_fuse = nn.Conv2d(
                len(memory_dims) * self.attn_dim,
                self.attn_dim,
                kernel_size=1,
            )
        else:
            self.scale_fuse = nn.Identity()

    @staticmethod
    def _debug_slice(
        tensor: Tensor,
        target_slice: tuple[tuple[int, int], tuple[int, int]],
    ) -> Tensor:
        (y0, y1), (x0, x1) = target_slice
        return tensor[..., y0:y1, x0:x1]

    def _collect_scale_debug(
        self,
        *,
        attention: Tensor,
        v_neighbors: Tensor,
        valid_mask: Tensor,
        target_slice: tuple[tuple[int, int], tuple[int, int]],
        batch_index: int,
        scale_factor: int,
        offsets: Sequence[tuple[int, int]],
        memory_shape: tuple[int, int],
        query_height: int,
        query_width: int,
    ) -> dict[str, Any]:
        attention_slice = self._debug_slice(attention[batch_index], target_slice)
        weighted_values = attention_slice.unsqueeze(2) * self._debug_slice(
            v_neighbors[batch_index],
            target_slice,
        )
        mask_source = valid_mask[0] if valid_mask.size(0) == 1 else valid_mask[batch_index]
        valid_mask_slice = self._debug_slice(mask_source.squeeze(0), target_slice)
        coord_meta = _candidate_coordinate_metadata(
            target_slice=target_slice,
            query_height=query_height,
            query_width=query_width,
            scale_factor=scale_factor,
            memory_height=memory_shape[0],
            memory_width=memory_shape[1],
            offsets=offsets,
            device=attention.device,
        )
        return {
            "attention": attention_slice.detach(),
            "weighted_values": weighted_values.detach(),
            "valid_mask": valid_mask_slice.detach(),
            "anchor_coords": coord_meta["anchor_coords"].detach(),
            "candidate_coords": coord_meta["candidate_coords"].detach(),
            "candidate_valid": coord_meta["candidate_valid"].detach(),
            "memory_shape": memory_shape,
            "scale_factor": scale_factor,
            "offsets": list(offsets),
        }

    def _score_candidates(self, q_heads: Tensor, k_neighbors: Tensor, valid_mask: Tensor) -> Tensor:
        scores = (q_heads.unsqueeze(2) * k_neighbors).sum(dim=3)
        scores = scores * scaled_dot_product_scale(self.head_dim)
        return scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)

    def forward(
        self,
        query: Tensor,
        memories: Sequence[Tensor],
        *,
        return_debug: bool = False,
        target_slice: tuple[tuple[int, int], tuple[int, int]] | None = None,
        batch_index: int = 0,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        if len(memories) != len(self.memory_dims):
            raise ValueError(
                f"Expected {len(self.memory_dims)} memories, received {len(memories)}."
            )
        if batch_index < 0 or batch_index >= query.size(0):
            raise ValueError(f"batch_index {batch_index} is out of range for batch size {query.size(0)}.")

        query_height, query_width = query.shape[-2:]
        normalized_slice = _normalize_target_slice(query_height, query_width, target_slice)
        q_heads = reshape_heads(self.query_proj(query), self.num_heads)

        if self.fusion_mode == "joint_softmax":
            logits_all = []
            values_all = []
            scale_debug_sources: list[dict[str, Any]] = []
            for memory, key_proj, value_proj, extractor, scale_factor in zip(
                memories,
                self.key_projs,
                self.value_projs,
                self.extractors,
                self.scale_factors,
            ):
                k_neighbors, v_neighbors, valid_mask = extractor(
                    key_proj(memory),
                    value_proj(memory),
                    query_height=query_height,
                    query_width=query_width,
                    scale_factor=scale_factor,
                )
                logits_all.append(self._score_candidates(q_heads, k_neighbors, valid_mask))
                values_all.append(v_neighbors)
                if return_debug:
                    scale_debug_sources.append(
                        {
                            "v_neighbors": v_neighbors,
                            "valid_mask": valid_mask,
                            "scale_factor": scale_factor,
                            "offsets": extractor.shift.offsets,
                            "memory_shape": memory.shape[-2:],
                        }
                    )

            logits = torch.cat(logits_all, dim=2)
            attention = torch.softmax(logits, dim=2)
            values = torch.cat(values_all, dim=2)
            context = merge_heads((attention.unsqueeze(3) * values).sum(dim=2))
            if not return_debug:
                return context

            per_scale_debug: list[dict[str, Any]] = []
            offset_index = 0
            for scale_source, scale_values in zip(scale_debug_sources, values_all):
                num_offsets = scale_values.size(2)
                scale_attention = attention[:, :, offset_index : offset_index + num_offsets]
                per_scale_debug.append(
                    self._collect_scale_debug(
                        attention=scale_attention,
                        v_neighbors=scale_source["v_neighbors"],
                        valid_mask=scale_source["valid_mask"],
                        target_slice=normalized_slice,
                        batch_index=batch_index,
                        scale_factor=scale_source["scale_factor"],
                        offsets=scale_source["offsets"],
                        memory_shape=scale_source["memory_shape"],
                        query_height=query_height,
                        query_width=query_width,
                    )
                )
                offset_index += num_offsets
            return context, {
                "target_slice": normalized_slice,
                "fusion_mode": self.fusion_mode,
                "per_scale": per_scale_debug,
                "query_shape": tuple(query.shape),
            }

        contexts = []
        per_scale_debug = []
        for memory, key_proj, value_proj, extractor, scale_factor in zip(
            memories,
            self.key_projs,
            self.value_projs,
            self.extractors,
            self.scale_factors,
        ):
            k_neighbors, v_neighbors, valid_mask = extractor(
                key_proj(memory),
                value_proj(memory),
                query_height=query_height,
                query_width=query_width,
                scale_factor=scale_factor,
            )
            logits = self._score_candidates(q_heads, k_neighbors, valid_mask)
            attention = torch.softmax(logits, dim=2)
            contexts.append(merge_heads((attention.unsqueeze(3) * v_neighbors).sum(dim=2)))
            if return_debug:
                per_scale_debug.append(
                    self._collect_scale_debug(
                        attention=attention,
                        v_neighbors=v_neighbors,
                        valid_mask=valid_mask,
                        target_slice=normalized_slice,
                        batch_index=batch_index,
                        scale_factor=scale_factor,
                        offsets=extractor.shift.offsets,
                        memory_shape=memory.shape[-2:],
                        query_height=query_height,
                        query_width=query_width,
                    )
                )

        if len(contexts) == 1:
            context = contexts[0]
        elif self.joint_scale_projection:
            context = self.scale_fuse(torch.cat(contexts, dim=1))
        else:
            context = torch.stack(contexts, dim=0).mean(dim=0)
        if not return_debug:
            return context
        return context, {
            "target_slice": normalized_slice,
            "fusion_mode": self.fusion_mode,
            "per_scale": per_scale_debug,
            "query_shape": tuple(query.shape),
        }


class SemanticMemoryBlock2d(nn.Module):
    """Build semantically enriched low-resolution memories with local transformer blocks."""

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        num_heads: int,
        window_size: int,
        dilation: int,
        implementation: AttentionImplementation = "optimized",
        boundary_pad: BoundaryPadMode = "zeros",
        use_local_transformer_block: bool = True,
        norm: NormKind = "batchnorm",
        act: ActKind = "gelu",
    ) -> None:
        super().__init__()
        blocks = []
        for _ in range(depth):
            if use_local_transformer_block:
                blocks.append(
                    LocalTransformerBlock2d(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        dilation=dilation,
                        implementation=implementation,
                        boundary_pad=boundary_pad,
                    )
                )
            else:
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                        make_norm2d(norm, dim),
                        make_activation(act),
                    )
                )
        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class HEAFusionBlock2d(nn.Module):
    """Inject hierarchical semantic context into a query map with configurable fusion."""

    def __init__(
        self,
        query_dim: int,
        memory_dims: Sequence[int],
        scale_factors: Sequence[int],
        *,
        num_heads: int,
        head_dim: int,
        window_sizes: Sequence[int],
        dilations: Sequence[int],
        implementation: AttentionImplementation = "optimized",
        boundary_pad: BoundaryPadMode = "zeros",
        fusion_mode: CrossScaleFusionMode = "per_scale",
        residual_fusion: ResidualFusionMode = "gated_residual",
        qkv_bias: bool = True,
        project_context: bool = True,
        joint_scale_projection: bool = True,
    ) -> None:
        super().__init__()
        self.residual_fusion = residual_fusion
        self.context = HierarchicalElevatorAttention2d(
            query_dim=query_dim,
            memory_dims=memory_dims,
            scale_factors=scale_factors,
            num_heads=num_heads,
            head_dim=head_dim,
            window_sizes=window_sizes,
            dilations=dilations,
            implementation=implementation,
            boundary_pad=boundary_pad,
            fusion_mode=fusion_mode,
            qkv_bias=qkv_bias,
            joint_scale_projection=joint_scale_projection,
        )
        attn_dim = num_heads * head_dim
        if project_context or attn_dim != query_dim:
            self.context_proj = nn.Conv2d(attn_dim, query_dim, kernel_size=1)
        else:
            self.context_proj = nn.Identity()

        if residual_fusion == "gated_residual":
            self.gate = nn.Conv2d(query_dim * 2, query_dim, kernel_size=1)
        elif residual_fusion == "concat_proj":
            self.concat_proj = nn.Conv2d(query_dim * 2, query_dim, kernel_size=1)

    def _project_candidate_contributions(self, weighted_values: Tensor) -> Tensor:
        """Project per-candidate attention-weighted values into query channels."""
        num_heads, num_offsets, head_dim, patch_h, patch_w = weighted_values.shape
        merged = (
            weighted_values.permute(1, 0, 2, 3, 4)
            .contiguous()
            .reshape(num_offsets, num_heads * head_dim, patch_h, patch_w)
        )
        return self.context_proj(merged)

    def _fuse_candidate_contributions(
        self,
        projected_candidates: Tensor,
        gate_slice: Tensor | None,
        query_dim: int,
    ) -> Tensor:
        if self.residual_fusion == "additive":
            return projected_candidates
        if self.residual_fusion == "concat_proj":
            weight = self.concat_proj.weight[:, query_dim:, :, :]
            return F.conv2d(projected_candidates, weight, bias=None)
        if gate_slice is None:
            raise RuntimeError("gated_residual fusion requires a gate tensor.")
        return projected_candidates * gate_slice.unsqueeze(0)

    def forward(
        self,
        query: Tensor,
        memories: Sequence[Tensor],
        *,
        return_debug: bool = False,
        target_slice: tuple[tuple[int, int], tuple[int, int]] | None = None,
        batch_index: int = 0,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        context_result = self.context(
            query,
            memories,
            return_debug=return_debug,
            target_slice=target_slice,
            batch_index=batch_index,
        )
        if return_debug:
            context, context_debug = context_result
        else:
            context = context_result
        projected = self.context_proj(context)

        if self.residual_fusion == "additive":
            output = query + projected
            gate = None
        elif self.residual_fusion == "concat_proj":
            output = self.concat_proj(torch.cat([query, projected], dim=1))
            gate = None
        else:
            gate = torch.sigmoid(self.gate(torch.cat([query, projected], dim=1)))
            output = query + gate * projected

        if not return_debug:
            return output

        query_dim = query.size(1)
        normalized_slice = context_debug["target_slice"]
        (y0, y1), (x0, x1) = normalized_slice
        query_slice = query[batch_index : batch_index + 1, :, y0:y1, x0:x1].detach()
        context_slice = context[batch_index : batch_index + 1, :, y0:y1, x0:x1].detach()
        projected_slice = projected[batch_index : batch_index + 1, :, y0:y1, x0:x1].detach()
        gate_slice = None
        if gate is not None:
            gate_slice = gate[batch_index, :, y0:y1, x0:x1].detach()

        per_scale_debug = []
        for scale_debug in context_debug["per_scale"]:
            projected_candidates = self._project_candidate_contributions(scale_debug["weighted_values"])
            fused_candidates = self._fuse_candidate_contributions(
                projected_candidates,
                gate_slice,
                query_dim,
            )
            per_scale_debug.append(
                {
                    **scale_debug,
                    "projected_candidates": projected_candidates.detach(),
                    "fused_candidates": fused_candidates.detach(),
                }
            )

        return output, {
            "target_slice": normalized_slice,
            "fusion_mode": self.context.fusion_mode,
            "residual_fusion": self.residual_fusion,
            "per_scale": per_scale_debug,
            "gate": gate_slice,
            "query_slice": query_slice,
            "context_slice": context_slice,
            "projected_context": projected_slice,
        }
