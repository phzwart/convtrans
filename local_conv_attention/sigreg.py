"""SIGReg regularization for dense LeJEPA latents."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SIGRegLoss(nn.Module):
    """Characteristic-function regularizer toward an isotropic Gaussian."""

    def __init__(
        self,
        *,
        num_slices: int = 256,
        num_knots: int = 17,
        t_max: float = 3.0,
    ) -> None:
        super().__init__()
        if num_slices <= 0:
            raise ValueError("num_slices must be positive.")
        if num_knots < 2:
            raise ValueError("num_knots must be at least 2.")
        if t_max <= 0:
            raise ValueError("t_max must be positive.")
        self.num_slices = num_slices
        self.num_knots = num_knots
        self.t_max = t_max

    def forward(self, embeddings: Tensor) -> Tensor:
        """Compute SIGReg on [N, D] embeddings."""
        if embeddings.dim() != 2:
            raise ValueError(f"Expected embeddings with shape [N, D], got {tuple(embeddings.shape)}.")
        num_samples, latent_dim = embeddings.shape
        if num_samples < 2:
            raise ValueError("SIGReg requires at least two samples.")

        directions = torch.randn(
            latent_dim,
            self.num_slices,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        directions = F.normalize(directions, dim=0)

        knots = torch.linspace(
            0.0,
            self.t_max,
            steps=self.num_knots,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        phi = torch.exp(-0.5 * knots.square())
        trap_weights = torch.ones_like(knots)
        trap_weights[0] = 0.5
        trap_weights[-1] = 0.5
        delta_t = knots[1] - knots[0]

        projected = embeddings @ directions
        angles = projected.unsqueeze(-1) * knots.view(1, 1, -1)
        empirical_cos = torch.cos(angles).mean(dim=0)
        empirical_sin = torch.sin(angles).mean(dim=0)
        target_cos = torch.exp(-0.5 * knots.square()).view(1, -1)
        target_sin = torch.zeros_like(target_cos)

        discrepancy = (
            (empirical_cos - target_cos).square()
            + (empirical_sin - target_sin).square()
        )
        discrepancy = discrepancy * phi.view(1, -1)
        return (discrepancy * trap_weights.view(1, -1)).sum(dim=-1).mean() * delta_t
