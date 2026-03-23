"""Tests for plot_latent_channels."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import torch

from local_conv_attention.visualization import plot_latent_channels


def test_plot_latent_channels_shapes() -> None:
    for shape, kwargs in [
        ((6, 8, 8), {}),
        ((2, 6, 8, 8), {"batch_index": 1}),
        ((2, 3, 6, 8, 8), {"batch_index": 0, "view_index": 2}),
    ]:
        z = torch.randn(*shape)
        fig = plot_latent_channels(z, max_cols=4, **kwargs)
        assert fig is not None
        matplotlib.pyplot.close(fig)


def test_plot_latent_channels_global_norm() -> None:
    z = torch.randn(3, 4, 4)
    fig = plot_latent_channels(z, global_norm=True, per_channel_norm=False)
    matplotlib.pyplot.close(fig)
