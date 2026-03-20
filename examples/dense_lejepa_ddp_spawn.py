"""
Multi-GPU dense LeJEPA training using ``torch.multiprocessing.spawn``.

Use this from a **Jupyter notebook** when you want DDP without ``torchrun``: spawn
starts one Python process per GPU. Workers are not IPython, so they do not hit the
notebook kernel's single-process / stale ``WORLD_SIZE`` logic.

Example from a notebook (repo root on ``sys.path``)::

    from pathlib import Path
    import sys
    ROOT = Path("/global/cfs/cdirs/m4880/convtrans")
    sys.path.insert(0, str(ROOT))
    from examples.dense_lejepa_ddp_spawn import run_spawn_training

    history = run_spawn_training(ROOT, epochs=50)

CLI (**needs ≥2 CUDA GPUs**; NCCL). Training uses **rotate** by default: each batch step picks one latent hook via ``rotate_latent_index=global_step`` (cycles all hooks from ``default_all_latent_hooks``). For **joint** (every hook every batch), set ``step_mode="joint"`` in ``_worker`` and you can drop the ``rotate_latent_index`` argument.

::

    cd /path/to/convtrans
    python examples/dense_lejepa_ddp_spawn.py --project-root . --epochs 50 --batch-size 8
    # optional: --output-dir ./my_run

**Outputs** (under ``--output-dir``, default ``examples/dense_lejepa_ddp_outputs/<timestamp>/``):

- ``config.json`` — full experiment config for rebuilding the model.
- ``architecture.txt`` — ``str(DenseLeJEPAModel)`` for human inspection.
- ``checkpoint_epoch_%04d.pt`` — every epoch: weights, optimizer, history, embedded ``config_dict``.
- ``checkpoint_latest.pt`` — copy of the most recent epoch (stable path for notebooks).
- ``scalars.json`` — same data as the legacy ``examples/.dense_lejepa_spawn_history.json`` path.

Load for inference / feature extraction on new images::

    from examples.dense_lejepa_ddp_spawn import load_dense_lejepa_from_checkpoint

    model, experiment_cfg = load_dense_lejepa_from_checkpoint(
        "path/to/checkpoint_latest.pt", map_location="cpu"
    )
    model.eval()
    with torch.no_grad():
        out = model(images_tensor)  # keys: latents, latents_by_source, loss, ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from local_conv_attention.config import HEAExperimentConfig
    from local_conv_attention.dense_lejepa import DenseLeJEPAModel


def _worker(
    rank: int,
    world_size: int,
    project_root: str,
    epochs: int,
    batch_size: int,
    master_addr: str,
    master_port: str,
    history_json: str,
    output_dir: str,
) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    root = Path(project_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from local_conv_attention import (
        DenseLeJEPAModel,
        DiscSquareDataset,
        default_decoder_latent_hooks,
        load_experiment_config,
    )

    def _contiguous_grad_hook(grad: torch.Tensor | None) -> torch.Tensor | None:
        """Checkpoint + conv can produce oddly-strided grads; DDP reducer warns unless contiguous."""
        if grad is None:
            return None
        return grad.contiguous()

    out_path = Path(output_dir)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # device_id (PyTorch 2.4+): ties the process group to this GPU and silences
    # NCCL “guess device from rank” / barrier context warnings on Perlmutter-style setups.
    try:
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl", init_method="env://")

    if rank == 0:
        out_path.mkdir(parents=True, exist_ok=True)
    dist.barrier()


    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    train_dataset = DiscSquareDataset(repeats_per_type=256)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    config = load_experiment_config(root / "configs" / "hea_dense_lejepa_default.yaml")
    config.model.in_channels = 1
    config.model.base_channels = 16
    config.model.channel_multipliers = [1, 2, 4, 8]
    config.model.encoder_depths = [1, 1, 1, 1]
    config.model.decoder_depths = [2, 2, 2]
    config.model.attention.heads = 4
    config.model.attention.head_dim = 16
    config.model.attention.operator_backend = "optimized"

    # Concat skips: decoder receives encoder features (not just upsample shape).
    config.model.use_raw_skips = True

    # Decoder-side hooks only: bottleneck, decoder_*, top — these carry HEA / cross-scale context.
    config.model.latent.sources = default_decoder_latent_hooks(len(config.model.channel_multipliers))
    config.model.latent.step_mode = "rotate"
    config.model.latent.latent_dim = 32
    config.model.latent.projector_depth = 2
    config.model.latent.projector_kernel_size = 3
    config.model.latent.normalize_latents = False

    # HEA fusion on all decoder stages (full cross-scale context in every decoder feature).
    config.model.hea.enabled_decoder_stages = [0, 1, 2]
    config.model.semantic_memory.window_sizes = [7, 7, 7]
    config.model.hea.per_scale_window_sizes = [7, 7, 7]
    config.model.hea.per_scale_dilations = [1, 1, 1]

    config.model.lejepa.num_views = 4
    # Lower peak VRAM vs fusing all views through the backbone in one batch.
    config.model.lejepa.sequential_view_forward = True
    config.model.lejepa.lambda_sigreg = 0.05
    config.model.lejepa.sigreg.enabled = True
    config.model.lejepa.sigreg.num_slices = 64
    config.model.lejepa.sigreg.num_knots = 17
    config.model.lejepa.sigreg.per_view = True

    config.model.lejepa.views.mode = "aligned_same_geometry"
    # Rotate → corrupt → derotate: same pixel grid as input (dense tokens aligned); ``reflection`` softens borders.
    config.model.lejepa.views.pre_corrupt_rotation = True
    config.model.lejepa.views.pre_corrupt_rotation_deg = (0.0, 360.0)
    # For θ ∈ {0,90,180,270} only (less grid_sample blur): set quarter_turns True (deg unused for sampling).
    # config.model.lejepa.views.pre_corrupt_rotation_quarter_turns = True
    config.model.lejepa.views.pre_corrupt_rotation_padding = "reflection"
    config.model.lejepa.views.corruption.intensity_jitter = True
    config.model.lejepa.views.corruption.blur = False
    config.model.lejepa.views.corruption.gaussian_noise = True
    config.model.lejepa.views.corruption.random_block_mask = True
    # Small blocks (ratio×side ~ few px at 128²); many of them → scattered holes, not huge occlusions.
    config.model.lejepa.views.corruption.block_mask_ratio = 0.04
    config.model.lejepa.views.corruption.block_mask_num_blocks = 20

    # Trades ~2× backbone forward compute for much lower activation memory (OOM mitigation).
    config.model.backbone_gradient_checkpointing = True

    config.validate()

    if rank == 0:
        # Human-readable full config for rebuilding ``DenseLeJEPAModel(config.model)``.
        with (out_path / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config.to_dict(), fh, indent=2)

    model = DenseLeJEPAModel(config.model).to(device)
    # With full ``latent.sources`` all backbone submodules get loss signal; keep True if you
    # use a subset of hooks (e.g. only encoder_0).
    # gradient_as_bucket_view=False: with backbone gradient checkpointing, some conv grads
    # are non-contiguous; the default True bucket views then warn and can slow allreduce.
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
        gradient_as_bucket_view=False,
    )
    for _p in model.parameters():
        if _p.requires_grad:
            _p.register_hook(_contiguous_grad_hook)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if rank == 0:
        with (out_path / "architecture.txt").open("w", encoding="utf-8") as fh:
            fh.write(str(model.module))
            fh.write("\n\n--- parameters ---\n")
            num_params = sum(p.numel() for p in model.module.parameters())
            fh.write(f"total_numel: {num_params:,}\n")

    history: dict[str, list[float]] = {"loss": [], "inv_loss": [], "sigreg_loss": []}
    global_step = 0
    try:
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            running = {key: 0.0 for key in history}
            batches = 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out = model(images, rotate_latent_index=global_step)
                global_step += 1
                out["loss"].backward()
                optimizer.step()
                for key in history:
                    running[key] += float(out[key].detach())
                batches += 1

            stats = torch.tensor(
                [running["loss"], running["inv_loss"], running["sigreg_loss"], float(batches)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_batches = max(int(stats[3].item()), 1)
            for i, key in enumerate(list(history.keys())):
                history[key].append(float(stats[i].item() / total_batches))

            if rank == 0:
                print(
                    f"epoch {epoch + 1:02d} | loss={history['loss'][-1]:.4f} | "
                    f"inv={history['inv_loss'][-1]:.4f} | sigreg={history['sigreg_loss'][-1]:.4f}"
                )

            if rank == 0:
                epoch_tag = epoch + 1
                ckpt = {
                    "epoch": epoch_tag,
                    "global_step": global_step,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": {k: list(v) for k, v in history.items()},
                    "config_dict": config.to_dict(),
                }
                epoch_file = out_path / f"checkpoint_epoch_{epoch_tag:04d}.pt"
                torch.save(ckpt, epoch_file)
                shutil.copy2(epoch_file, out_path / "checkpoint_latest.pt")

        if rank == 0:
            Path(history_json).write_text(json.dumps(history))
            with (out_path / "scalars.json").open("w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)
            readme = out_path / "README_OUTPUT.txt"
            readme.write_text(
                "Dense LeJEPA DDP spawn — training artifacts\n\n"
                "- config.json: experiment dict; rebuild with "
                "local_conv_attention.experiment_config_from_dict(...)\n"
                "- architecture.txt: model module string + parameter count\n"
                "- checkpoint_epoch_####.pt: one per epoch (weights, optimizer, history, config_dict)\n"
                "- checkpoint_latest.pt: copy of last epoch\n"
                "- scalars.json: loss / inv_loss / sigreg per epoch\n\n"
                "Python load:\n"
                "  from examples.dense_lejepa_ddp_spawn import load_dense_lejepa_from_checkpoint\n"
                "  model, cfg = load_dense_lejepa_from_checkpoint('checkpoint_latest.pt')\n",
                encoding="utf-8",
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def load_dense_lejepa_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | None = None,
) -> tuple[DenseLeJEPAModel, HEAExperimentConfig]:
    """Load ``DenseLeJEPAModel`` + ``HEAExperimentConfig`` from a spawn training checkpoint.

    Checkpoints embed ``config_dict``; you can also train with the standalone ``config.json``
    in the same output folder if you prefer.
    """
    import torch

    from local_conv_attention import DenseLeJEPAModel, experiment_config_from_dict

    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    if "config_dict" not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} has no 'config_dict'; use a training checkpoint.")
    experiment_cfg = experiment_config_from_dict(ckpt["config_dict"])
    model = DenseLeJEPAModel(experiment_cfg.model)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, experiment_cfg


def run_spawn_training(
    project_root: str | Path | None = None,
    *,
    epochs: int = 50,
    batch_size: int = 8,
    master_addr: str = "127.0.0.1",
    master_port: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Spawn one process per visible GPU; return training history (rank 0 metrics only).

    Artifacts are written under ``output_dir`` (default timestamped folder under
    ``<repo>/examples/dense_lejepa_ddp_outputs/``).
    """
    import torch
    import torch.multiprocessing as mp

    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parent.parent
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(
            f"Need at least 2 visible CUDA devices for multi-GPU spawn; got {world_size}. "
            "For one GPU, use the normal notebook cells without spawn."
        )

    if master_port is None:
        master_port = str(29500 + random.randint(0, 99))

    history_json = str(root / "examples" / ".dense_lejepa_spawn_history.json")

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root / "examples" / "dense_lejepa_ddp_outputs" / stamp
    output_dir = Path(output_dir).resolve()

    mp.spawn(
        _worker,
        args=(
            world_size,
            str(root),
            epochs,
            batch_size,
            master_addr,
            master_port,
            history_json,
            str(output_dir),
        ),
        nprocs=world_size,
        join=True,
    )

    p = Path(history_json)
    if p.is_file():
        return json.loads(p.read_text())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense LeJEPA multi-GPU training (spawn).")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repo root (default: parent of examples/)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints, config.json, architecture.txt "
        "(default: examples/dense_lejepa_ddp_outputs/<timestamp>/)",
    )
    args = parser.parse_args()

    run_spawn_training(
        args.project_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
