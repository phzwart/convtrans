"""
Multi-GPU hybrid-encoder dense LeJEPA training (conv stem + local self-attention blocks).

Mirrors ``examples/dense_lejepa_ddp_spawn.py``, but uses ``HybridDenseLeJEPAModel``
(conv + local attention stack) instead of the HEA U-Net backbone.

Needs >=2 CUDA GPUs (NCCL).

CLI::

    cd /path/to/convtrans
    python examples/hybrid_dense_lejepa_ddp_spawn.py --project-root . --epochs 50 --batch-size 8

Artifacts under ``--output-dir`` (default ``examples/hybrid_dense_lejepa_ddp_outputs/<timestamp>/``):
``config.json``, ``architecture.txt``, ``checkpoint_epoch_*.pt``, ``checkpoint_latest.pt``, ``scalars.json``.

Load::

    from examples.hybrid_dense_lejepa_ddp_spawn import load_hybrid_dense_lejepa_from_checkpoint
    model, cfg = load_hybrid_dense_lejepa_from_checkpoint("checkpoint_latest.pt", map_location="cpu")
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
    from local_conv_attention.hybrid_dense_lejepa import HybridDenseLeJEPAModel


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

    from local_conv_attention import DiscSquareDataset, HybridDenseLeJEPAModel
    from local_conv_attention.config import (
        HEAExperimentConfig,
        HEAUNetModelConfig,
        HybridAttentionBlockConfig,
        HybridConvAttentionEncoderConfig,
        ResidualStemConfig,
    )

    def _contiguous_grad_hook(grad: torch.Tensor | None) -> torch.Tensor | None:
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

    from local_conv_attention.config import DenseLatentConfig, DenseLeJEPAObjectiveConfig

    hidden = 64
    model_cfg = HEAUNetModelConfig(
        name="hybrid_dense_lejepa",
        in_channels=1,
        act="gelu",
        hybrid_encoder=HybridConvAttentionEncoderConfig(
            stem=ResidualStemConfig(
                in_channels=1,
                hidden_channels=hidden,
                kernel_size=3,
                stride=1,
                padding=1,
                use_bias=True,
            ),
            block=HybridAttentionBlockConfig(
                channels=hidden,
                num_heads=4,
                window_size=7,
                dilation=1,
                implementation="optimized",
                boundary_pad="zeros",
                hidden_channels=hidden * 4,
            ),
            depth=6,
            output_mode="feature_map",
        ),
        latent=DenseLatentConfig(
            source="encoder_out",
            step_mode="joint",
            latent_dim=32,
            projector_depth=2,
            projector_kernel_size=3,
            normalize_latents=False,
        ),
        lejepa=DenseLeJEPAObjectiveConfig(
            num_views=4,
            sequential_view_forward=True,
            lambda_sigreg=0.05,
        ),
    )
    model_cfg.lejepa.sigreg.enabled = True
    model_cfg.lejepa.sigreg.num_slices = 64
    model_cfg.lejepa.sigreg.num_knots = 17
    model_cfg.lejepa.sigreg.per_view = True
    model_cfg.lejepa.views.mode = "aligned_same_geometry"
    model_cfg.lejepa.views.pre_corrupt_rotation = True
    model_cfg.lejepa.views.pre_corrupt_rotation_deg = (0.0, 360.0)
    model_cfg.lejepa.views.pre_corrupt_rotation_padding = "reflection"
    model_cfg.lejepa.views.corruption.intensity_jitter = True
    model_cfg.lejepa.views.corruption.blur = False
    model_cfg.lejepa.views.corruption.gaussian_noise = True
    model_cfg.lejepa.views.corruption.random_block_mask = True
    model_cfg.lejepa.views.corruption.block_mask_ratio = 0.04
    model_cfg.lejepa.views.corruption.block_mask_num_blocks = 20

    config = HEAExperimentConfig(model=model_cfg)
    config.validate()

    if rank == 0:
        with (out_path / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config.to_dict(), fh, indent=2)

    model = HybridDenseLeJEPAModel(config.model).to(device)
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
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
                "Hybrid dense LeJEPA DDP spawn — training artifacts\n\n"
                "- config.json: experiment dict; rebuild with experiment_config_from_dict\n"
                "- architecture.txt: model module string + parameter count\n"
                "- checkpoint_epoch_####.pt / checkpoint_latest.pt\n"
                "- scalars.json\n\n"
                "Python load:\n"
                "  from examples.hybrid_dense_lejepa_ddp_spawn import "
                "load_hybrid_dense_lejepa_from_checkpoint\n",
                encoding="utf-8",
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def load_hybrid_dense_lejepa_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | None = None,
) -> tuple[HybridDenseLeJEPAModel, HEAExperimentConfig]:
    """Load :class:`HybridDenseLeJEPAModel` + config from a hybrid spawn checkpoint."""
    import torch

    from local_conv_attention import HybridDenseLeJEPAModel, experiment_config_from_dict

    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    if "config_dict" not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} has no 'config_dict'.")
    experiment_cfg = experiment_config_from_dict(ckpt["config_dict"])
    if experiment_cfg.model.name != "hybrid_dense_lejepa":
        raise ValueError(
            f"Expected hybrid_dense_lejepa checkpoint, got model.name={experiment_cfg.model.name!r}."
        )
    model = HybridDenseLeJEPAModel(experiment_cfg.model)
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
    import torch
    import torch.multiprocessing as mp

    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parent.parent
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(
            f"Need at least 2 visible CUDA devices for multi-GPU spawn; got {world_size}."
        )

    if master_port is None:
        master_port = str(29600 + random.randint(0, 99))

    history_json = str(root / "examples" / ".hybrid_dense_lejepa_spawn_history.json")

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root / "examples" / "hybrid_dense_lejepa_ddp_outputs" / stamp
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
    parser = argparse.ArgumentParser(description="Hybrid encoder dense LeJEPA multi-GPU training (spawn).")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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
