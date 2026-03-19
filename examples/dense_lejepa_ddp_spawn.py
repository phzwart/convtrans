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

CLI::

    python examples/dense_lejepa_ddp_spawn.py --project-root /path/to/convtrans
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


def _worker(
    rank: int,
    world_size: int,
    project_root: str,
    epochs: int,
    batch_size: int,
    master_addr: str,
    master_port: str,
    history_json: str,
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

    from local_conv_attention import DenseLeJEPAModel, DiscSquareDataset, load_experiment_config

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{rank}")

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
    config.model.decoder_depths = [1, 1, 1]
    config.model.attention.heads = 4
    config.model.attention.head_dim = 16
    config.model.attention.operator_backend = "optimized"

    config.model.latent.source = "encoder_0"
    config.model.latent.latent_dim = 32
    config.model.latent.projector_depth = 1
    config.model.latent.normalize_latents = False

    config.model.hea.enabled_decoder_stages = [0]
    config.model.semantic_memory.window_sizes = [7, 7, 7]
    config.model.hea.per_scale_window_sizes = [7, 7, 7]

    config.model.lejepa.num_views = 4
    config.model.lejepa.lambda_sigreg = 0.05
    config.model.lejepa.sigreg.enabled = True
    config.model.lejepa.sigreg.num_slices = 64
    config.model.lejepa.sigreg.num_knots = 17
    config.model.lejepa.sigreg.per_view = True

    config.model.lejepa.views.mode = "aligned_same_geometry"
    config.model.lejepa.views.corruption.intensity_jitter = True
    config.model.lejepa.views.corruption.blur = False
    config.model.lejepa.views.corruption.gaussian_noise = True
    config.model.lejepa.views.corruption.random_block_mask = True
    config.model.lejepa.views.corruption.block_mask_ratio = 0.04
    config.model.lejepa.views.corruption.block_mask_num_blocks = 4

    config.validate()

    model = DenseLeJEPAModel(config.model).to(device)
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history: dict[str, list[float]] = {"loss": [], "inv_loss": [], "sigreg_loss": []}
    try:
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            running = {key: 0.0 for key in history}
            batches = 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out = model(images)
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
            Path(history_json).write_text(json.dumps(history))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_spawn_training(
    project_root: str | Path | None = None,
    *,
    epochs: int = 50,
    batch_size: int = 16,
    master_addr: str = "127.0.0.1",
    master_port: str | None = None,
) -> dict[str, Any] | None:
    """Spawn one process per visible GPU; return training history (rank 0 metrics only)."""
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default=None)
    args = parser.parse_args()

    run_spawn_training(
        args.project_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )


if __name__ == "__main__":
    main()
