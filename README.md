# Local Convolutional Attention

`local-conv-attention` is a small PyTorch toolkit for exact 2D local self-attention over NCHW feature maps. It implements local attention as a fixed convolutional-style scaffold: the spatial operator is a non-learned local neighborhood extractor over an odd `M x M` window, while the mixing coefficients remain data-dependent through `Q · K`.

## Mathematical target

For `Q, K, V in R^{B x C x H x W}` and offsets `Δ = (Δy, Δx)` in an odd-sized local window `W`,

```text
score_Δ(i,j) = <Q(i,j), K(i+Δy, j+Δx)> / sqrt(d_head)
alpha_Δ(i,j) = softmax_Δ(score_Δ(i,j))
Y(i,j) = sum_Δ alpha_Δ(i,j) * V(i+Δy, j+Δx)
```

Invalid border positions are masked out before the softmax. Exact equivalence here is to transformer attention restricted to the same local window, not to unrestricted global attention.

## Why this is not a static convolution

This package does **not** learn an `M x M` spatial kernel. Instead, it keeps the spatial structure fixed:

- the same neighborhood geometry is used at every pixel
- the operator graph is fixed: shift or extract neighbors, compute `Q · K`, mask invalid positions, softmax over offsets, and aggregate `V`
- the attention coefficients are input-dependent, so the local mixing changes from one spatial location and one input to another

That is why this is best viewed as a fixed convolutional scaffold with data-dependent mixing, rather than a static convolution.

## Package layout

```text
local_conv_attention/
  __init__.py
  attention.py     # optimized and reference attention backends plus 1x1 QKV wrappers
  backbone.py      # reusable HEA trunk without task heads
  block.py         # transformer-style blocks
  dense_lejepa.py  # dense teacher-free LeJEPA adaptor
  explain.py       # region-level HEA explanation API
  masks.py         # border-validity masks and flattened local masks
  ops.py           # explicit and conv2d-backed fixed shift banks
  reference.py     # unfold and flattened masked-attention references
  sigreg.py        # SIGReg characteristic-function regularizer
  utils.py         # shape helpers, LayerNorm wrapper, MLP
  visualization.py # HEA explanation heatmap and overlay helpers
  views.py         # aligned multi-view SSL generation
tests/
benchmarks/
examples/
```

## Implementations

### Optimized default path

`LocalAttention2d` is the default optimized backend. It uses `ConvShiftBank2d`, a grouped `conv2d` with fixed one-hot kernels, to extract exact local neighbors. The kernels are not learned; they only encode spatial offsets.

The extracted tensors are still explicit:

```text
K_neighbors: [B, heads, M*M, d_head, H, W]
V_neighbors: [B, heads, M*M, d_head, H, W]
Q_center:    [B, heads, 1,   d_head, H, W]
```

`LocalAttention2d` then applies the exact local-attention steps:

1. extract aligned `K` and `V` neighbors with the fixed shift bank
2. compute local dot products `Q · K`
3. scale by `1 / sqrt(d_head)`
4. mask out-of-bounds offsets (skipped when using reflected boundaries; see below)
5. softmax over the `M*M` offset dimension only
6. take the weighted sum over aligned `V`

**Boundary padding:** The default is zero padding outside the feature map, with invalid stencil positions masked before softmax. You can use **reflection** at the edges instead: set `attention.local_attention_boundary_pad: reflect` in YAML (or `HEAAttentionConfig.local_attention_boundary_pad` / `trunk.local_attention_boundary_pad`). Neighbors are then read from a mirrored map (PyTorch `reflect`; on tiny H×W where reflect is illegal, `replicate` is used). The flattened reference backend (`implementation: flattened`) only supports `zeros`.

This optimized path is meant to take advantage of backend convolution kernels on CPU, CUDA, and especially MPS/CUDA-capable hardware.

### Readable shift-bank reference

`ShiftLocalAttention2d` preserves the original pad-and-slice implementation as a direct reference path. It exposes the fixed neighborhood extraction very explicitly and is useful for debugging and equivalence checks.

### Reference paths

- `ReferenceLocalAttention2d` uses `torch.nn.functional.unfold`, which makes the local extraction especially transparent.
- `FlattenedMaskedLocalAttention2d` flattens the 2D grid to `L = H * W`, builds a transformer-style `L x L` local mask from 2D coordinates, and runs standard masked attention. This path is for correctness checks on small images, not for scalability.

### Transformer-style blocks

- `LocalTransformerBlock2d` is a pre-norm block with channel-wise `LayerNorm`, learned `1x1` QKV projections, fixed-scaffold local attention, output projection, residual connections, and a GELU MLP.
- `ReferenceLocalTransformerBlock2d` uses the flattened masked-attention reference path while keeping the same block structure and parameterization.

## Installation

```bash
pip install torch  # or follow the PyTorch install selector for your platform/backend
pip install -e .
```

`torch` is intentionally not forced as a hard packaging dependency so you can choose the right PyTorch build for your machine first, including CPU, CUDA, or MPS-enabled installs.

### Dense LeJEPA / VRAM

**View rotation (grid-aligned):** enable **`lejepa.views.pre_corrupt_rotation`** to apply **rotate(θ) → photometric corruption → rotate(−θ)** per view so **pixel / token indices stay registered** with the original image while corruptions are applied in a random orientation. Use **`pre_corrupt_rotation_padding`** (**`reflection`** recommended) to reduce dark boundary wedges vs **`zeros`**. Set **`pre_corrupt_rotation_deg=(low, high)`** for **uniform continuous** θ, or **`pre_corrupt_rotation_quarter_turns=True`** for θ uniform on **{0°, 90°, 180°, 270°}** (no arbitrary-angle `grid_sample` blur). Use **`(0, 0)`** for deg with the flag on for a no-op path (when quarter turns is off).

Dense LeJEPA runs the backbone on **`batch_size × num_views`** images per step unless you enable **`lejepa.sequential_view_forward`** (runs one view at a time; lower peak memory, slightly slower). Set **`latent.sources`** to a list of hooks (`encoder_<k>`, `bottleneck`, `decoder_<k>`, `top`) to train **multiple** dense latent heads. Use **`latent.step_mode`**: **`joint`** averages the loss over all hooks every batch; **`rotate`** trains one hook per batch — pass **`rotate_latent_index=global_batch_step`** (e.g. increment each batch) so hooks cycle (`global_step % num_hooks`). That lowers per-step projector/SIGReg work versus **`joint`**. Use **`default_all_latent_hooks(num_scales)`** for every pyramid location. Single-hook **`latent.source`** still works when **`latent.sources`** is unset. If you hit CUDA OOM, try in order: **lower `batch_size`**, enable **`sequential_view_forward`**, enable **`backbone_gradient_checkpointing`** on **`HEAUNetModelConfig`** (recomputes backbone chunks on backward; lowers activation memory; prefer **GroupNorm** over BatchNorm with checkpointing), reduce **`num_views`** (minimum 2), use fewer **`latent.sources`**, shrink **`base_channels` / `channel_multipliers`**, or lower **`sigreg.num_slices`**. **Note:** UNet skip connections mean **`rotate`** still runs a **full** backbone forward/backward — it only skips inactive **projector** heads in the graph, not decoder stages. PyTorch also suggests `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when fragmentation is an issue.

## Quick usage

```python
import torch
from local_conv_attention import LocalAttention2d, LocalSelfAttention2d, ShiftLocalAttention2d

q = torch.randn(1, 64, 16, 16)
k = torch.randn(1, 64, 16, 16)
v = torch.randn(1, 64, 16, 16)

optimized_attn = LocalAttention2d(num_heads=4, window_size=3)
y = optimized_attn(q, k, v)

shift_reference = ShiftLocalAttention2d(num_heads=4, window_size=3)
y_ref = shift_reference(q, k, v)

self_attn = LocalSelfAttention2d(dim=64, num_heads=4, window_size=3)
z = self_attn(torch.randn(1, 64, 16, 16))
```

## HEA-UNet Instance Segmentation

The repository now also includes a bottom-up instance segmentation variant built around the existing Hierarchical Elevator Attention trunk. The HEA-UNet trunk still acts as the centerpiece: low-resolution semantic memories are built with the local transformer operators already in the repo, then injected back into the top-resolution decoder feature map.

On top of that refined high-resolution feature map, a lightweight `BottomUpInstanceHead2d` predicts foreground logits, center heatmaps, and per-pixel offsets to instance centers. The design is intentionally simple and modular so it is easy to train and easy to ablate against a plain U-Net trunk.

```python
from local_conv_attention import (
    HEAExperimentConfig,
    HEAUNetModelConfig,
    HEAUNetInstanceModel,
    BottomUpInstanceLoss,
    build_instance_targets,
    decode_instances,
)

experiment = HEAExperimentConfig(
    model=HEAUNetModelConfig(
        name="hea_unet_instance",
        in_channels=1,
        num_classes=2,
        base_channels=16,
    )
)
model = HEAUNetInstanceModel(experiment.model, postprocess_config=experiment.postprocess)
criterion = BottomUpInstanceLoss(experiment.loss)

images = torch.randn(2, 1, 128, 128)
instance_labels = torch.zeros(2, 128, 128, dtype=torch.long)
targets = build_instance_targets(instance_labels, experiment.targets)
predictions = model(images)
loss_dict = criterion(predictions, targets)
decoded = decode_instances(predictions, experiment.postprocess)
```

## HEA Region Explanations

The repository also includes a region-level explanation toolkit for HEA fusion. It explains where low-resolution semantic context was retrieved from for a chosen output pixel or patch, rather than trying to interpret individual feature channels.

`HEAExplainer` can report:

- raw attention over coarse source regions
- gated contextual contribution magnitude per region
- signed contribution to a simple top-stage segmentation logit when the final head is a `1x1` projection
- ablation-based score changes after zeroing a selected semantic-memory region

This is a mechanistic trace of HEA's contextual retrieval path into a chosen decoder stage. It is not a full causal explanation of the whole model. The ablation mode is slower, but it provides a stronger sanity check.

```python
from local_conv_attention import HEAExplainer

explainer = HEAExplainer(model)
result = explainer.explain_pixel(
    image,
    target_xy=(96, 128),
    stage="top",
    mode="gated_magnitude",
)
figure = explainer.visualize_explanation(image[0], result)
```

## Dense LeJEPA Pretraining

The repository now also includes a dense, teacher-free LeJEPA-style pretraining adaptor for the HEA backbone. Multiple aligned views of the same image are passed through the same `HEABackbone` and a dense projector to produce latent maps of shape `[B, V, D, H_lat, W_lat]`.

Training uses a symmetric objective:

- dense invariance across views at matching spatial positions
- `SIGReg` on the collection of dense latent vectors
- no EMA teacher, no predictor head, and no stop-gradient branch

```python
from local_conv_attention import build_model_from_yaml

model = build_model_from_yaml("configs/hea_dense_lejepa_default.yaml")
out = model(images)
print(out["latents"].shape, out["inv_loss"], out["sigreg_loss"], out["loss"])
```

**Multi-GPU notebook / spawn** (`examples/dense_lejepa_ddp_spawn.py`): uses `torch.multiprocessing.spawn` (needs ≥2 visible GPUs). Each run writes a timestamped folder under `examples/dense_lejepa_ddp_outputs/` (override with `--output-dir`) containing:

- `config.json` — full experiment config to rebuild the architecture
- `architecture.txt` — `str(model)` plus parameter count
- `checkpoint_epoch_####.pt` — weights, optimizer, scalar history, and embedded `config_dict` **every epoch**
- `checkpoint_latest.pt` — copy of the last epoch (stable path for notebooks)
- `scalars.json` — per-epoch losses

The default `DiscSquareDataset` applies **integer pixel shifts** (zero-padded, default range `[-32, 32]` per axis) **then** a **uniform random rotation** in `[0°, 360°)` (via `shift_image_2d_zero_pad` then `rotate_image_2d`). Use `random_shift=False` / `random_rotation=False`, or `shift_px_range=`, `rotation_range_deg=`, and `generator=` to control augmentation and seeding.

Load and run on new images:

```python
from examples.dense_lejepa_ddp_spawn import load_dense_lejepa_from_checkpoint

model, experiment_cfg = load_dense_lejepa_from_checkpoint(
    "examples/dense_lejepa_ddp_outputs/<run>/checkpoint_latest.pt",
    map_location="cpu",
)
model.eval()
```

**Hybrid encoder (conv stem + local attention) + dense LeJEPA:** build `HEAUNetModelConfig` with `name="hybrid_dense_lejepa"`, a non-null `hybrid_encoder` (`HybridConvAttentionEncoderConfig`), and `latent.source="encoder_out"` (only supported hook). Use `HybridDenseLeJEPAModel` or `build_model(config)`. Multi-GPU spawn: `examples/hybrid_dense_lejepa_ddp_spawn.py` → artifacts under `examples/hybrid_dense_lejepa_ddp_outputs/`; load with `load_hybrid_dense_lejepa_from_checkpoint` in that module.

**Notebook:** `examples/inspect_dense_lejepa_checkpoint.ipynb` — browse `dense_lejepa_ddp_outputs/`, load a checkpoint on CPU, run inference on synthetic or custom grayscale images, plot per-hook latent norm maps, and run **linear pixel-decoder + edge-correlation probes** to relate latents to input intensity.

## Running tests

```bash
python -m pytest
```

The test suite includes:

- optimized conv-backed vs. explicit shift-bank equivalence
- optimized conv-backed vs. unfold equivalence
- optimized conv-backed vs. flattened masked-attention equivalence on small images
- block-level equivalence with shared weights
- gradient equivalence for operator and block paths
- sanity checks for `window_size = 1`, border masking, and uniform-score averaging

## Running benchmarks

```bash
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_block.py
```

Each benchmark prints a readable table and writes a CSV file under `benchmarks/`.

Useful variants:

```bash
python benchmarks/benchmark_attention.py --device mps
python benchmarks/benchmark_attention.py --device mps --compile --compile-mode reduce-overhead --channels-last
python benchmarks/benchmark_block.py --device mps --compile --compile-mode reduce-overhead --channels-last
```

Notes:

- The benchmark scripts automatically prefer `cuda`, then `mps`, then `cpu`.
- The operator benchmark compares the optimized `conv2d` backend, the explicit shift reference, the `unfold` reference, and the flattened masked-attention reference.
- The block benchmark compares optimized blocks, shift-reference blocks, and flattened-reference blocks.
- Benchmarks now report estimated backward-only time and `bwd/fwd`, the ratio of backward time to forward time.
- The flattened transformer-style reference is intentionally restricted to smaller images because it scales quadratically in the flattened sequence length.
- CUDA peak memory is reported when available.
- Because backend kernel quality differs by device, the optimized grouped-conv path is most likely to pay off on accelerators such as MPS or CUDA. It may or may not be the fastest option on CPU for every shape.
- On MPS, `unfold`-style references may fall back to CPU for parts of backward due to missing backend kernels. That is a property of the backend, not of the local-attention math.

### Interpreting gradient efficiency

The benchmark CSVs include:

- `forward_seconds`: eager or compiled forward pass time
- `backward_seconds_estimate`: directly timed backward pass cost for a fresh forward graph
- `backward_over_forward_ratio`: how expensive gradient computation is relative to inference for that operator
- `forward_share_of_train_step`: the fraction of the training step spent in forward rather than backward

For training-oriented comparisons, `backward_seconds_estimate` and `backward_over_forward_ratio` are the most useful gradient-efficiency metrics.

## Equivalence being demonstrated

This toolkit demonstrates numerical equivalence among four views of the same local operator:

1. an optimized grouped-`conv2d` fixed shift bank
2. an explicit non-learned pad-and-slice shift bank
3. a direct `unfold` / local-extraction implementation
4. a flattened transformer-style attention implementation with an explicit local-neighborhood mask derived from 2D coordinates

The repository proves that the fixed spatial operator structure can match local-window transformer attention exactly, while remaining explicit about the convolution-like neighborhood geometry.
