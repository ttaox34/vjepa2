# Retro Adaptation Notes

This repository now supports training V-JEPA 2 and V-JEPA 2-AC models on retro game trajectories stored as JSON/PNG frame pairs. The changes below summarize the core additions and adjustments.

## Data Loading
- `app/vjepa_droid/game_dataset.py`: New dataset class that scans one or more folders (or glob patterns) containing `step_*.json`/`step_*.png` files, constructs sliding windows, and pads or trims actions/states so every clip has consistent shapes. It also supports per-game action remapping via JSON mapping files so multiple games can share a common "mega" action vector.
- `app/vjepa_droid/droid.py`: `init_data` now switches between the DROID loader and the new retro dataset. It logs clip counts, validates that trajectories exist, and honours parameters such as `frame_stride`, `state_keys`, and `action_dim`.
- `tools/retro_preprocess.py`: Optional offline cleaner that scans raw retro dumps, detects dark screens (while keeping legitimate "no-action" pauses), and emits a manifest describing which frames to keep and where each episode terminates. Training consumes the manifest instead of touching the raw PNG/JSON files.
- `app/vjepa/retro_dataset.py`: Adapter that reuses `RetroGameDataset` inside the vanilla V-JEPA pipeline by exposing samples in the `(clips, label, clip_indices)` format consumed by the multiseq mask collator. Setting `data.dataset_type: retro` in any V-JEPA config now routes through this adapter.
- `tools/plot_metric_boxplot.py`: Utility that reads the per-frame metric CSVs produced by evaluation scripts and draws comparative box (or violin) plots (useful when benchmarking multiple checkpoints or games).
- `tools/train_reward_head.py`: Freezes the trained encoder/predictor, extracts latent features for each transition, and fits a small MLP to predict per-step rewards (using the rewards stored in the JSON trajectories). The resulting head can be reused as a lightweight reward model for downstream agents.
- `configs/train/vitl16/game-retro-256px-8f.yaml`: Example configuration for training a ViT-L AC model on retro data. It shows how to point at multiple directories, set action/state dimensions, and reuse pretrained checkpoints.

## Training Pipeline
- `app/vjepa_droid/train.py`: Updated to read the retro-specific config fields and to propagate the desired `action_embed_dim`. The trainer now:
  - Respects dataset type (`droid` or `retro`) when constructing loaders.
  - Infers `action_embed_dim` and `state_keys` from the JSON payloads when these fields are left unset in the config, so different games with different button layouts work out of the box.
  - Supports per-game action remapping (via `action_mappings`) so multiple games can share one canonical action vector.
  - Derives the actual batch size per iteration to avoid shape mismatches on short batches.
  - Pads/trims temporal sequences before each predictor call and logs detailed diagnostics if a malformed sequence slips through.
- `app/vjepa_droid/utils.py`: Unchanged externally but now receives the correct `max_num_frames` so the ViT encoder matches the retro clip stride.

## Monitoring
- Additional logging highlights dataset initialization and warns if action/state widths are inconsistent, helping trace any future schema issues.

Use `python -m app.main --fname configs/train/vitl16/game-retro-256px-8f.yaml --devices cuda:0 --debugmode True` for action-conditioned training or `configs/train/vitl16/retro-pretrain-256px-16f.yaml` for the pure ViT-L/16 JEPA encoder. Ensure `data.datasets` lists every retro folder (or glob) you want included. When manifests and action mappings are available, add them alongside the raw directories:

```
data:
  datasets:
    - /path/to/raw/game
  manifest_paths:
    - /path/to/raw/game/manifest.jsonl
  action_mappings:
    - /path/to/raw/game/action_map.json
```

For clusters without Slurm, launch with `torchrun`:

```
torchrun --nnodes=8 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=<master_host>:29500 \
  app/main_torchrun.py --fname configs/train/vitl16/game-retro-256px-8f.yaml \
  --folder /path/to/output
```

Set `--nnodes`/`--nproc_per_node` to match your setup and ensure all nodes share the same filesystem for checkpoints and training data.

### Visualising Per-frame Metrics

The evaluation scripts now support `--per-frame-output` to dump per-frame losses or cosine similarities. To compare several runs, use `tools/plot_metric_boxplot.py`:

```
python tools/plot_metric_boxplot.py \
  metrics/runA_loss.csv metrics/runB_loss.csv \
  --labels RunA RunB \
  --value-column loss \
  --plot-type violin \
  --output plots/loss_violin.png
```

Change `--value-column` to `cosine` when visualising cosine similarity CSVs.

### Training a Reward Head

To learn a standalone reward predictor on top of a frozen V-JEPA AC model:

```
python tools/train_reward_head.py \
  --fname configs/train/vitl16/game-retro-256px-8f.yaml \
  --checkpoint /path/to/ac_checkpoint.pt \
  --datasets /path/to/retro/data \
  --manifest /path/to/retro/manifest.jsonl \
  --action-mappings /path/to/retro/action_map.json \
  --epochs 5 --batch-size 64 --frames-per-clip 2 \
  --num-workers 8 --prefetch-factor 4 --persistent-workers --logdir runs/reward_head
```

By default the head trains on predicted latents; pass `--use-target` to learn from ground-truth latents. With `--mode value` (and optional `--discount`), the target becomes a discounted return, which is usually more stable for sparse rewards. The dataset must contain per-frame `reward` fields in the JSON files.

For multi-GPU acceleration, launch via `torchrun` (freeze the encoder/predictor while training a distributed head):

```
torchrun --nnodes=1 --nproc_per_node=4 tools/train_reward_head.py \
  --fname configs/train/vitl16/game-retro-256px-8f.yaml \
  --checkpoint /path/to/ac_checkpoint.pt \
  --datasets /path/to/retro/data \
  --manifest /path/to/retro/manifest.jsonl \
  --action-mappings /path/to/retro/action_map.json \
  --epochs 5 --batch-size 256 --num-workers 8 --prefetch-factor 4 --persistent-workers --logdir runs/reward_head
```

When training on many games, consider using a dataset config (see `configs/reward/multi_game_example.yaml`) and pass it via `--data-config`. Each entry lists raw data folders plus matching manifest/action mapping files, allowing the script to construct a unified dataset automatically. Checkpoints (including `latest.pt`) are written to `--logdir/checkpoints`; pass `--resume` to continue from the most recent epoch.

### Training a Value Head from Encoder Latents

If you just need a value predictor that consumes encoder outputs (without running the AC predictor), use `tools/train_value_head_from_encoder.py`. It freezes the ViT encoder, extracts the target-frame patch tokens, concatenates every patch latent into a single vector per frame, and trains a small MLP to regress discounted returns:

```
python tools/train_value_head_from_encoder.py \
  --fname configs/train/vitl16/retro-pretrain-256px-16f.yaml \
  --checkpoint /path/to/vitl.pt \
  --datasets /path/to/retro/data \
  --manifest /path/to/retro/manifest.jsonl \
  --action-mappings /path/to/retro/action_map.json \
  --frames-per-clip 2 --batch-size 24 --epochs 5 \
  --logdir runs/value_head_encoder
```

By default it concatenates every patch latent per frame before passing them to the MLP; switch to `--pooling mean` if you prefer the (lighter) mean-pooled variant used by the older reward head. The trained head can consume encoder targets or AC predictor latents (same feature space). Set `--target reward` to learn immediate rewards instead of discounted values, and `--recompute-returns` to rebuild returns from raw rewards when the dataset lacks discounted targets.

### Training an Action-conditioned Q Head

To score candidate actions (e.g., proposed by a VLM agent), use `tools/train_q_head_from_encoder.py`. The script freezes the encoder, projects each action vector, adds it to every patch token of the corresponding frame, pools the conditioned tokens (mean or concatenation), and regresses the manifest-provided discounted returns:

```
python tools/train_q_head_from_encoder.py \
  --fname configs/train/vitl16/retro-pretrain-256px-16f.yaml \
  --checkpoint /path/to/vitl.pt \
  --datasets /path/to/retro/data \
  --manifest /path/to/retro/manifest.jsonl \
  --action-mappings /path/to/retro/action_map.json \
  --frames-per-clip 2 --batch-size 24 --epochs 5 \
  --pooling mean --logdir runs/q_head_encoder
```

Switch to `--pooling concat` to keep the entire patch grid (higher dimensional, potentially more expressive). The trained Q head takes a `(state_latent, action_vector)` pair and outputs `Q(s, a)` so you can pick the highest-valued action among VLM suggestions. Use `--recompute-returns` when you want to rebuild discounted returns with a different `--discount`.

For quick experiments, `tools/q_head_demo.py` spins up a Gradio UI: upload a frame (or drop in a file) and list candidate action vectors (one per line), and the page will display each action's Q-value so you can sanity-check rankings before wiring the head into your VLM control loop.

### Action Mapping File Format

Each mapping JSON describes how a game's raw action vector maps into the shared global vector:

```
{
  "map_to": [0, 1, 2, 3, 4, 5, 6, 6, -1],
  "raw_dim": 9,
  "global_dim": 32,
  "reduction": "and"
}
```

- `map_to[i]` lists the global index (or indices) fed by raw button `i`. Use `-1`/`null` to ignore a button.
- `global_dim` should match the configured `model.action_embed_dim`; if omitted, the value from the training config is used.
- When multiple raw buttons map to the same target, the reducer controls the merge strategy (`and` keeps the target high only if **all** contributors are active; `or` uses the maximum; `mean` averages them).

States are currently zero-filled (you can set `state_keys: ["noob"]`), so the predictor relies purely on encoder context plus the mapped global action vector.
