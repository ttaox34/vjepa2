# Retro Adaptation Notes

This repository now supports training V-JEPA 2 and V-JEPA 2-AC models on retro game trajectories stored as JSON/PNG frame pairs. The changes below summarize the core additions and adjustments.

## Data Loading
- `app/vjepa_droid/game_dataset.py`: New dataset class that scans one or more folders (or glob patterns) containing `step_*.json`/`step_*.png` files, constructs sliding windows, and pads or trims actions/states so every clip has consistent shapes. It also supports per-game action remapping via JSON mapping files so multiple games can share a common "mega" action vector.
- `app/vjepa_droid/droid.py`: `init_data` now switches between the DROID loader and the new retro dataset. It logs clip counts, validates that trajectories exist, and honours parameters such as `frame_stride`, `state_keys`, and `action_dim`.
- `tools/retro_preprocess.py`: Optional offline cleaner that scans raw retro dumps, detects dark screens (while keeping legitimate "no-action" pauses), and emits a manifest describing which frames to keep and where each episode terminates. Training consumes the manifest instead of touching the raw PNG/JSON files.
- `tools/plot_metric_boxplot.py`: Utility that reads the per-frame metric CSVs produced by evaluation scripts and draws comparative box plots (useful when benchmarking multiple checkpoints or games).
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

Use `python -m app.main --fname configs/train/vitl16/game-retro-256px-8f.yaml --devices cuda:0 --debugmode True` for single-GPU debugging, then drop `--debugmode` to scale out. Ensure `data.datasets` lists every retro folder (or glob) you want included. When manifests and action mappings are available, add them alongside the raw directories:

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
