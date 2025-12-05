#!/usr/bin/env python3
"""
Contrast two (or more) ViT encoders by measuring how well a frozen
linear probe can predict discrete actions from their representations.

This script:
  1. Loads a retro dataset using the same clip geometry as the training config.
  2. Samples clips whose first action timestep is non-zero (optionally restricted
     to specific action indices).
  3. Extracts averaged token features from each checkpoint's encoder.
  4. Fits a small linear probe on a train split and reports train/val accuracy.

Usage example:
python tools/compare_encoder_action_probe.py \
    --fname configs/train/vitl16/retro-pretrain-256px-16f.yaml \
    --checkpoint official=/path/to/vitl_official.pt \
    --checkpoint custom=/path/to/my_run/latest.pt \
    --manifest-paths /path/to/manifest.jsonl \
    --datasets /path/to/raw_game \
    --action-mappings /path/to/action_map.json \
    --max-samples 2000 \
    --allowed-actions 4 5
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml

from app.vjepa.transforms import make_transforms
from app.vjepa_droid.game_dataset import RetroGameDataset
from src.models import vision_transformer as video_vit


def parse_args():
    parser = argparse.ArgumentParser(description="Compare encoder action discrimination via linear probes.")
    parser.add_argument("--fname", required=True, help="Training config (YAML) describing clip geometry.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Identifier and path in the form label=/path/to/ckpt.pt. "
        "Provide multiple entries to compare several models.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional override for data.datasets (defaults to config values).",
    )
    parser.add_argument(
        "--manifest-paths",
        nargs="*",
        default=None,
        help="Optional override for data.manifest_paths (defaults to config values).",
    )
    parser.add_argument(
        "--action-mappings",
        nargs="*",
        default=None,
        help="Optional override for data.action_mappings (defaults to config values).",
    )
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum labeled clips to sample.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of clips used for probe training.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size when extracting encoder features.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and splits.")
    parser.add_argument(
        "--allowed-actions",
        type=int,
        nargs="*",
        default=None,
        help="Subset of action indices to keep (e.g., --allowed-actions 4 5 for left/right).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for feature extraction/probe."
    )
    parser.add_argument("--output", default=None, help="Optional path to dump metrics as JSON.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def build_deterministic_transform(crop_size: int):
    """Disable random augs to ensure identical clips across encoders."""
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )


def build_dataset(data_cfg: Dict, transform):
    if data_cfg.get("dataset_type", "retro").lower() != "retro":
        raise ValueError("This comparison script currently supports retro datasets only.")

    frames_per_clip = data_cfg.get("dataset_fpcs", [None])[0]
    if frames_per_clip is None:
        raise ValueError("data.dataset_fpcs must specify frames per clip for retro datasets.")

    datasets = data_cfg.get("datasets")
    manifest_paths = data_cfg.get("manifest_paths")
    action_mappings = data_cfg.get("action_mappings")

    return RetroGameDataset(
        data_paths=datasets,
        frames_per_clip=frames_per_clip,
        frame_stride=data_cfg.get("frame_stride", 1),
        transform=transform,
        action_dim=data_cfg.get("action_dim"),
        state_keys=data_cfg.get("state_keys"),
        manifest_paths=manifest_paths,
        action_mappings=action_mappings,
        include_returns=False,
        include_action_latents=False,
        return_raw_clips=False,
        manifest_cache=data_cfg.get("manifest_cache", False),
        manifest_cache_dir=data_cfg.get("manifest_cache_dir"),
    )


def _clip_from_sample(sample_clip) -> torch.Tensor:
    clip = sample_clip
    if not torch.is_tensor(clip):
        clip = torch.from_numpy(np.asarray(clip))
    clip = clip.contiguous().float()
    if clip.ndim == 4:
        return clip.unsqueeze(0)
    if clip.ndim == 5:
        return clip
    raise ValueError(f"Unsupported clip shape {clip.shape}")


def select_samples(
    dataset: RetroGameDataset,
    max_samples: int,
    allowed_actions: Sequence[int],
    seed: int,
) -> Tuple[List[Tuple[int, int]], Dict[int, torch.Tensor]]:
    """Return (index, label) pairs for clips whose first action timestep is informative."""
    allowed_set = set(allowed_actions) if allowed_actions else None
    rng = random.Random(seed)
    candidate_indices = list(range(len(dataset)))
    rng.shuffle(candidate_indices)

    selected: List[Tuple[int, int]] = []
    clip_cache: Dict[int, torch.Tensor] = {}
    for idx in candidate_indices:
        if len(selected) >= max_samples:
            break
        sample = dataset[idx]
        actions = sample[1]
        if isinstance(actions, np.ndarray):
            action_tensor = torch.from_numpy(actions)
        elif torch.is_tensor(actions):
            action_tensor = actions
        else:
            action_tensor = torch.as_tensor(actions)

        if action_tensor.numel() == 0:
            continue
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        first_step = action_tensor[0]
        if torch.all(first_step == 0):
            continue
        label = int(torch.argmax(first_step).item())
        if allowed_set is not None and label not in allowed_set:
            continue
        selected.append((idx, label))
        clip_cache[idx] = _clip_from_sample(sample[0])

    if len(selected) < 2:
        raise RuntimeError("Unable to find enough labeled clips. Increase max-samples or relax filters.")

    # Stratify per class to ensure every label appears at least once.
    label_groups: Dict[int, List[int]] = defaultdict(list)
    for idx, label in selected:
        label_groups[label].append(idx)
    for label, indices in label_groups.items():
        if not indices:
            raise RuntimeError(f"No samples collected for label {label}.")

    return selected, clip_cache


def stratified_split(
    pairs: List[Tuple[int, int]],
    train_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, label in pairs:
        by_label[label].append(idx)

    train_pairs: List[Tuple[int, int]] = []
    val_pairs: List[Tuple[int, int]] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_train = max(1, int(len(items) * train_fraction))
        train_items = items[:n_train]
        val_items = items[n_train:] or items[-1:]
        train_pairs.extend((i, label) for i in train_items)
        val_pairs.extend((i, label) for i in val_items)
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    return train_pairs, val_pairs


def remap_labels(pairs: List[Tuple[int, int]]) -> Dict[int, int]:
    labels = sorted({label for _, label in pairs})
    if len(labels) < 2:
        raise RuntimeError("Need at least two distinct action labels for a meaningful probe.")
    return {label: new_idx for new_idx, label in enumerate(labels)}


def compute_features(
    encoder: torch.nn.Module,
    pairs: List[Tuple[int, int]],
    label_map: Dict[int, int],
    device: torch.device,
    batch_size: int,
    clip_cache: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[torch.Tensor] = []
    labels: List[int] = []
    encoder.eval()

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            clips: List[torch.Tensor] = []
            batch_labels = []
            for idx, label in batch_pairs:
                clip = clip_cache[idx]
                clips.append(clip)
                batch_labels.append(label_map[label])
            clip_tensor = torch.cat(clips, dim=0).to(device, non_blocking=True)
            tokens = encoder(clip_tensor)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(1)
            batch_size_local = clip_tensor.shape[0]
            tokens = tokens.view(batch_size_local, -1, tokens.size(-1))
            feats.append(tokens.mean(dim=1).cpu())
            labels.extend(batch_labels)

    return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.long)


def train_linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
) -> Dict[str, float]:
    num_classes = int(train_labels.max().item()) + 1
    feat_dim = train_feats.size(1)

    train_feats = train_feats.to(device)
    val_feats = val_feats.to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)

    probe = torch.nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    best_train = 0.0

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        logits = probe(train_feats)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = (logits.argmax(dim=1) == train_labels).float().mean().item()
            val_logits = probe(val_feats)
            val_acc = (val_logits.argmax(dim=1) == val_labels).float().mean().item()
        if val_acc >= best_val:
            best_val = val_acc
            best_train = train_acc

    return {"train_acc": best_train, "val_acc": best_val}


def build_encoder(model_cfg: Dict, data_cfg: Dict, device: torch.device) -> torch.nn.Module:
    model_name = model_cfg.get("model_name", "vit_large")
    crop_size = data_cfg.get("crop_size", 224)
    patch_size = data_cfg.get("patch_size", 16)
    tubelet_size = data_cfg.get("tubelet_size", 2)
    frames_per_clip = data_cfg.get("dataset_fpcs", [16])[0]

    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=model_cfg.get("uniform_power", False),
        use_sdpa=model_cfg.get("use_sdpa", True),
        use_silu=model_cfg.get("use_silu", False),
        wide_silu=model_cfg.get("wide_silu", True),
        use_activation_checkpointing=False,
        use_rope=model_cfg.get("use_rope", False),
    )
    encoder.to(device)
    encoder.eval()
    return encoder


def load_encoder_weights(encoder: torch.nn.Module, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "encoder" not in payload:
        raise KeyError(f"No 'encoder' key in checkpoint: {checkpoint_path}")
    state = payload["encoder"]
    clean_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "").replace("backbone.", "")
        clean_state[nk] = v
    msg = encoder.load_state_dict(clean_state, strict=False)
    missing = set(msg.missing_keys)
    if missing:
        print(f"[WARN] Missing keys when loading {checkpoint_path}: {sorted(missing)[:5]} ...")


def parse_checkpoint_entry(entry: str) -> Tuple[str, str]:
    if "=" not in entry:
        raise ValueError(f"Checkpoint entry must be label=path, got: {entry}")
    label, path = entry.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid checkpoint entry: {entry}")
    return label, path


def evaluate_checkpoint(
    name: str,
    path: str,
    model_cfg: Dict,
    data_cfg: Dict,
    train_pairs: List[Tuple[int, int]],
    val_pairs: List[Tuple[int, int]],
    label_map: Dict[int, int],
    device: torch.device,
    batch_size: int,
    clip_cache: Dict[int, torch.Tensor],
) -> Dict[str, float]:
    encoder = build_encoder(model_cfg, data_cfg, device)
    load_encoder_weights(encoder, path)
    train_feats, train_labels = compute_features(encoder, train_pairs, label_map, device, batch_size, clip_cache)
    val_feats, val_labels = compute_features(encoder, val_pairs, label_map, device, batch_size, clip_cache)
    metrics = train_linear_probe(train_feats, train_labels, val_feats, val_labels, device)
    metrics.update(
        {
            "model": name,
            "checkpoint": path,
            "num_classes": len(label_map),
            "train_samples": int(train_labels.size(0)),
            "val_samples": int(val_labels.size(0)),
        }
    )
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.fname)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    if args.datasets is not None and args.datasets:
        data_cfg = dict(data_cfg)
        data_cfg["datasets"] = args.datasets
    if args.manifest_paths is not None and args.manifest_paths:
        data_cfg = dict(data_cfg)
        data_cfg["manifest_paths"] = args.manifest_paths
    if args.action_mappings is not None and args.action_mappings:
        data_cfg = dict(data_cfg)
        data_cfg["action_mappings"] = args.action_mappings

    transform = build_deterministic_transform(data_cfg.get("crop_size", 224))
    dataset = build_dataset(data_cfg, transform)

    sample_pairs, clip_cache = select_samples(dataset, args.max_samples, args.allowed_actions, args.seed)
    label_map = remap_labels(sample_pairs)
    train_pairs, val_pairs = stratified_split(sample_pairs, args.train_fraction, args.seed)

    checkpoints = [parse_checkpoint_entry(entry) for entry in args.checkpoint]
    device = torch.device(args.device)

    results = []
    for name, path in checkpoints:
        metrics = evaluate_checkpoint(
            name=name,
            path=path,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_pairs=[(idx, lbl) for idx, lbl in train_pairs],
            val_pairs=[(idx, lbl) for idx, lbl in val_pairs],
            label_map=label_map,
            device=device,
            batch_size=args.batch_size,
            clip_cache=clip_cache,
        )
        results.append(metrics)

    header = f"{'Model':20s} | {'Train Acc':>9s} | {'Val Acc':>8s} | {'Classes':>7s} | {'Train/Val':>11s}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['model']:20s} | "
            f"{row['train_acc']*100:8.2f}% | "
            f"{row['val_acc']*100:7.2f}% | "
            f"{row['num_classes']:7d} | "
            f"{row['train_samples']:4d}/{row['val_samples']:<4d}"
        )

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
