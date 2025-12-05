#!/usr/bin/env python3
"""
Visualize how different encoders organize retro game frames in latent space.

Given one or more checkpoints (label=/path/to/ckpt.pt), this script:
  1. Loads a RetroGameDataset using the geometry from a training config.
  2. Samples up to N clips (optionally filtering specific action indices).
  3. Extracts mean patch tokens from each encoder.
  4. Runs dimensionality reduction (PCA or t-SNE) on the concatenated latents.
  5. Saves a scatter plot colored by action labels and shaped by encoder.

Example:
python tools/visualize_encoder_latents.py \
    --fname configs/train/vitl16/retro-pretrain-256px-16f.yaml \
    --datasets /path/to/game \
    --manifest-paths /path/to/game/manifest.jsonl \
    --action-mappings /path/to/game/action_map.json \
    --checkpoint official=/path/to/vitl_official.pt \
    --checkpoint custom=/path/to/my_run/latest.pt \
    --allowed-actions 4 5 \
    --max-samples 1500 \
    --method tsne \
    --output latent_tsne.png
"""

import argparse
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from app.vjepa.transforms import make_transforms
from app.vjepa_droid.game_dataset import RetroGameDataset
from src.models import vision_transformer as video_vit


COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize latent separability for different encoders.")
    parser.add_argument("--fname", required=True, help="Training config describing clip geometry.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="label=/path/to/ckpt.pt format; supply multiple entries to compare encoders.",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional override for data.datasets.")
    parser.add_argument("--manifest-paths", nargs="*", default=None, help="Optional override for data.manifest_paths.")
    parser.add_argument("--action-mappings", nargs="*", default=None, help="Optional override for data.action_mappings.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum number of clips to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--allowed-actions",
        type=int,
        nargs="*",
        default=None,
        help="Focus on these action indices (e.g. --allowed-actions 4 5 for left/right).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for encoding.")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Dimensionality reduction method.")
    parser.add_argument("--pca-dim", type=int, default=2, help="Target dimension when method=pca.")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--tsne-dim", type=int, default=2, help="Output dim for t-SNE.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("--output", required=True, help="Path to save the scatter plot (png).")
    parser.add_argument("--dump-json", default=None, help="Optional JSON file to store raw embeddings/meta.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def build_transform(crop_size: int):
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )


def build_dataset(cfg: Dict, transform):
    if cfg.get("dataset_type", "retro").lower() != "retro":
        raise ValueError("Only retro datasets are supported for visualization.")
    fpcs = cfg.get("dataset_fpcs", [None])[0]
    if fpcs is None:
        raise ValueError("data.dataset_fpcs must specify frames-per-clip.")

    return RetroGameDataset(
        data_paths=cfg.get("datasets"),
        frames_per_clip=fpcs,
        frame_stride=cfg.get("frame_stride", 1),
        transform=transform,
        action_dim=cfg.get("action_dim"),
        state_keys=cfg.get("state_keys"),
        manifest_paths=cfg.get("manifest_paths"),
        action_mappings=cfg.get("action_mappings"),
        include_returns=False,
        include_action_latents=False,
        manifest_cache=cfg.get("manifest_cache", False),
        manifest_cache_dir=cfg.get("manifest_cache_dir"),
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


def sample_indices(
    dataset: RetroGameDataset,
    max_samples: int,
    allowed_actions: Sequence[int],
    seed: int,
) -> Tuple[List[int], List[int], Dict[int, torch.Tensor]]:
    allowed = set(allowed_actions) if allowed_actions else None
    rng = random.Random(seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)

    chosen_indices: List[int] = []
    labels: List[int] = []
    clip_cache: Dict[int, torch.Tensor] = {}
    for idx in order:
        if len(chosen_indices) >= max_samples:
            break
        sample = dataset[idx]
        actions = sample[1]
        if isinstance(actions, np.ndarray):
            action_tensor = torch.from_numpy(actions)
        elif torch.is_tensor(actions):
            action_tensor = actions
        else:
            action_tensor = torch.as_tensor(actions)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        first_action = action_tensor[0]
        if torch.all(first_action == 0):
            continue
        label = int(torch.argmax(first_action).item())
        if allowed is not None and label not in allowed:
            continue
        chosen_indices.append(idx)
        labels.append(label)
        clip_cache[idx] = _clip_from_sample(sample[0])

    if len(chosen_indices) < 10:
        raise RuntimeError("Insufficient labeled samples for visualization. Relax filters or increase max-samples.")
    return chosen_indices, labels, clip_cache


def build_encoder(model_cfg: Dict, data_cfg: Dict, device: torch.device):
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
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["encoder"]
    clean_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "").replace("backbone.", "")
        clean_state[nk] = v
    encoder.load_state_dict(clean_state, strict=False)


def encode_clips(
    encoder: torch.nn.Module,
    indices: List[int],
    device: torch.device,
    batch_size: int,
    clip_cache: Dict[int, torch.Tensor],
) -> np.ndarray:
    feats: List[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            clips: List[torch.Tensor] = []
            for idx in batch_indices:
                clips.append(clip_cache[idx])
            clip_tensor = torch.cat(clips, dim=0).to(device, non_blocking=True)
            tokens = encoder(clip_tensor)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(1)
            tokens = tokens.view(clip_tensor.size(0), -1, tokens.size(-1))
            feats.append(tokens.mean(dim=1).cpu())
    return torch.cat(feats, dim=0).numpy()


def reduce_latents(method: str, data: np.ndarray, args) -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=args.pca_dim)
        return reducer.fit_transform(data)
    if method == "tsne":
        tsne_kw = {
            "n_components": args.tsne_dim,
            "perplexity": args.tsne_perplexity,
            "init": "pca",
        }
        try:
            reducer = TSNE(learning_rate="auto", **tsne_kw)
        except TypeError:
            reducer = TSNE(**tsne_kw)
        return reducer.fit_transform(data)
    raise ValueError(f"Unsupported method {method}")


def plot_embeddings(
    coords: np.ndarray,
    encoder_ids: List[int],
    labels: List[int],
    checkpoint_names: List[str],
    output_path: str,
):
    plt.figure(figsize=(10, 6))
    unique_encoders = sorted(set(encoder_ids))
    markers = ["o", "s", "^", "D", "P", "X", "*"]
    enc2marker = {enc: markers[i % len(markers)] for i, enc in enumerate(unique_encoders)}

    unique_labels = sorted(set(labels))
    label2color = {label: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, label in enumerate(unique_labels)}

    for idx in range(coords.shape[0]):
        enc = encoder_ids[idx]
        label = labels[idx]
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            marker=enc2marker[enc],
            color=label2color[label],
            edgecolors="none",
            alpha=0.7,
            s=30,
        )

    legend_elements = []
    for enc in unique_encoders:
        marker = enc2marker[enc]
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color="w", label=f"Encoder: {checkpoint_names[enc]}", markerfacecolor="gray", markersize=9))
    for label in unique_labels:
        color = label2color[label]
        legend_elements.append(plt.Line2D([0], [0], marker="o", color="w", label=f"Action {label}", markerfacecolor=color, markersize=9))
    plt.legend(handles=legend_elements, loc="best", fontsize=8)
    plt.title("Encoder latent visualization")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.fname)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    if args.datasets:
        data_cfg = dict(data_cfg)
        data_cfg["datasets"] = args.datasets
    if args.manifest_paths:
        data_cfg = dict(data_cfg)
        data_cfg["manifest_paths"] = args.manifest_paths
    if args.action_mappings:
        data_cfg = dict(data_cfg)
        data_cfg["action_mappings"] = args.action_mappings

    transform = build_transform(data_cfg.get("crop_size", 224))
    dataset = build_dataset(data_cfg, transform)
    indices, labels, clip_cache = sample_indices(dataset, args.max_samples, args.allowed_actions, args.seed)

    checkpoints = []
    for entry in args.checkpoint:
        if "=" not in entry:
            raise ValueError(f"Checkpoint entry must be label=path, got {entry}")
        label, path = entry.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid checkpoint spec: {entry}")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        checkpoints.append((label, path))

    device = torch.device(args.device)
    all_latents = []
    encoder_ids = []

    for enc_id, (label, path) in enumerate(checkpoints):
        encoder = build_encoder(model_cfg, data_cfg, device)
        load_encoder_weights(encoder, path)
        latents = encode_clips(encoder, indices, device, args.batch_size, clip_cache)
        all_latents.append(latents)
        encoder_ids.extend([enc_id] * len(latents))

    concatenated = np.concatenate(all_latents, axis=0)
    coords = reduce_latents(args.method, concatenated, args)
    labels_tiled = labels * len(checkpoints)
    plot_embeddings(coords, encoder_ids, labels_tiled, [name for name, _ in checkpoints], args.output)

    print(f"Saved visualization to {args.output}")
    if args.dump_json:
        payload = {
            "coords": coords.tolist(),
            "encoder_ids": encoder_ids,
            "action_labels": labels_tiled,
            "encoders": [dict(id=i, name=name, checkpoint=path) for i, (name, path) in enumerate(checkpoints)],
            "method": args.method,
        }
        with open(args.dump_json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote raw coordinates to {args.dump_json}")


if __name__ == "__main__":
    main()
