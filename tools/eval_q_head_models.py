#!/usr/bin/env python3
"""
Evaluate trained Q heads (ViT-based or CNN-based) on retro datasets and report L1/MSE losses.
Supports the same dataset configuration used during training (either explicit --datasets or a YAML data-config).
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.vjepa.utils import init_video_model
from app.vjepa_droid.sample_utils import unpack_sample
from src.utils.logging import get_logger
from tools.eval_retro_predictor import build_dataset, encode_clip, make_eval_transform

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Q-head models on retro datasets")
    parser.add_argument("--model-type", choices=["vit", "cnn"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to the trained Q-head checkpoint.")
    parser.add_argument("--fname", required=True, help="Training config YAML for data geometry.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Retro dataset directories.")
    parser.add_argument("--manifest", nargs="*", default=None, help="Manifest files.")
    parser.add_argument("--action-mappings", nargs="*", default=None, help="Action mapping JSON files.")
    parser.add_argument("--data-config", type=str, default=None, help="YAML listing multiple dataset entries.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--frames-per-clip", type=int, default=2)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--recompute-returns", action="store_true")
    parser.add_argument("--pooling", choices=["mean", "concat"], default="mean", help="For ViT Q-heads.")
    parser.add_argument("--encoder-checkpoint", type=str, help="ViT encoder checkpoint (for model-type=vit).")
    parser.add_argument("--summary", type=str, default="q_head_eval_summary.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_dataset_entries(args) -> List[Dict]:
    entries: List[Dict] = []
    from yaml import safe_load

    if args.data_config:
        data_cfg = safe_load(Path(args.data_config).read_text())
        for entry in data_cfg.get("datasets", []):
            entries.append(
                {
                    "paths": ensure_list(entry.get("paths") or entry.get("datasets")),
                    "manifest": ensure_list(entry.get("manifest_paths") or entry.get("manifest")),
                    "action_mappings": ensure_list(entry.get("action_mappings") or entry.get("action_mapping")),
                    "name": entry.get("name"),
                }
            )
    elif args.datasets:
        entries.append(
            {
                "paths": ensure_list(args.datasets),
                "manifest": ensure_list(args.manifest),
                "action_mappings": ensure_list(args.action_mappings),
                "name": None,
            }
        )
    return entries


class ViTEncoder:
    def __init__(self, cfg: Dict, checkpoint: str, device: torch.device, frames_per_clip: int):
        self.cfg = cfg
        self.device = device
        self.frames_per_clip = frames_per_clip
        crop_size = cfg["data"]["crop_size"]
        patch_size = cfg["data"]["patch_size"]
        tubelet_size = cfg["data"]["tubelet_size"]
        self.tokens_per_frame = int((crop_size // patch_size) ** 2)
        self.tubelet_size = tubelet_size
        self.encoder, _ = init_video_model(
            device=device,
            patch_size=patch_size,
            max_num_frames=frames_per_clip,
            tubelet_size=tubelet_size,
            model_name=cfg["model"]["model_name"],
            crop_size=crop_size,
            pred_depth=cfg["model"]["pred_depth"],
            pred_num_heads=cfg["model"].get("pred_num_heads"),
            pred_embed_dim=cfg["model"]["pred_embed_dim"],
            use_sdpa=cfg["meta"].get("use_sdpa", False),
            use_silu=cfg["model"].get("use_silu", False),
            use_pred_silu=cfg["model"].get("use_pred_silu", False),
            wide_silu=cfg["model"].get("wide_silu", True),
            use_rope=cfg["model"].get("use_rope", True),
            use_activation_checkpointing=False,
        )
        payload = torch.load(checkpoint, map_location="cpu")
        self.encoder.load_state_dict(payload["encoder"], strict=False)
        self.encoder.eval()
        self.embed_dim = self.encoder.embed_dim

    def encode_tokens(self, clips: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = encode_clip(
                self.encoder,
                clips,
                self.frames_per_clip,
                self.tokens_per_frame,
                self.tubelet_size,
            )
        tokens = feats.view(clips.size(0), self.frames_per_clip, self.tokens_per_frame, self.embed_dim)
        return tokens


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    b, t = rewards.shape
    out = torch.zeros_like(rewards)
    running = torch.zeros(b, device=rewards.device, dtype=rewards.dtype)
    for idx in reversed(range(t)):
        running = rewards[:, idx] + gamma * running
        out[:, idx] = running
    return out


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.output_dim = 64 * 7 * 7

    def forward(self, x):
        return self.net(x)


class QHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = max(hidden_dim, 64)
        mid_dim = max(hidden_dim // 2, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim + action_dim),
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, state_latent, action_vec):
        x = torch.cat([state_latent, action_vec], dim=-1)
        return self.net(x).squeeze(-1)


def evaluate_vit_entry(entry, args, cfg, device):
    transform = make_eval_transform(cfg["data"]["crop_size"])
    dataset = build_dataset(
        entry["paths"],
        frames_per_clip=args.frames_per_clip,
        transform=transform,
        action_dim=None,
        manifest_paths=entry["manifest"] or None,
        action_mappings=entry["action_mappings"] or None,
        include_returns=True,
    )
    action_dim = dataset.action_dim
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
    )
    vit_encoder = ViTEncoder(cfg, args.encoder_checkpoint, device, args.frames_per_clip)
    feature_dim = vit_encoder.embed_dim if args.pooling == "mean" else vit_encoder.embed_dim * vit_encoder.tokens_per_frame

    action_proj = nn.Linear(action_dim, vit_encoder.embed_dim).to(device)
    q_head = QHead(feature_dim, action_dim, hidden_dim=512).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    action_proj.load_state_dict(payload["action_proj"])
    q_head.load_state_dict(payload["q_head"])
    action_proj.eval()
    q_head.eval()

    total_l1 = 0.0
    total_mse = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            (
                clips,
                raw_actions,
                _states,
                _extrinsics,
                rewards,
                returns,
                _action_latents,
                _raw_clips,
                _,
            ) = unpack_sample(batch, include_returns=True, include_action_latents=False, include_raw_clips=False)

            clips = clips.to(device)
            raw_actions = raw_actions.to(device, dtype=torch.float32)
            rewards = rewards.to(device, dtype=torch.float32)
            returns = returns.to(device, dtype=torch.float32)

            tokens = vit_encoder.encode_tokens(clips)
            steps = min(tokens.size(1) - 1, raw_actions.size(1))
            if steps <= 0:
                continue
            state_tokens = tokens[:, :steps, :, :]
            actions = raw_actions[:, :steps, :]
            if args.recompute_returns or returns.abs().sum() == 0:
                targets = discounted_returns(rewards, args.discount)[:, :steps]
            else:
                targets = returns[:, :steps]

            action_emb = action_proj(actions).unsqueeze(2)
            conditioned = state_tokens + action_emb

            if args.pooling == "mean":
                features = conditioned.mean(dim=2)
            else:
                features = conditioned.flatten(2)

            preds = q_head(features.reshape(-1, feature_dim))
            diffs = preds - targets.reshape(-1)
            total_mse += torch.mean(diffs ** 2).item() * preds.numel()
            total_l1 += torch.mean(torch.abs(diffs)).item() * preds.numel()
            total_count += preds.numel()

    denom = max(total_count, 1)
    return total_l1 / denom, total_mse / denom


def clip_transform(buffer):
    tensor = (
        torch.from_numpy(buffer)
        .permute(0, 3, 1, 2)
        .float()
        / 255.0
    )
    tensor = torch.nn.functional.interpolate(tensor, size=(84, 84), mode="bilinear", align_corners=False)
    tensor = tensor.permute(1, 0, 2, 3).contiguous()
    return tensor


def evaluate_cnn_entry(entry, args, device):
    dataset = build_dataset(
        entry["paths"],
        frames_per_clip=args.frames_per_clip,
        transform=clip_transform,
        action_dim=None,
        manifest_paths=entry["manifest"] or None,
        action_mappings=entry["action_mappings"] or None,
        include_returns=True,
    )
    action_dim = dataset.action_dim
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
    )

    encoder = CNNEncoder().to(device)
    q_head = QHead(encoder.output_dim, action_dim, hidden_dim=512).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    encoder.load_state_dict(payload["encoder"])
    q_head.load_state_dict(payload["q_head"])
    encoder.eval()
    q_head.eval()

    total_l1 = 0.0
    total_mse = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            (
                clips,
                raw_actions,
                _states,
                _extrinsics,
                rewards,
                returns,
                _action_latents,
                _raw_clips,
                _,
            ) = unpack_sample(batch, include_returns=True, include_action_latents=False, include_raw_clips=False)

            frames = clips.to(device).float()
            actions = raw_actions.to(device, dtype=torch.float32)
            rewards = rewards.to(device, dtype=torch.float32)
            returns = returns.to(device, dtype=torch.float32)
            latents = encoder(frames[:, 0, :3, :, :])
            steps = min(actions.size(1), returns.size(1))
            if steps <= 0:
                continue
            latents = latents[:, None, :].expand(-1, steps, -1)
            if args.recompute_returns or returns.abs().sum() == 0:
                targets = discounted_returns(rewards, args.discount)[:, :steps]
            else:
                targets = returns[:, :steps]
            preds = q_head(
                latents.reshape(-1, encoder.output_dim),
                actions[:, :steps, :].reshape(-1, action_dim),
            )
            diffs = preds - targets.reshape(-1)
            total_mse += torch.mean(diffs ** 2).item() * preds.numel()
            total_l1 += torch.mean(torch.abs(diffs)).item() * preds.numel()
            total_count += preds.numel()

    denom = max(total_count, 1)
    return total_l1 / denom, total_mse / denom


def main():
    args = parse_args()
    entries = build_dataset_entries(args)
    if not entries:
        raise ValueError("No datasets specified for evaluation.")
    from yaml import safe_load

    cfg = safe_load(Path(args.fname).read_text())
    device = torch.device(args.device)

    results = []
    for idx, entry in enumerate(entries):
        name = entry.get("name") or Path(entry["paths"][0]).name
        logger.info(f"Evaluating dataset '{name}'")
        if args.model_type == "vit":
            if not args.encoder_checkpoint:
                raise ValueError("model-type=vit requires --encoder-checkpoint.")
            l1, mse = evaluate_vit_entry(entry, args, cfg, device)
        else:
            l1, mse = evaluate_cnn_entry(entry, args, device)
        logger.info(f"[{name}] L1={l1:.6f} | MSE={mse:.6f}")
        results.append((name, l1, mse))

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "l1", "mse"])
        for row in results:
            writer.writerow(row)
    logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
