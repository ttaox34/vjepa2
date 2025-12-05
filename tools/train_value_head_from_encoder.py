#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Train a value head directly on encoder targets (no predictor). The head sees the
# full patch token sequence for each target frame by concatenating all patch latents.

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from app.vjepa_droid.sample_utils import unpack_sample
from app.vjepa_droid.utils import init_video_model
from tools.eval_retro_predictor import build_dataset, encode_clip, make_eval_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a value head on frozen encoder latents (no AC predictor required)."
    )
    parser.add_argument("--fname", required=True, help="Training config YAML (used to build encoder geometry).")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint containing the encoder weights.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Paths to retro datasets.")
    parser.add_argument("--manifest", nargs="*", default=None, help="Optional manifest files.")
    parser.add_argument("--action-mappings", nargs="*", default=None, help="Optional action mapping files.")
    parser.add_argument("--data-config", type=str, default=None, help="YAML file describing multiple datasets.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--frames-per-clip", type=int, default=2, help="Clip length to request from dataset.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden width of the MLP head.")
    parser.add_argument(
        "--pooling",
        choices=["concat", "mean"],
        default="concat",
        help="How to aggregate patch tokens per frame before feeding the value head.",
    )
    parser.add_argument(
        "--target",
        choices=["value", "reward"],
        default="value",
        help="Train on discounted returns (`value`) or immediate rewards (`reward`).",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="Discount factor when recomputing returns from rewards.",
    )
    parser.add_argument(
        "--recompute-returns",
        action="store_true",
        help="Always recompute discounted returns from rewards instead of trusting dataset-provided returns.",
    )
    parser.add_argument("--logdir", type=str, default="runs/value_head_encoder", help="Directory for logs/checkpoints.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in logdir if it exists.")
    parser.add_argument("--device", default=None, help="Optional device override (defaults to cuda:<local_rank>).")
    return parser.parse_args()


def init_distributed(device_override: Optional[str]):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if distributed and "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    if device_override:
        device = torch.device(device_override)
        if distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
    else:
        if distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        target = device.index if device.index is not None else (local_rank if distributed else 0)
        torch.cuda.set_device(target)
        device = torch.device(f"cuda:{target}")

    rank = dist.get_rank() if dist.is_initialized() else 0
    return device, distributed, rank, local_rank


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def load_encoder(cfg: Dict, device: torch.device, checkpoint_path: str, frames_per_clip: int):
    crop_size = cfg["data"]["crop_size"]
    patch_size = cfg["data"]["patch_size"]
    tubelet_size = cfg["data"]["tubelet_size"]
    max_num_frames = frames_per_clip
    model_num_frames = max_num_frames * max(1, tubelet_size)

    action_embed_dim = cfg["model"].get("action_embed_dim")
    if action_embed_dim is None:
        action_embed_dim = int(cfg["data"].get("action_dim") or 1)
    encoder, _ = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=model_num_frames,
        tubelet_size=tubelet_size,
        model_name=cfg["model"]["model_name"],
        crop_size=crop_size,
        pred_depth=cfg["model"]["pred_depth"],
        pred_num_heads=cfg["model"].get("pred_num_heads"),
        pred_embed_dim=cfg["model"]["pred_embed_dim"],
        action_embed_dim=action_embed_dim,
        pred_is_frame_causal=cfg["model"].get("pred_is_frame_causal", True),
        use_extrinsics=cfg["model"].get("use_extrinsics", False),
        use_sdpa=cfg["meta"].get("use_sdpa", False),
        use_silu=cfg["model"].get("use_silu", False),
        use_pred_silu=cfg["model"].get("use_pred_silu", False),
        wide_silu=cfg["model"].get("wide_silu", True),
        use_rope=cfg["model"].get("use_rope", True),
        use_activation_checkpointing=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["encoder"]
    cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    encoder.load_state_dict(cleaned, strict=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.to(device)
    return encoder, max_num_frames, patch_size, tubelet_size


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    b, t = rewards.shape
    out = torch.zeros_like(rewards)
    running = torch.zeros(b, device=rewards.device, dtype=rewards.dtype)
    for idx in reversed(range(t)):
        running = rewards[:, idx] + gamma * running
        out[:, idx] = running
    return out


class ValueHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = max(hidden_dim, 32)
        mid_dim = max(hidden_dim // 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    args = parse_args()
    device, distributed, rank, local_rank = init_distributed(args.device)
    is_master = rank == 0

    from yaml import safe_load

    cfg = safe_load(Path(args.fname).read_text())
    transform = make_eval_transform(cfg["data"]["crop_size"])

    dataset_entries: List[Dict] = []
    if args.data_config:
        data_cfg = safe_load(Path(args.data_config).read_text())
        for entry in data_cfg.get("datasets", []):
            dataset_entries.append(
                {
                    "datasets": ensure_list(entry.get("paths") or entry.get("datasets")),
                    "manifest": ensure_list(entry.get("manifest_paths") or entry.get("manifest")),
                    "action_mappings": ensure_list(entry.get("action_mappings") or entry.get("action_mapping")),
                }
            )
    else:
        dataset_entries.append(
            {
                "datasets": ensure_list(args.datasets),
                "manifest": ensure_list(args.manifest),
                "action_mappings": ensure_list(args.action_mappings),
            }
        )
    if not dataset_entries:
        raise ValueError("No datasets provided. Use --datasets or --data-config.")
    for entry in dataset_entries:
        if not entry["datasets"]:
            raise ValueError("Dataset entry missing 'paths'.")

    datasets_list = []
    for entry in dataset_entries:
        ds = build_dataset(
            entry["datasets"],
            frames_per_clip=args.frames_per_clip,
            transform=transform,
            action_dim=None,
            manifest_paths=entry["manifest"] or None,
            action_mappings=entry["action_mappings"] or None,
            include_returns=True,
        )
        datasets_list.append(ds)

    if len(datasets_list) == 1:
        dataset = datasets_list[0]
    else:
        class MultiRetroDataset(Dataset):
            def __init__(self, datasets):
                self.datasets = datasets
                lengths = [len(ds) for ds in datasets]
                self.cumulative = np.cumsum([0] + lengths)

            def __len__(self):
                return int(self.cumulative[-1])

            def __getitem__(self, index):
                ds_idx = np.searchsorted(self.cumulative, index, side="right") - 1
                sample_idx = index - self.cumulative[ds_idx]
                return self.datasets[ds_idx][sample_idx]

        dataset = MultiRetroDataset(datasets_list)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
    )

    encoder, max_num_frames, patch_size, tubelet_size = load_encoder(cfg, device, args.checkpoint, args.frames_per_clip)
    tokens_per_frame = int((cfg["data"]["crop_size"] // patch_size) ** 2)
    embed_dim = encoder.embed_dim
    feature_dim = embed_dim if args.pooling == "mean" else tokens_per_frame * embed_dim
    if is_master:
        mode_desc = "mean pooling" if args.pooling == "mean" else "patch concatenation"
        print(
            f"Value head input dimension: {feature_dim:,} via {mode_desc} "
            f"(tokens_per_frame={tokens_per_frame}, embed_dim={embed_dim})"
        )

    value_head = ValueHead(feature_dim, args.hidden_dim).to(device)
    if distributed:
        device_id = device.index if device.type == "cuda" else None
        value_head = DDP(value_head, device_ids=[device_id] if device_id is not None else None)
    optimizer = torch.optim.AdamW(value_head.parameters(), lr=args.lr, weight_decay=1e-4)

    logdir = Path(args.logdir).expanduser()
    if is_master:
        logdir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(logdir / "tb"))
    else:
        writer = None

    ckpt_path = logdir / "latest.pt"
    start_epoch = 1
    global_step = 0
    if args.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        value_head.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        if is_master:
            print(f"Resuming from epoch {start_epoch-1}")

    def save_checkpoint(epoch):
        if not is_master:
            return
        payload = {
            "model": (value_head.module if isinstance(value_head, DDP) else value_head).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }
        torch.save(payload, ckpt_path)

    def compute_targets(clips: torch.Tensor):
        with torch.no_grad():
            feats = encode_clip(encoder, clips, max_num_frames, tokens_per_frame, tubelet_size)
        z_target = feats[:, tokens_per_frame:, :]
        b = clips.size(0)
        t = z_target.size(1) // tokens_per_frame
        if t == 0:
            raise RuntimeError("Not enough frames to form target latents. Increase frames-per-clip.")
        z_target = z_target.view(b, t, tokens_per_frame, embed_dim)
        if args.pooling == "mean":
            pooled = z_target.mean(dim=2)  # [B, T, embed_dim]
        else:
            pooled = z_target.flatten(2)  # [B, T, tokens_per_frame*embed_dim]
        return pooled, t

    def select_value_targets(rewards, returns, steps):
        if args.target == "reward":
            target_seq = rewards[:, 1 : 1 + steps]
        else:
            if args.recompute_returns or returns.abs().sum() == 0:
                recomputed = discounted_returns(rewards, args.discount)
                target_seq = recomputed[:, 1 : 1 + steps]
            else:
                target_seq = returns[:, 1 : 1 + steps]
        return target_seq

    value_head.train()
    model_ref = value_head.module if isinstance(value_head, DDP) else value_head
    for epoch in range(start_epoch, args.epochs + 1):
        value_head.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        running_loss = 0.0
        running_count = 0
        for batch in loader:
            (
                clips,
                _actions,
                _states,
                _extrinsics,
                rewards,
                returns,
                _action_latents,
                _raw_clips,
                _,
            ) = unpack_sample(batch, include_returns=True, include_action_latents=False, include_raw_clips=False)

            clips = clips.to(device)
            rewards = rewards.to(device, dtype=torch.float32)
            returns = returns.to(device, dtype=torch.float32)

            flat_feats, steps = compute_targets(clips)
            targets = select_value_targets(rewards, returns, steps).to(device, dtype=torch.float32)
            feats = flat_feats.to(device).reshape(-1, feature_dim)
            preds = value_head(feats)
            loss = F.mse_loss(preds, targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.detach() * preds.numel()
            running_loss += batch_loss.item()
            running_count += preds.numel()
            global_step += 1

            if writer is not None:
                writer.add_scalar("value_head/step_mse", loss.item(), global_step)

        total_tensor = torch.tensor([running_loss, running_count], device=device)
        if distributed:
            dist.all_reduce(total_tensor)
        total_loss, total_count = total_tensor.tolist()
        avg_loss = total_loss / max(total_count, 1)
        if is_master:
            print(f"Epoch {epoch}: MSE {avg_loss:.6f}")
            if writer is not None:
                writer.add_scalar("value_head/epoch_mse", avg_loss, epoch)
            save_checkpoint(epoch)

    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
