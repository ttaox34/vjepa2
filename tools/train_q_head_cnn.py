#!/usr/bin/env python3
"""
Action-conditioned Q head training that uses a lightweight CNN encoder instead of the ViT encoder.
This script mirrors tools/train_q_head_from_encoder.py but replaces the ViT latent extraction with a
simple ConvNet (largely mirroring the NatureCNN structure).
"""

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
from tools.eval_retro_predictor import build_dataset, make_eval_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN-based action-conditioned Q head.")
    parser.add_argument("--fname", required=True, help="Training config YAML (for data geometry only).")
    parser.add_argument("--datasets", nargs="+", default=None, help="Paths to retro datasets.")
    parser.add_argument("--manifest", nargs="*", default=None, help="Optional manifest files.")
    parser.add_argument("--action-mappings", nargs="*", default=None, help="Optional action mapping files.")
    parser.add_argument("--data-config", type=str, default=None, help="YAML file describing multiple datasets.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--frames-per-clip", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--recompute-returns", action="store_true")
    parser.add_argument("--logdir", type=str, default="runs/q_head_cnn")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-every-epochs", type=int, default=1)
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
    return device, distributed, rank


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    b, t = rewards.shape
    out = torch.zeros_like(rewards)
    running = torch.zeros(b, device=rewards.device, dtype=rewards.dtype)
    for idx in reversed(range(t)):
        running = rewards[:, idx] + gamma * running
        out[:, idx] = running
    return out


class CNNEncoder(nn.Module):
    """Simple NatureCNN-like encoder that maps frames -> latent vectors."""

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
        self.output_dim = 64 * 7 * 7  # assuming input 84x84

    def forward(self, x):
        return self.net(x)


class QHead(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = max(hidden_dim, 64)
        mid_dim = max(hidden_dim // 2, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim + action_dim),
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, state_latent, action_vec):
        x = torch.cat([state_latent, action_vec], dim=-1)
        return self.net(x).squeeze(-1)


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def main():
    args = parse_args()
    device, distributed, rank = init_distributed(args.device)
    is_master = rank == 0
    from yaml import safe_load

    cfg = safe_load(Path(args.fname).read_text())
    def clip_transform(buffer):
        tensor = (
            torch.from_numpy(buffer)
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )  # [T, C, H, W]
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(84, 84),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.permute(1, 0, 2, 3).contiguous()
        return tensor

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
    datasets_list = []
    for entry in dataset_entries:
        ds = build_dataset(
            entry["datasets"],
            frames_per_clip=args.frames_per_clip,
            transform=clip_transform,
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
                self.action_dim = datasets[0].action_dim

            def __len__(self):
                return int(self.cumulative[-1])

            def __getitem__(self, index):
                ds_idx = np.searchsorted(self.cumulative, index, side="right") - 1
                sample_idx = index - self.cumulative[ds_idx]
                return self.datasets[ds_idx][sample_idx]

        dataset = MultiRetroDataset(datasets_list)

    action_dim = dataset.action_dim
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

    encoder = CNNEncoder().to(device)
    encoder_output_dim = encoder.output_dim
    q_head = QHead(encoder_output_dim, action_dim, args.hidden_dim).to(device)
    if distributed:
        encoder = DDP(encoder, device_ids=[device.index] if device.type == "cuda" else None)
        q_head = DDP(q_head, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(q_head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    logdir = Path(args.logdir).expanduser()
    if is_master:
        logdir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(logdir / "tb"))
    else:
        writer = None

    def save_checkpoint(label: str, epoch: int, step: int):
        if not is_master:
            return
        payload = {
            "encoder": (encoder.module if isinstance(encoder, DDP) else encoder).state_dict(),
            "q_head": (q_head.module if isinstance(q_head, DDP) else q_head).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": step,
        }
        ckpt_path = logdir / f"{label}_e{epoch}_s{step}.pt"
        torch.save(payload, ckpt_path)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running_loss = 0.0
        running_count = 0
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

            frames = clips.to(device).float() / 255.0  # [B, T, C, H, W]
            latents = encoder(frames[:, 0, :3, :, :])
            actions = raw_actions.to(device, dtype=torch.float32)
            rewards = rewards.to(device, dtype=torch.float32)
            returns = returns.to(device, dtype=torch.float32)
            steps = min(actions.size(1), returns.size(1))
            targets = returns[:, :steps]
            latents = latents[:, None, :].expand(-1, steps, -1)
            preds = q_head(
            latents.reshape(-1, encoder_output_dim),
                actions[:, :steps, :].reshape(-1, action_dim),
            )
            loss = F.mse_loss(preds, targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * preds.numel()
            running_count += preds.numel()
            global_step += 1
            if writer is not None:
                writer.add_scalar("q_head/step_mse", loss.item(), global_step)
            if args.save_every_steps and (global_step % args.save_every_steps == 0):
                save_checkpoint("step", epoch, global_step)

        total_loss = torch.tensor([running_loss, running_count], device=device)
        if distributed:
            dist.all_reduce(total_loss)
        avg = total_loss[0].item() / max(total_loss[1].item(), 1)
        if is_master:
            print(f"Epoch {epoch}: MSE {avg:.6f}")
            if writer is not None:
                writer.add_scalar("q_head/epoch_mse", avg, epoch)
            if epoch % args.save_every_epochs == 0:
                save_checkpoint("epoch", epoch, global_step)

    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
