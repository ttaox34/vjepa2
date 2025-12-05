import argparse
import os
from pathlib import Path
from typing import List, Sequence, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from tools.eval_retro_predictor import build_dataset, make_eval_transform
from app.vjepa_droid.sample_utils import unpack_sample


class FrameEncoder(nn.Module):
    """
    Lightweight per-frame conv encoder.
    """

    def __init__(self, in_ch: int = 3, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = width * 4

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames).flatten(1)


class PixelRewardHead(nn.Module):
    """
    Predict reward/value directly from RGB clips.
    """

    def __init__(self, projector_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(projector_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        return self.head(clip_features).squeeze(-1)


class PixelRewardModel(nn.Module):
    """
    Frame-wise encoder + clip pooling + regression head.
    """

    def __init__(self, in_ch: int = 3, width: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.encoder = FrameEncoder(in_ch=in_ch, width=width)
        self.projector = PixelRewardHead(self.encoder.out_dim, hidden_dim)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: [B, C, T, H, W]
        b, c, t, h, w = clip.shape
        frames = clip.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        features = self.encoder(frames).view(b, t, -1).mean(dim=1)
        return self.projector(features)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a pixel-based reward/value predictor directly from RGB clips."
    )
    parser.add_argument("--fname", required=True, help="Training config YAML (used for crop size defaults).")
    parser.add_argument("--datasets", nargs="+", default=None, help="Paths to retro datasets.")
    parser.add_argument("--manifest", nargs="*", default=None, help="Optional manifest files.")
    parser.add_argument("--action-mappings", nargs="*", default=None, help="Optional action mapping files.")
    parser.add_argument("--data-config", type=str, default=None, help="YAML file describing multiple datasets.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--mode", choices=["reward", "value"], default="reward")
    parser.add_argument("--target-step", type=int, default=-1, help="Index of the timestep to supervise (default: last).")
    parser.add_argument("--logdir", type=str, default="runs/pixel_reward_head")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--crop-size", type=int, default=None, help="Override crop size from config.")
    return parser.parse_args()


def setup_distributed(device_override: Optional[str]):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if device_override:
        device = torch.device(device_override)
    else:
        if torch.cuda.is_available():
            idx = local_rank if distributed else 0
            device = torch.device(f"cuda:{idx}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

    rank = dist.get_rank() if dist.is_initialized() else 0
    return device, rank, dist.is_initialized()


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_multi_dataset(entries: List[dict], frames_per_clip: int, transform):
    datasets = []
    for entry in entries:
        ds = build_dataset(
            entry["datasets"],
            frames_per_clip=frames_per_clip,
            transform=transform,
            action_dim=None,
            manifest_paths=entry["manifest"] or None,
            action_mappings=entry["action_mappings"] or None,
            include_returns=True,
            include_action_latents=False,
            return_raw_clips=False,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]

    class MultiRetroDataset(Dataset):
        def __init__(self, parts: Sequence[Dataset]):
            self.parts = list(parts)
            lengths = [len(ds) for ds in self.parts]
            self.cumulative = np.cumsum([0] + lengths)
            self.action_dim = self.parts[0].action_dim

        def __len__(self):
            return int(self.cumulative[-1])

        def __getitem__(self, index):
            ds_idx = np.searchsorted(self.cumulative, index, side="right") - 1
            local_idx = index - self.cumulative[ds_idx]
            return self.parts[ds_idx][local_idx]

    return MultiRetroDataset(datasets)


def main():
    args = parse_args()
    device, rank, distributed = setup_distributed(args.device)
    is_master = rank == 0

    from yaml import safe_load

    cfg = safe_load(Path(args.fname).read_text())
    crop_size = args.crop_size or cfg["data"].get("crop_size", 256)
    transform = make_eval_transform(crop_size)

    dataset_entries = []
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
        raise ValueError("No datasets provided.")
    for entry in dataset_entries:
        if not entry["datasets"]:
            raise ValueError("Dataset entry missing 'paths'.")

    dataset = build_multi_dataset(dataset_entries, args.frames_per_clip, transform)

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

    model = PixelRewardModel(hidden_dim=args.hidden_dim).to(device)
    if distributed:
        device_id = device.index if device.type == "cuda" else None
        model = DDP(model, device_ids=[device_id] if device_id is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logdir = Path(args.logdir).expanduser()
    ckpt_dir = logdir / "checkpoints"
    if is_master:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    writer = None
    if is_master:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(logdir)

    latest_ckpt = ckpt_dir / "latest.pt"
    start_epoch = 1
    global_step = 0
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        if is_master:
            print(f"Resuming from epoch {start_epoch - 1}")

    def select_target(rewards: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        targets = returns if args.mode == "value" else rewards
        idx = args.target_step
        if idx < 0:
            idx = targets.size(1) + idx
        idx = max(0, min(idx, targets.size(1) - 1))
        return targets[:, idx]

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_count = 0
        model.train()
        for batch in loader:
            (
                clips,
                _actions,
                _states,
                _extrinsics,
                rewards,
                returns,
                _latents,
                _raw,
                _indices,
            ) = unpack_sample(batch, include_returns=True)

            clips = clips.to(device, non_blocking=True)
            rewards = rewards.to(device)
            returns = returns.to(device)

            preds = model(clips)
            targets = select_target(rewards, returns)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss * preds.size(0)
            epoch_count += preds.size(0)
            if writer is not None and is_master:
                writer.add_scalar("pixel_reward/step_mse", batch_loss, global_step)
            global_step += 1

        total = torch.tensor([epoch_loss, epoch_count], device=device)
        if dist.is_initialized():
            dist.all_reduce(total)
        avg_loss = (total[0] / max(total[1], 1)).item()

        if is_master:
            print(f"[PixelReward] Epoch {epoch}: MSE {avg_loss:.6f}")
            if writer is not None:
                writer.add_scalar("pixel_reward/train_mse", avg_loss, epoch)
            state = {
                "epoch": epoch,
                "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "mode": args.mode,
                "target_step": args.target_step,
                "frames_per_clip": args.frames_per_clip,
            }
            torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")
            torch.save(state, latest_ckpt)

    if writer is not None:
        writer.close()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
