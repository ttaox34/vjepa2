import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from app.vjepa_droid.game_dataset import RetroGameDataset
from app.vjepa_droid.utils import init_video_model
from tools.eval_retro_predictor import (
    build_dataset,
    encode_clip,
    load_models,
    make_eval_transform,
    predictor_step,
)


class RewardHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 512):
        super().__init__()
        hidden_dim = max(hidden_dim, embed_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reward prediction head on top of frozen V-JEPA AC predictor.")
    parser.add_argument("--fname", required=True, help="Training config YAML (same as AC training).")
    parser.add_argument("--checkpoint", required=True, help="Path to frozen V-JEPA checkpoint (contains encoder/predictor).")
    parser.add_argument("--datasets", nargs="+", default=None, help="Paths to retro datasets.")
    parser.add_argument("--manifest", nargs="*", default=None, help="Optional manifest files.")
    parser.add_argument("--action-mappings", nargs="*", default=None, help="Optional action mapping files.")
    parser.add_argument("--data-config", type=str, default=None, help="YAML file describing multiple datasets.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Number of batches prefetched per worker.")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep data-loader workers alive between epochs.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--mode", choices=["reward", "value"], default="reward", help="Train to predict immediate reward or discounted return.")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor when mode=value.")
    parser.add_argument("--frames-per-clip", type=int, default=2)
    parser.add_argument("--device", default=None, help="Optional device override (defaults to cuda:<local_rank> if available).")
    parser.add_argument("--use-target", action="store_true", help="Train on target latents instead of predicted latents.")
    parser.add_argument("--logdir", type=str, default="runs/reward_head", help="Directory for logs and checkpoints.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint in logdir.")
    return parser.parse_args()


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if distributed and "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    if args.device:
        device = torch.device(args.device)
        if distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
    else:
        if distributed:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if device.index is None:
            target_idx = local_rank if distributed else 0
            torch.cuda.set_device(target_idx)
            device = torch.device(f"cuda:{target_idx}")
        else:
            torch.cuda.set_device(device.index)
    rank = dist.get_rank() if dist.is_initialized() else 0
    distributed = dist.is_initialized()
    is_master = (rank == 0)

    from yaml import safe_load

    cfg = safe_load(Path(args.fname).read_text())
    transform = make_eval_transform(cfg["data"]["crop_size"])

    def ensure_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

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

    base_action_dim = datasets_list[0].action_dim
    for ds in datasets_list[1:]:
        if ds.action_dim != base_action_dim:
            raise ValueError("All datasets must share the same action embedding dimension.")

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

    inferred_action_dim = dataset.action_dim

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)

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

    encoder, predictor, crop_size, patch_size, tubelet_size, max_num_frames = load_models(
        cfg,
        inferred_action_dim,
        device,
        args.checkpoint,
    )
    encoder.eval()
    predictor.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)

    tokens_per_frame = int((crop_size // patch_size) ** 2)
    normalize_reps = cfg["loss"].get("normalize_reps", False)

    base_predictor = predictor.module if hasattr(predictor, "module") else predictor
    if hasattr(base_predictor, "predictor_proj"):
        embed_dim = base_predictor.predictor_proj.out_features
    elif hasattr(base_predictor, "predictor_embed"):
        embed_dim = base_predictor.predictor_embed.out_features
    else:
        raise AttributeError("Unable to determine predictor embed dimension.")
    reward_head = RewardHead(embed_dim, hidden_dim=args.hidden_dim).to(device)
    if distributed:
        device_id = None
        if device.type == "cuda":
            device_id = device.index if device.index is not None else local_rank
        reward_head = DDP(reward_head, device_ids=[device_id] if device_id is not None else None)
    optimizer = torch.optim.Adam(reward_head.parameters(), lr=args.lr)

    logdir = Path(args.logdir).expanduser()
    ckpt_dir = logdir / "checkpoints"
    if is_master:
        logdir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    writer = None
    if is_master:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(logdir)

    start_epoch = 1
    global_step = 0
    latest_ckpt = ckpt_dir / "latest.pt"
    if args.resume and latest_ckpt.exists():
        checkpoint = torch.load(latest_ckpt, map_location=device)
        reward_head.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        if is_master:
            print(f"Resuming from epoch {start_epoch - 1} (global_step {global_step})")
    elif args.resume and is_master:
        print("No checkpoint found; starting from epoch 1.")
    if dist.is_initialized():
        dist.barrier()

    def compute_features(clips, actions, states, extrinsics):
        with torch.no_grad():
            h = encode_clip(encoder, clips, max_num_frames, tokens_per_frame, tubelet_size)
            z_context = h[:, :-tokens_per_frame, :]
            z_target = h[:, tokens_per_frame:, :]
            if normalize_reps:
                z_target = F.layer_norm(z_target, (z_target.size(-1),))
            z_pred = predictor_step(
                predictor,
                z_context,
                actions,
                states[:, :-1],
                extrinsics[:, :-1],
                normalize_reps,
                tokens_per_frame,
            )
        features = z_target if args.use_target else z_pred
        return features

    reward_head.train()
    feature_dim = embed_dim
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        count = 0
        for batch in loader:
            clips = batch[0].to(device)
            actions = batch[1].to(device, dtype=torch.float32)
            states = batch[2].to(device, dtype=torch.float32)
            extrinsics = batch[3].to(device, dtype=torch.float32)
            rewards = batch[4].to(device, dtype=torch.float32)
            returns = batch[5].to(device, dtype=torch.float32)

            features = compute_features(clips, actions, states, extrinsics)
            # reshape features to [B, T, tokens_per_frame, embed_dim]
            B = clips.size(0)
            T = features.size(1) // tokens_per_frame
            features = features.view(B, T, tokens_per_frame, -1).mean(dim=2)  # [B, T, embed_dim]
            targets = rewards[:, 1 : 1 + T]
            if args.mode == "value":
                targets = returns[:, 1 : 1 + T]
            pred = reward_head(features.reshape(-1, features.size(-1)))
            loss = F.mse_loss(pred, targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss * pred.numel()
            count += pred.numel()
            feature_dim = features.size(-1)
            if is_master and writer is not None:
                writer.add_scalar("reward_head/step_mse", batch_loss, global_step)
            global_step += 1

        total_tensor = torch.tensor([epoch_loss, count], device=device)
        if dist.is_initialized():
            dist.all_reduce(total_tensor)
        epoch_loss, count = total_tensor.tolist()
        avg = epoch_loss / max(count, 1)
        if is_master:
            print(f"Epoch {epoch}: MSE {avg:.6f}")
            if writer is not None:
                writer.add_scalar("reward_head/train_mse", avg, epoch)

            state_dict = reward_head.module.state_dict() if isinstance(reward_head, DDP) else reward_head.state_dict()
            checkpoint = {
                "epoch": epoch,
                "state_dict": state_dict,
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "embed_dim": feature_dim,
                "hidden_dim": args.hidden_dim,
                "use_target": args.use_target,
                "mode": args.mode,
                "discount": args.discount,
            }
            torch.save(checkpoint, ckpt_dir / f"epoch_{epoch:04d}.pt")
            torch.save(checkpoint, latest_ckpt)

    if writer is not None:
        writer.close()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
