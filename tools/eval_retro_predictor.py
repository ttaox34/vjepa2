# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from app.vjepa_droid.game_dataset import RetroGameDataset
from app.vjepa_droid.lam_action_encoder import build_lam_encoder
from app.vjepa_droid.sample_utils import unpack_sample
from app.vjepa_droid.utils import init_video_model
from src.utils.logging import get_logger

logger = get_logger(force=True)


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name.strip())
    return cleaned or "dataset"


def build_dataset_entries(args):
    entries = []
    if args.data_config:
        data_cfg = yaml.safe_load(Path(args.data_config).read_text())
        for idx, entry in enumerate(data_cfg.get("datasets", [])):
            paths = ensure_list(entry.get("paths") or entry.get("datasets"))
            if not paths:
                continue
            manifest = ensure_list(entry.get("manifest_paths") or entry.get("manifest"))
            mappings = ensure_list(entry.get("action_mappings") or entry.get("action_mapping"))
            name = entry.get("name") or Path(paths[0]).name or f"dataset_{idx}"
            entries.append(
                {
                    "name": name,
                    "paths": paths,
                    "manifest": manifest,
                    "action_mappings": mappings,
                }
            )
    elif args.datasets:
        name = args.dataset_name or Path(args.datasets[0]).name if args.datasets else "dataset"
        entries.append(
            {
                "name": name,
                "paths": args.datasets,
                "manifest": ensure_list(args.manifest),
                "action_mappings": ensure_list(args.action_mappings),
            }
        )
    return entries


def clean_state_dict(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_models(cfg, action_dim, device, checkpoint):
    crop_size = cfg["data"]["crop_size"]
    patch_size = cfg["data"]["patch_size"]
    tubelet_size = cfg["data"]["tubelet_size"]
    max_num_frames = cfg["data"]["dataset_fpcs"][0]
    model_num_frames = max_num_frames * max(1, tubelet_size)

    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=model_num_frames,
        tubelet_size=tubelet_size,
        model_name=cfg["model"]["model_name"],
        crop_size=crop_size,
        pred_depth=cfg["model"]["pred_depth"],
        pred_num_heads=cfg["model"].get("pred_num_heads"),
        pred_embed_dim=cfg["model"]["pred_embed_dim"],
        action_embed_dim=action_dim,
        pred_is_frame_causal=cfg["model"].get("pred_is_frame_causal", True),
        use_extrinsics=cfg["model"].get("use_extrinsics", False),
        use_sdpa=cfg["meta"].get("use_sdpa", False),
        use_silu=cfg["model"].get("use_silu", False),
        use_pred_silu=cfg["model"].get("use_pred_silu", False),
        wide_silu=cfg["model"].get("wide_silu", True),
        use_rope=cfg["model"].get("use_rope", True),
        use_activation_checkpointing=False,
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    encoder.load_state_dict(clean_state_dict(ckpt["encoder"]), strict=True)
    predictor.load_state_dict(clean_state_dict(ckpt["predictor"]), strict=True)

    encoder.eval().to(device)
    predictor.eval().to(device)
    return encoder, predictor, crop_size, patch_size, tubelet_size, max_num_frames


def make_eval_transform(crop_size: int):
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32) * 255.0
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32) * 255.0

    def _transform(buffer: np.ndarray) -> torch.Tensor:
        frames = []
        for frame in buffer:
            img = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32)
            img = TF.resize(img, [crop_size, crop_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
            frames.append(img)
        tensor = torch.stack(frames, dim=1)  # C T H W
        c, t, h, w = tensor.shape
        tensor = tensor.view(c, -1).permute(1, 0)
        tensor.sub_(mean).div_(std)
        tensor = tensor.permute(1, 0).view(c, t, h, w)
        return tensor

    return _transform


def build_dataset(
    data_paths: Sequence[str],
    frames_per_clip: int,
    transform,
    action_dim: Optional[int],
    manifest_paths: Optional[Sequence[str]] = None,
    action_mappings: Optional[Sequence[str]] = None,
    include_returns: bool = False,
    include_action_latents: bool = False,
    action_latent_suffix: Optional[str] = None,
    return_raw_clips: bool = False,
    raw_resize: Optional[int] = None,
):
    dataset = RetroGameDataset(
        data_paths=data_paths,
        frames_per_clip=frames_per_clip,
        frame_stride=1,
        transform=transform,
        action_dim=action_dim,
        state_keys=None,
        manifest_paths=manifest_paths,
        action_mappings=action_mappings,
        include_returns=include_returns,
        include_action_latents=include_action_latents,
        action_latent_suffix=action_latent_suffix if include_action_latents else None,
        return_raw_clips=return_raw_clips,
        raw_resize=raw_resize,
        require_action_latents=include_action_latents,
    )
    return dataset



def encode_clip(encoder, clip, max_num_frames, tokens_per_frame, tubelet_size):
    with torch.no_grad():
        b = clip.size(0)
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1)
        c = c.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1)
        h = encoder(c)
        return h.view(b, max_num_frames, -1, h.size(-1)).flatten(1, 2)


def predictor_step(predictor, z_context, actions, states, extrinsics, normalize, dim):
    def _prep(x, target_len):
        if x.size(1) == target_len:
            return x
        if x.size(1) < target_len:
            pad = x[:, -1:, :].expand(-1, target_len - x.size(1), -1)
            return torch.cat([x, pad], dim=1)
        return x[:, :target_len, :]

    context_frames = z_context.size(1) // dim
    actions = _prep(actions, context_frames)
    states = _prep(states, context_frames)
    extrinsics = _prep(extrinsics, context_frames)
    z_hat = predictor(z_context, actions, states, extrinsics)
    if normalize:
        z_hat = F.layer_norm(z_hat, (z_hat.size(-1),))
    return z_hat


def _ensure_width(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.size(-1) == target_dim:
        return tensor
    if tensor.size(-1) > target_dim:
        return tensor[..., :target_dim]
    pad_shape = list(tensor.shape[:-1]) + [target_dim - tensor.size(-1)]
    pad = torch.zeros(*pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=-1)


def randomize_actions(actions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Replace action vectors with random multi-hot vectors while preserving the number of active buttons per step.
    """
    if actions.dim() != 3:
        raise ValueError("Expected actions tensor of shape [B, T, D]")
    B, T, D = actions.shape
    device = actions.device
    active_counts = (actions > threshold).sum(dim=-1).to(torch.long)
    randomized = torch.zeros_like(actions)
    for b in range(B):
        for t in range(T):
            k = int(active_counts[b, t])
            if k <= 0:
                continue
            if k >= D:
                randomized[b, t] = 1.0
                continue
            idx = torch.randperm(D, device=device)[:k]
            randomized[b, t, idx] = 1.0
    return randomized


def save_metric_csv(dest_dir: Path, name: str, values: List[Tuple[int, int, float]], header: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with (dest_dir / f"{name}.csv").open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for global_idx, batch_idx, val in values:
            f.write(f"{global_idx},{batch_idx},{val}\n")


def evaluate_metrics(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(Path(args.fname).read_text())
    transform = make_eval_transform(cfg["data"]["crop_size"])
    entries = build_dataset_entries(args)
    if not entries:
        raise ValueError("No datasets specified. Provide --datasets or --data-config.")

    model_requires_latents = cfg["model"].get("use_external_action_tokens", False)
    using_file_latents = bool(args.use_action_latents)
    using_lam = model_requires_latents and not using_file_latents
    if using_file_latents and not model_requires_latents:
        logger.warning("Using action latents even though model.use_external_action_tokens is False.")
    if args.random_actions and (using_file_latents or model_requires_latents):
        raise ValueError("Cannot randomize actions when using external action representations.")

    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    multi = len(entries) > 1
    summary_rows = []

    for idx, entry in enumerate(entries):
        entry_name = entry.get("name") or f"dataset_{idx}"
        entry_dir = base_output / sanitize_name(entry_name) if multi else base_output
        metrics = evaluate_single_entry(
            args=args,
            cfg=cfg,
            transform=transform,
            device=device,
            entry=entry,
            entry_dir=entry_dir,
            using_file_latents=using_file_latents,
            using_lam=using_lam,
            model_requires_latents=model_requires_latents,
        )
        summary_rows.append(metrics)

    summary_csv = base_output / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "l1", "mse", "cosine"])
        for row in summary_rows:
            writer.writerow([row["name"], row["l1"], row["mse"], row["cosine"]])
    logger.info(f"Wrote summary metrics to {summary_csv}")


def evaluate_single_entry(
    args,
    cfg,
    transform,
    device,
    entry,
    entry_dir: Path,
    using_file_latents: bool,
    using_lam: bool,
    model_requires_latents: bool,
):
    entry_name = entry.get("name", "dataset")
    action_dim_override = cfg["data"].get("action_dim") or cfg["model"].get("action_embed_dim")
    if action_dim_override is None:
        raise ValueError(
            "Unable to determine action dimension. Set data.action_dim or model.action_embed_dim in the config."
        )
    dataset = build_dataset(
        entry["paths"],
        frames_per_clip=2,
        transform=transform,
        action_dim=action_dim_override,
        manifest_paths=entry["manifest"] or None,
        action_mappings=entry["action_mappings"] or None,
        include_returns=False,
        include_action_latents=using_file_latents,
        action_latent_suffix=args.action_latent_suffix if using_file_latents else None,
        return_raw_clips=using_lam,
        raw_resize=cfg["data"]["crop_size"] if using_lam else None,
    )
    dataset_action_dim = dataset.action_dim
    logger.info(f"[{entry_name}] action dimension: {dataset_action_dim}")
    action_embed_dim = cfg["model"].get("action_embed_dim", dataset_action_dim)
    if action_embed_dim is None:
        action_embed_dim = dataset_action_dim
    logger.info(f"[{entry_name}] using action embedding dim: {action_embed_dim}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    encoder, predictor, crop_size, patch_size, tubelet_size, max_num_frames = load_models(
        cfg, action_embed_dim, device, args.checkpoint
    )
    lam_encoder = None
    if using_lam:
        lam_encoder = build_lam_encoder(cfg["model"], device)
    normalize_reps = cfg["loss"].get("normalize_reps", False)
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    total_l1 = 0.0
    total_mse = 0.0
    total_cos = 0.0
    total_count = 0
    per_frame_l1: List[Tuple[int, int, float]] = []
    per_frame_mse: List[Tuple[int, int, float]] = []
    per_frame_cos: List[Tuple[int, int, float]] = []
    global_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            (
                clips,
                raw_actions,
                states,
                extrinsics,
                _,
                _,
                action_latents,
                raw_clips,
                _,
            ) = unpack_sample(
                batch,
                include_returns=False,
                include_action_latents=args.use_action_latents,
                include_raw_clips=model_requires_latents,
            )
            clips = clips.to(device)
            raw_actions = _ensure_width(raw_actions.to(device, dtype=torch.float32), action_embed_dim)
            states = _ensure_width(states.to(device, dtype=torch.float32), action_embed_dim)
            extrinsics = _ensure_width(extrinsics.to(device, dtype=torch.float32), action_embed_dim)
            if using_lam:
                if raw_clips is None:
                    raise RuntimeError("Dataset did not return raw clips for LAM inference.")
                actions_tensor = lam_encoder.encode(raw_clips.to(device, non_blocking=True))
            elif using_file_latents:
                if action_latents is None:
                    raise RuntimeError("Dataset did not return action latents.")
                actions_tensor = _ensure_width(action_latents.to(device, dtype=torch.float32), action_embed_dim)
            else:
                actions_tensor = raw_actions
                if args.random_actions:
                    actions_tensor = randomize_actions(actions_tensor)

            h = encode_clip(encoder, clips, max_num_frames, tokens_per_frame, tubelet_size)
            z_context = h[:, :-tokens_per_frame, :]
            z_target = h[:, tokens_per_frame:, :]
            if normalize_reps:
                z_target = F.layer_norm(z_target, (z_target.size(-1),))
            z_hat = predictor_step(
                predictor,
                z_context,
                actions_tensor,
                states[:, :-1],
                extrinsics[:, :-1],
                normalize_reps,
                tokens_per_frame,
            )

            diff = z_hat - z_target
            l1 = diff.abs().mean(dim=-1)
            mse = (diff**2).mean(dim=-1)
            cos = F.cosine_similarity(z_hat, z_target, dim=-1)

            total_l1 += l1.sum().item()
            total_mse += mse.sum().item()
            total_cos += cos.sum().item()
            total_count += l1.numel()

            l1_vals = l1.flatten().cpu().tolist()
            mse_vals = mse.flatten().cpu().tolist()
            cos_vals = cos.flatten().cpu().tolist()
            per_frame_l1.extend((global_idx + i, batch_idx, float(v)) for i, v in enumerate(l1_vals))
            per_frame_mse.extend((global_idx + i, batch_idx, float(v)) for i, v in enumerate(mse_vals))
            per_frame_cos.extend((global_idx + i, batch_idx, float(v)) for i, v in enumerate(cos_vals))
            global_idx += l1.numel()

    denom = max(total_count, 1)
    avg_l1 = total_l1 / denom
    avg_mse = total_mse / denom
    avg_cos = total_cos / denom
    logger.info(f"[{entry_name}] L1: {avg_l1:.6f} | MSE: {avg_mse:.6f} | Cosine: {avg_cos:.6f}")

    entry_dir.mkdir(parents=True, exist_ok=True)
    summary_file = entry_dir / "metrics.txt"
    summary_file.write_text(f"l1={avg_l1}\nmse={avg_mse}\ncosine={avg_cos}\n")

    save_metric_csv(entry_dir, "l1", per_frame_l1, "index,batch,l1")
    save_metric_csv(entry_dir, "mse", per_frame_mse, "index,batch,mse")
    save_metric_csv(entry_dir, "cosine", per_frame_cos, "index,batch,cosine")

    return {"name": entry_name, "l1": avg_l1, "mse": avg_mse, "cosine": avg_cos}


def collect_unique_actions(dataset: RetroGameDataset, max_actions: Optional[int] = None) -> List[Tuple[int, ...]]:
    unique = set()
    for idx in range(len(dataset)):
        _, actions, _, _, _ = dataset[idx]
        for act in actions:
            key = tuple(int(v) for v in act.tolist())
            unique.add(key)
            if max_actions and len(unique) >= max_actions:
                return sorted(unique)
    return sorted(unique)


def build_action_tensor(candidates: Sequence[Tuple[int, ...]], device):
    arr = torch.tensor(candidates, dtype=torch.float32, device=device)
    return arr.unsqueeze(1)


def plot_energy(actions, energies, output_path):
    actions = np.array(actions)
    energies = np.array(energies)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if actions.shape[1] == 1:
        order = np.argsort(actions[:, 0])
        plt.figure()
        plt.plot(actions[order, 0], energies[order], marker="o")
        plt.xlabel("Action 0")
        plt.ylabel("Energy")
    elif actions.shape[1] == 2:
        xs = np.unique(actions[:, 0])
        ys = np.unique(actions[:, 1])
        grid = np.full((len(ys), len(xs)), np.nan)
        for (x, y), e in zip(actions, energies):
            xi = np.where(xs == x)[0][0]
            yi = np.where(ys == y)[0][0]
            grid[yi, xi] = e
        plt.figure()
        plt.imshow(grid, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], aspect="auto", cmap="viridis")
        plt.colorbar(label="Energy (L1)")
        plt.xlabel("Action 0")
        plt.ylabel("Action 1")
    else:
        plt.figure()
        plt.scatter(actions[:, 0], actions[:, 1], c=energies, cmap="viridis", s=60)
        plt.colorbar(label="Energy (L1)")
        plt.xlabel("Action 0 (proj)")
        plt.ylabel("Action 1 (proj)")
    plt.title("Energy Landscape")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_energy_landscape(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(Path(args.fname).read_text())
    if args.use_action_latents:
        raise ValueError("Energy landscape mode does not support external action latents.")
    transform = make_eval_transform(cfg["data"]["crop_size"])
    action_dim_override = cfg["data"].get("action_dim") or cfg["model"].get("action_embed_dim")
    if action_dim_override is None:
        raise ValueError(
            "energy_landscape mode requires data.action_dim or model.action_embed_dim to be set in the config."
        )
    dataset = build_dataset(
        args.datasets,
        frames_per_clip=2,
        transform=transform,
        action_dim=action_dim_override,
        manifest_paths=args.manifest,
        action_mappings=args.action_mappings,
    )
    action_dim = dataset.action_dim
    action_candidates = collect_unique_actions(dataset, max_actions=args.max_actions)
    if len(action_candidates) == 0:
        raise RuntimeError("No actions detected in dataset.")
    logger.info(f"Collected {len(action_candidates)} unique actions for evaluation.")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    encoder, predictor, crop_size, patch_size, tubelet_size, max_num_frames = load_models(
        cfg, action_dim, device, args.checkpoint
    )
    normalize_reps = cfg["loss"].get("normalize_reps", False)
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    sample_idx = min(args.sample_index, len(dataset) - 1)
    clips, actions_true, states, extrinsics, rewards, _ = dataset[sample_idx]
    clips = clips.unsqueeze(0).to(device)
    actions_true = torch.tensor(actions_true, dtype=torch.float32, device=device).unsqueeze(0)
    states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
    extrinsics = torch.tensor(extrinsics, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        h = encode_clip(encoder, clips, max_num_frames, tokens_per_frame, tubelet_size)
        z_start = h[:, :-tokens_per_frame, :]
        z_goal = h[:, tokens_per_frame:, :]
        if normalize_reps:
            z_goal = F.layer_norm(z_goal, (z_goal.size(-1),))

    energies = []
    for act in action_candidates:
        act_tensor = torch.tensor(act, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        pred = predictor_step(
            predictor,
            z_start,
            act_tensor,
            states[:, :1, :],
            extrinsics[:, :1, :],
            normalize_reps,
            tokens_per_frame,
        )
        energy = torch.mean(torch.abs(pred - z_goal)).item()
        energies.append(energy)

    logger.info(f"Computed energies for {len(action_candidates)} candidates.")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w") as f:
            for action, energy in zip(action_candidates, energies):
                f.write(f"{action},{energy}\n")
    if args.plot:
        plot_energy(action_candidates, energies, args.plot)


def parse_args():
    parser = argparse.ArgumentParser(description="Retro predictor evaluation utilities")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--fname", required=True, help="Training config YAML used during training.")
    shared.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint (latest.pt).")
    shared.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Test dataset directories or glob patterns (omit when using --data-config).",
    )
    shared.add_argument("--manifest", nargs="*", default=None, help="Optional manifest JSONL files.")
    shared.add_argument("--action-mappings", nargs="*", default=None, help="Optional action mapping JSON files.")
    shared.add_argument("--data-config", type=str, default=None, help="YAML listing multiple dataset entries.")
    shared.add_argument("--dataset-name", type=str, default=None, help="Optional label for single-dataset runs.")
    shared.add_argument(
        "--use-action-latents",
        action="store_true",
        help="Load precomputed action latent vectors (expects files next to JSON using --action-latent-suffix).",
    )
    shared.add_argument(
        "--action-latent-suffix",
        type=str,
        default=".latent.pt",
        help="File suffix for per-frame action latent tensors.",
    )
    shared.add_argument(
        "--random-actions",
        action="store_true",
        help="Ignore dataset actions during evaluation and replace them with random multi-hot vectors.",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    pred = subparsers.add_parser("prediction", parents=[shared], help="Compute L1, MSE, and cosine metrics.")
    pred.add_argument("--batch-size", type=int, default=8)
    pred.add_argument("--num-workers", type=int, default=4)
    pred.add_argument("--output-dir", type=str, required=True, help="Directory to store metric CSVs and summary.")

    energy = subparsers.add_parser("energy_landscape", parents=[shared], help="Compute action energy landscape.")
    energy.add_argument("--sample-index", type=int, default=0, help="Dataset sample index to visualize.")
    energy.add_argument("--max-actions", type=int, default=None, help="Optional limit on unique actions collected.")
    energy.add_argument("--output", type=str, default=None, help="Optional CSV output for (action, energy).")
    energy.add_argument("--plot", type=str, default="energy_landscape.png", help="Path to save the plot.")

    args = parser.parse_args()
    if args.mode == "prediction" and not (args.datasets or args.data_config):
        parser.error("prediction mode requires either --datasets or --data-config.")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "prediction":
        evaluate_metrics(args)
    elif args.mode == "energy_landscape":
        if not args.datasets:
            raise ValueError("energy_landscape mode requires --datasets.")
        evaluate_energy_landscape(args)
