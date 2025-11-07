import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.vjepa_droid.game_dataset import ActionMapper
from tools.eval_retro_predictor import encode_clip, load_models, make_eval_transform, predictor_step
from tools.train_reward_head import RewardHead
from yaml import safe_load


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize reward/value head predictions over a full episode.")
    parser.add_argument("--fname", required=True, help="Training config YAML (same one used for AC model).")
    parser.add_argument("--checkpoint", required=True, help="Path to frozen V-JEPA checkpoint (encoder/predictor).")
    parser.add_argument("--reward-head", required=True, help="Path to trained reward/value head checkpoint.")
    parser.add_argument("--dataset", required=True, help="Retro dataset root directory.")
    parser.add_argument("--manifest", required=True, help="Manifest JSONL produced by preprocessing.")
    parser.add_argument("--action-mapping", default=None, help="Optional action mapping JSON.")
    parser.add_argument("--frames-per-clip", type=int, default=2, help="Clip length used when training the head.")
    parser.add_argument("--episode-index", type=int, default=0, help="Which episode (0-based) to visualize.")
    parser.add_argument("--output", type=str, default="reward_episode.png", help="Path to save the plot.")
    parser.add_argument("--mode", choices=["reward", "value"], default="reward", help="Head training mode (reward/value).")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor if mode=value.")
    parser.add_argument("--device", default=None, help="Device override (e.g. cuda:0).")
    return parser.parse_args()


def read_manifest(manifest_path: Path) -> List[dict]:
    entries: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("keep", True):
                continue
            entries.append(record)
    return entries


def group_real_episodes(manifest_entries: Sequence[dict]) -> List[List[dict]]:
    segments: List[List[dict]] = []
    current: List[dict] = []
    for record in manifest_entries:
        current.append(record)
        if record.get("terminated", False):
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    real_eps: List[List[dict]] = []
    current_ep: List[dict] = []
    for segment in segments:
        current_ep.extend(segment)
        forced = segment[-1].get("forced_termination", False)
        if not forced:
            real_eps.append(current_ep)
            current_ep = []
    if current_ep:
        # leftover without explicit termination; treat as final episode
        real_eps.append(current_ep)
    return real_eps


def load_action_mapper(mapping_path: Optional[Path], global_dim: Optional[int]) -> Optional[ActionMapper]:
    if mapping_path is None:
        return None
    return ActionMapper.from_file(mapping_path, global_dim=global_dim)


def load_step(json_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    image_rel = payload["observation_image_path"]
    image_path = (json_path.parent / image_rel).resolve()
    image = Image.open(image_path).convert("RGB")
    frame = np.asarray(image, dtype=np.uint8)
    action = np.asarray(payload.get("action", []), dtype=np.float32)
    reward = float(payload.get("reward", 0.0))
    return frame, action, reward


def map_action(vec: np.ndarray, mapper: Optional[ActionMapper], action_dim: int) -> np.ndarray:
    if mapper is not None:
        return mapper.map(vec)
    if action_dim <= 0:
        return vec.astype(np.float32, copy=True)
    out = np.zeros((action_dim,), dtype=np.float32)
    limit = min(action_dim, vec.size)
    out[:limit] = vec[:limit]
    return out


def prepare_episode(
    episode_entries: Sequence[dict],
    action_mapper: Optional[ActionMapper],
    action_dim: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    returns: List[float] = []

    for idx, entry in enumerate(episode_entries):
        json_path = Path(entry["json_path"])
        frame, raw_action, reward = load_step(json_path)
        frames.append(frame)
        rewards.append(reward)
        returns.append(float(entry.get("discounted_return", 0.0)))
        if idx < len(episode_entries) - 1:
            actions.append(map_action(raw_action, action_mapper, action_dim))
    return frames, actions, rewards, returns


def compute_predictions(
    frames: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    rewards: Sequence[float],
    returns: Sequence[float],
    encoder,
    predictor,
    reward_head,
    transform,
    device,
    max_num_frames: int,
    tubelet_size: int,
    tokens_per_frame: int,
    normalize_reps: bool,
    use_target: bool,
    mode: str,
    discount: float,
    frames_per_clip: int,
) -> Tuple[np.ndarray, np.ndarray]:
    preds: List[float] = []
    targets: List[float] = []
    actions_arr = np.asarray(actions, dtype=np.float32)
    action_dim = actions_arr.shape[1] if actions_arr.size > 0 else 0
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    returns_arr = np.asarray(returns, dtype=np.float32)

    if frames_per_clip < 2:
        raise ValueError("frames_per_clip must be >= 2 for reward head evaluation.")

    total_steps = max(0, len(frames) - frames_per_clip + 1)
    for start in range(total_steps):
        clip_np = np.stack(frames[start : start + frames_per_clip], axis=0)
        clip_tensor = transform(clip_np).unsqueeze(0).to(device)

        act_steps = frames_per_clip - 1
        act_tensor = torch.zeros((1, act_steps, action_dim), dtype=torch.float32, device=device)
        if action_dim > 0 and actions_arr.size:
            for idx in range(act_steps):
                act_tensor[0, idx] = torch.as_tensor(actions_arr[start + idx], device=device)
        state_tensor = torch.zeros_like(act_tensor)
        extrinsics_tensor = torch.zeros_like(act_tensor)

        with torch.no_grad():
            h = encode_clip(encoder, clip_tensor, max_num_frames, tokens_per_frame, tubelet_size)
            z_context = h[:, :-tokens_per_frame, :]
            z_target = h[:, tokens_per_frame:, :]
            if normalize_reps:
                z_target = F.layer_norm(z_target, (z_target.size(-1),))
            z_pred = predictor_step(
                predictor,
                z_context,
                act_tensor,
                state_tensor,
                extrinsics_tensor,
                normalize_reps,
                dim=tokens_per_frame,
            )
            features = z_target if use_target else z_pred
            features = features.view(1, -1, tokens_per_frame, features.size(-1)).mean(dim=2)
            pred = reward_head(features.reshape(-1, features.size(-1)))
            preds.append(float(pred.item()))

        target_idx = start + frames_per_clip - 1
        if mode == "reward":
            targets.append(float(rewards_arr[target_idx]))
        else:
            if returns_arr.size:
                targets.append(float(returns_arr[target_idx]))
            else:
                future = 0.0
                for t in range(target_idx, len(rewards_arr)):
                    future = rewards_arr[t] + discount * future
                targets.append(float(future))

    return np.array(preds), np.array(targets)


def plot_episode(preds: np.ndarray, targets: np.ndarray, output_path: Path, mode: str):
    steps = np.arange(len(preds))
    plt.figure(figsize=(10, 4))
    plt.plot(steps, targets, label="Ground Truth", linewidth=2)
    plt.plot(steps, preds, label="Prediction", linewidth=2)
    plt.xlabel("Step")
    ylabel = "Reward" if mode == "reward" else "Discounted Return"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over Episode")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = safe_load(Path(args.fname).read_text())
    action_embed_dim = cfg["model"].get("action_embed_dim")

    mapper = load_action_mapper(Path(args.action_mapping) if args.action_mapping else None, action_embed_dim)
    if mapper is not None:
        action_dim = mapper.global_dim
    else:
        action_dim = action_embed_dim or 0

    transform = make_eval_transform(cfg["data"]["crop_size"])
    encoder, predictor, crop_size, patch_size, tubelet_size, max_num_frames = load_models(
        cfg, action_dim, device, args.checkpoint
    )

    reward_ckpt = torch.load(args.reward_head, map_location=device)
    embed_dim = reward_ckpt.get("embed_dim")
    hidden_dim = reward_ckpt.get("hidden_dim", 512)
    use_target = reward_ckpt.get("use_target", False)
    mode = reward_ckpt.get("mode", args.mode)
    discount = reward_ckpt.get("discount", args.discount)
    head = RewardHead(embed_dim, hidden_dim=hidden_dim).to(device)
    head.load_state_dict(reward_ckpt["state_dict"])
    head.eval()

    manifest_entries = read_manifest(Path(args.manifest))
    real_eps = group_real_episodes(manifest_entries)
    if args.episode_index < 0 or args.episode_index >= len(real_eps):
        raise IndexError(f"Episode index {args.episode_index} out of range (found {len(real_eps)} episodes).")

    episode_entries = real_eps[args.episode_index]
    frames, actions, rewards, returns = prepare_episode(episode_entries, mapper, action_dim)

    tokens_per_frame = int((crop_size // patch_size) ** 2)
    normalize_reps = cfg["loss"].get("normalize_reps", False)

    preds, targets = compute_predictions(
        frames,
        actions,
        rewards,
        returns,
        encoder,
        predictor,
        head,
        transform,
        device,
        max_num_frames,
        tubelet_size,
        tokens_per_frame,
        normalize_reps,
        use_target,
        mode,
        discount,
        args.frames_per_clip,
    )

    mse = float(np.mean((preds - targets) ** 2)) if preds.size else 0.0
    mae = float(np.mean(np.abs(preds - targets))) if preds.size else 0.0
    print(f"Episode {args.episode_index}: steps={preds.size}, MSE={mse:.6f}, MAE={mae:.6f}")

    plot_episode(preds, targets, Path(args.output), args.mode)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
