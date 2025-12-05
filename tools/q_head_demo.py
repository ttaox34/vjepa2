#!/usr/bin/env python3
"""Interactive demo for the action-conditioned Q head.

Usage:
    python tools/q_head_demo.py \
        --fname configs/train/vitl16/retro-pretrain-256px-16f.yaml \
        --encoder-checkpoint /path/to/vitl.pt \
        --q-head-checkpoint runs/q_head_encoder/latest.pt \
        --frames-per-clip 2
"""

import argparse
import os
from pathlib import Path
from typing import List

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from tools.eval_retro_predictor import encode_clip, make_eval_transform
from tools.train_q_head_from_encoder import QActionHead, load_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for Q-head scoring.")
    parser.add_argument("--fname", required=True, help="Training config YAML.")
    parser.add_argument("--encoder-checkpoint", required=True, help="Path to encoder checkpoint (.pt).")
    parser.add_argument("--q-head-checkpoint", required=True, help="Path to trained Q-head checkpoint.")
    parser.add_argument("--frames-per-clip", type=int, default=2, help="Clip length expected by the encoder.")
    parser.add_argument("--pooling", choices=["mean", "concat"], default="mean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_q_head(checkpoint_path: str, action_dim: int, embed_dim: int, tokens_per_frame: int, pooling: str, device):
    action_proj = nn.Linear(action_dim, embed_dim)
    feature_dim = embed_dim if pooling == "mean" else tokens_per_frame * embed_dim
    q_head = QActionHead(feature_dim, hidden_dim=512)
    payload = torch.load(checkpoint_path, map_location="cpu")
    action_proj.load_state_dict(payload["action_proj"])
    q_head.load_state_dict(payload["q_head"])
    action_proj.eval().to(device)
    q_head.eval().to(device)
    return action_proj, q_head, feature_dim


def build_clip_from_image(image_path: str, frames_per_clip: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    frame = np.array(img)
    frames = np.stack([frame for _ in range(frames_per_clip)], axis=0)
    return frames


def parse_action_lines(text: str, action_dim: int) -> List[List[float]]:
    actions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        vec = [float(p) for p in parts]
        if len(vec) != action_dim:
            raise ValueError(f"Expected {action_dim} values per action, got {len(vec)}")
        actions.append(vec)
    if not actions:
        raise ValueError("No valid actions provided.")
    return actions


def main():
    args = parse_args()
    device = torch.device(args.device)

    from yaml import safe_load

    cfg = safe_load(Path(args.fname).read_text())
    transform = make_eval_transform(cfg["data"]["crop_size"])
    encoder, max_num_frames, patch_size, tubelet_size = load_encoder(
        cfg, device, args.encoder_checkpoint, args.frames_per_clip
    )
    tokens_per_frame = int((cfg["data"]["crop_size"] // patch_size) ** 2)
    embed_dim = encoder.embed_dim
    action_dim = cfg["model"].get("action_embed_dim")
    if action_dim is None:
        raise ValueError("Config must specify data.action_dim to interpret action vectors.")

    action_proj, q_head, feature_dim = load_q_head(
        args.q_head_checkpoint, action_dim, embed_dim, tokens_per_frame, args.pooling, device
    )

    def score_actions(image_path: str, actions_text: str):
        try:
            actions = parse_action_lines(actions_text, action_dim)
        except Exception as exc:
            return f"Error parsing actions: {exc}"

        if not os.path.exists(image_path):
            return f"Image not found: {image_path}"
        buffer = build_clip_from_image(image_path, args.frames_per_clip)
        clip = transform(buffer).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = encode_clip(encoder, clip, max_num_frames, tokens_per_frame, tubelet_size)
            tokens = feats.view(1, max_num_frames, tokens_per_frame, embed_dim)
            state_tokens = tokens[:, :1, :, :]  # use first frame

            results = []
            for action_vec in actions:
                action_tensor = torch.tensor(action_vec, device=device, dtype=torch.float32).view(1, 1, -1)
                action_emb = action_proj(action_tensor).unsqueeze(2)
                conditioned = state_tokens + action_emb
                if args.pooling == "mean":
                    features = conditioned.mean(dim=2).reshape(1, feature_dim)
                else:
                    features = conditioned.flatten(2).reshape(1, feature_dim)
                q_val = q_head(features).item()
                results.append((action_vec, q_val))

        lines = ["Action\tQ(s,a)"]
        for vec, score in results:
            vec_str = ", ".join(f"{v:.3f}" for v in vec)
            lines.append(f"[{vec_str}]\t{score:.4f}")
        return "\n".join(lines)

    demo = gr.Interface(
        fn=score_actions,
        inputs=[
            gr.Textbox(label="Image Path", placeholder="/path/to/frame.png"),
            gr.Textbox(
                label="Actions (one per line, numbers separated by space/comma)",
                placeholder="1 0 0 0\n0 1 0 0",
                lines=5,
            ),
        ],
        outputs=gr.Textbox(label="Q-values", lines=15),
        title="Q-head Demo",
        description=(
            "Enter an image path and candidate action vectors (one per line). "
            "The Q-head will evaluate each (state, action) pair and return Q-values."
        ),
    )
    demo.launch()


if __name__ == "__main__":
    main()
