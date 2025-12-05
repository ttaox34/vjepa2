import os
from typing import Iterable, List, Optional

import torch
from torch import nn

from latent.lam import UncontrolledDINOLatentActionModel


class LamActionEncoder(nn.Module):
    """
    Thin wrapper that loads a pretrained LAM checkpoint and exposes an encode() helper
    returning flattened action latents matching the predictor embed dimension.
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        checkpoint = config.get("checkpoint")
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError(f"LAM checkpoint not found: {checkpoint}")

        dino_version = config.get("dino_version", "v3")
        dino_model_path = config.get("dino_model_path")
        model_dim = config.get("model_dim", 4096)
        latent_dim = config.get("latent_dim", 1024)
        num_codebook = config.get("num_codebook", 12)
        num_codes = config.get("num_codes", 1)
        commitment_cost = config.get("commitment_cost", 0.05)
        enc_blocks = config.get("enc_blocks", 6)
        dec_blocks = config.get("dec_blocks", 4)
        num_heads = config.get("num_heads", 8)
        dropout = config.get("dropout", 0.1)
        patch_size = config.get("patch_size", 16)

        self.model = UncontrolledDINOLatentActionModel(
            in_dim=3,
            model_dim=model_dim,
            latent_dim=latent_dim,
            num_latents=num_codebook,
            patch_size=patch_size,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
            dino_version=dino_version,
            dino_model_path=dino_model_path,
            num_codes=num_codes,
            commitment_cost=commitment_cost,
        )
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("model_state_dict", state)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[LAM] load_state_dict warnings: missing={missing}, unexpected={unexpected}")
        self.model.eval().to(device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.device = device

    @torch.no_grad()
    def encode(self, raw_clips: torch.Tensor, prompts: Optional[Iterable[str]] = None) -> torch.Tensor:
        """
        Args:
            raw_clips: uint8 tensor [B, T, C, H, W] in 0-255 range.
            prompts: optional iterable of strings (length B). Uses "none" if omitted.

        Returns:
            Tensor of shape [B, T-1, latent_dim * num_codes].
        """
        if raw_clips.dim() != 5:
            raise ValueError(f"Expected raw clips with shape [B, T, C, H, W], got {raw_clips.shape}")
        videos = raw_clips.to(self.device, dtype=torch.float32) / 255.0
        videos = (videos - self.mean) / self.std
        videos = videos.clamp_(-10.0, 10.0)
        B = videos.size(0)
        if prompts is None:
            prompts = ["none"] * B
        outputs = self.model(videos, prompts=list(prompts))
        z_q = outputs["z_q"]  # [B, T-1, num_codes, latent_dim]
        return z_q.reshape(z_q.size(0), z_q.size(1), -1).detach()


def build_lam_encoder(cfg_model: dict, device: torch.device) -> LamActionEncoder:
    lam_cfg = cfg_model.get("lam")
    if lam_cfg is None:
        raise ValueError("model.use_external_action_tokens=True but model.lam config is missing.")
    return LamActionEncoder(lam_cfg, device)
