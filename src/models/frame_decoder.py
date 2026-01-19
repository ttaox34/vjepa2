# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Frame decoder for visualizing V-JEPA representations.

This follows Appendix B.3 of the paper: a deterministic feedforward network trained
to regress RGB pixels from frozen V-JEPA token representations with an L2 loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Union

import torch
import torch.nn as nn

from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


_Act = Literal["none", "sigmoid", "tanh"]


@dataclass(frozen=True)
class FrameDecoderConfig:
    img_size: Union[int, Tuple[int, int]] = 256
    patch_size: int = 16
    in_dim: int = 1024
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    out_chans: int = 3
    use_sdpa: bool = True
    use_silu: bool = False
    wide_silu: bool = True
    out_act: _Act = "none"


class FrameDecoderViT(nn.Module):
    """Token->RGB deterministic decoder (ViT-style, MAE-like unpatchify head)."""

    def __init__(self, cfg: FrameDecoderConfig):
        super().__init__()
        self.cfg = cfg

        if isinstance(cfg.img_size, int):
            img_h, img_w = cfg.img_size, cfg.img_size
        else:
            img_h, img_w = cfg.img_size

        if img_h % cfg.patch_size != 0 or img_w % cfg.patch_size != 0:
            raise ValueError(f"img_size must be divisible by patch_size, got img_size={cfg.img_size}, {cfg.patch_size=}")

        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.grid_h = self.img_h // cfg.patch_size
        self.grid_w = self.img_w // cfg.patch_size
        self.num_patches = int(self.grid_h * self.grid_w)

        self.in_proj = None
        if cfg.in_dim != cfg.embed_dim:
            self.in_proj = nn.Linear(cfg.in_dim, cfg.embed_dim, bias=True)

        # Fixed 2D sin/cos positional embedding (matches repo ViT conventions).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim), requires_grad=False)
        self._init_pos_embed(self.pos_embed.data)

        dpr = [x.item() for x in torch.linspace(0, 0.0, cfg.depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    use_sdpa=cfg.use_sdpa,
                    use_rope=False,
                    act_layer=nn.SiLU if cfg.use_silu else nn.GELU,
                    wide_silu=cfg.wide_silu,
                    drop_path=dpr[i],
                    grid_size=self.grid_h,
                )
                for i in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

        # Predict per-patch RGB pixels, then unpatchify.
        self.head = nn.Linear(cfg.embed_dim, cfg.patch_size * cfg.patch_size * cfg.out_chans, bias=True)

        self.init_std = 0.02
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed: torch.Tensor) -> None:
        embed_dim = pos_embed.size(-1)
        if self.grid_h != self.grid_w:
            raise ValueError("Only square crops are supported for fixed sin/cos pos embed in this decoder.")
        sincos = get_2d_sincos_pos_embed(embed_dim, self.grid_h, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self) -> None:
        # Same rescaling used in other ViT modules in this repo.
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(float((2.0 * (layer_id + 1)) ** 0.5))
            if hasattr(layer.mlp, "fc2"):
                layer.mlp.fc2.weight.data.div_(float((2.0 * (layer_id + 1)) ** 0.5))
            elif hasattr(layer.mlp, "fc3"):
                layer.mlp.fc3.weight.data.div_(float((2.0 * (layer_id + 1)) ** 0.5))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D_in] patch tokens for a single frame (spatial only).
        Returns:
            rgb: [B, 3, H, W] reconstructed frame in the decoder's output space.
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B,N,D], got {tuple(tokens.shape)}")
        b, n, _ = tokens.shape
        if n != self.num_patches:
            raise ValueError(f"Expected N={self.num_patches} patches, got {n}")

        x = tokens
        if self.in_proj is not None:
            x = self.in_proj(x)

        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        patch_rgb = self.head(x)  # [B, N, p*p*3]
        rgb = self._unpatchify(patch_rgb)

        if self.cfg.out_act == "sigmoid":
            rgb = torch.sigmoid(rgb)
        elif self.cfg.out_act == "tanh":
            rgb = torch.tanh(rgb)
        elif self.cfg.out_act != "none":
            raise ValueError(f"Unknown out_act={self.cfg.out_act}")
        return rgb

    def _unpatchify(self, patch_rgb: torch.Tensor) -> torch.Tensor:
        """Inverse of patchify: [B, N, p*p*C] -> [B, C, H, W]."""
        b, n, dim = patch_rgb.shape
        p = self.cfg.patch_size
        c = self.cfg.out_chans
        if dim != p * p * c:
            raise ValueError(f"Expected last dim {p*p*c}, got {dim}")
        if n != self.num_patches:
            raise ValueError(f"Expected N={self.num_patches}, got {n}")

        x = patch_rgb.reshape(b, self.grid_h, self.grid_w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(b, c, self.grid_h * p, self.grid_w * p)
        return x


def unnormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Inverse of app.vjepa.transforms default normalize; returns pixel-space 0..255 floats."""
    mean = torch.tensor((0.485, 0.456, 0.406), device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1) * 255.0
    std = torch.tensor((0.229, 0.224, 0.225), device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1) * 255.0
    return x * std + mean


def clip_to_unit_rgb(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert normalized model input clip to [0,1] RGB floats."""
    px = unnormalize_imagenet(x)
    return (px / 255.0).clamp(min=0.0 - eps, max=1.0 + eps)
