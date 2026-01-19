# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.utils.data

from app.vjepa_droid.game_dataset import RetroGameDataset

logger = logging.getLogger(__name__)


class RetroFrameDataset(torch.utils.data.Dataset):
    """Return only the transformed clip tensor (C, T, H, W) for frame-decoder training."""

    def __init__(
        self,
        data_paths: Optional[Sequence[str]],
        frames_per_clip: int,
        frame_stride: int,
        transform,
        manifest_paths: Optional[Sequence[str]] = None,
        action_mappings: Optional[Sequence[Optional[str]]] = None,
        action_dim: Optional[int] = None,
        state_keys: Optional[Sequence[str]] = None,
        manifest_cache: bool = False,
        manifest_cache_dir: Optional[str] = None,
    ):
        self.dataset = RetroGameDataset(
            data_paths=data_paths,
            frames_per_clip=frames_per_clip,
            frame_stride=frame_stride,
            transform=transform,
            action_dim=action_dim,
            state_keys=state_keys,
            manifest_paths=manifest_paths,
            action_mappings=action_mappings,
            include_returns=False,
            include_action_latents=False,
            require_action_latents=False,
            return_raw_clips=False,
            manifest_cache=manifest_cache,
            manifest_cache_dir=manifest_cache_dir,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.dataset[index]
        clip = sample[0]  # C T H W
        if not torch.is_tensor(clip):
            clip = torch.as_tensor(clip)
        return clip


def init_retro_frame_data(
    data_paths,
    batch_size: int,
    frames_per_clip: int,
    frame_stride: int,
    transform,
    manifest_paths=None,
    action_mappings=None,
    action_dim=None,
    state_keys=None,
    num_workers: int = 8,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    world_size: int = 1,
    rank: int = 0,
    drop_last: bool = True,
    manifest_cache: bool = False,
    manifest_cache_dir: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.distributed.DistributedSampler]:
    dataset = RetroFrameDataset(
        data_paths=data_paths,
        frames_per_clip=frames_per_clip,
        frame_stride=frame_stride,
        transform=transform,
        manifest_paths=manifest_paths,
        action_mappings=action_mappings,
        action_dim=action_dim,
        state_keys=state_keys,
        manifest_cache=manifest_cache,
        manifest_cache_dir=manifest_cache_dir,
    )
    if len(dataset) == 0:
        raise ValueError("RetroFrameDataset is empty; verify dataset paths and manifest filters.")

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info(
        "RetroFrameDataset initialized with %d clips (frames_per_clip=%d, stride=%d, batch_size=%d)",
        len(dataset),
        frames_per_clip,
        frame_stride,
        batch_size,
    )
    return loader, sampler

