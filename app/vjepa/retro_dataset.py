# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Optional, Sequence

import numpy as np
import torch
import torch.utils.data

from app.vjepa_droid.game_dataset import RetroGameDataset

logger = logging.getLogger(__name__)


class RetroVisionDataset(torch.utils.data.Dataset):
    """
    Thin adapter that exposes RetroGameDataset samples using the tuple format expected by
    the generic JEPA data pipeline (list of clips, dummy label, clip indices).
    """

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        clip = sample[0]  # C T H W tensor produced by RetroGameDataset transforms
        indices = sample[-1]
        if isinstance(indices, torch.Tensor):
            clip_indices = indices.to(torch.long)
        else:
            clip_indices = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        return [clip], 0, [clip_indices]


def init_retro_data(
    data_paths,
    batch_size,
    frames_per_clip,
    frame_stride,
    transform,
    collator,
    manifest_paths=None,
    action_mappings=None,
    action_dim=None,
    state_keys=None,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    world_size=1,
    rank=0,
    drop_last=True,
    manifest_cache=False,
    manifest_cache_dir=None,
):
    dataset = RetroVisionDataset(
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
    dataset_len = len(dataset)
    if dataset_len == 0:
        raise ValueError("RetroVisionDataset is empty; verify dataset paths and manifest filters.")

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info(
        "RetroVisionDataset initialized with %d clips (frames_per_clip=%d, stride=%d, batch_size=%d)",
        dataset_len,
        frames_per_clip,
        frame_stride,
        batch_size,
    )
    return data_loader, dist_sampler
