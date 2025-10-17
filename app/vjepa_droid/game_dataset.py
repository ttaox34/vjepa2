# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch.utils.data
from PIL import Image


@dataclass
class _Episode:
    root: str
    step_files: List[str]


class RetroGameDataset(torch.utils.data.Dataset):
    """
    Dataset that loads frame/action/state tuples stored as {step_XXXXX.json, step_XXXXX.png}
    inside per-trajectory folders.
    """

    def __init__(
        self,
        data_paths: Sequence[str],
        frames_per_clip: int = 16,
        frame_stride: int = 1,
        transform=None,
        action_dim: Optional[int] = None,
        state_keys: Optional[Sequence[str]] = None,
        default_state_value: float = 0.0,
    ):
        if frames_per_clip < 2:
            raise ValueError("frames_per_clip must be >= 2 for action-conditioned training.")
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.frame_stride = max(1, frame_stride)
        self.default_state_value = default_state_value
        self.state_keys = list(state_keys) if state_keys is not None else None
        self.episodes: List[_Episode] = []
        self.indices: List[Tuple[int, int]] = []
        normalized_paths = self._expand_data_paths(data_paths)
        for path in normalized_paths:
            self._register_path(path)
        if not self.episodes:
            raise ValueError("No valid retro game trajectories found. Check dataset paths and file layout.")
        self.action_dim = action_dim or self._infer_action_dim()
        if self.action_dim is None:
            raise ValueError("Unable to infer action dimension from dataset; please set action_dim explicitly.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                episode_idx, start_idx = self.indices[index]
                episode = self.episodes[episode_idx]
                stride = self.frame_stride
                step_slice = slice(start_idx, start_idx + stride * self.frames_per_clip, stride)
                step_files = episode.step_files[step_slice]
                images: List[np.ndarray] = []
                states: List[np.ndarray] = []
                actions: List[np.ndarray] = []
                for i, step_path in enumerate(step_files):
                    data = self._load_step(step_path, episode.root)
                    images.append(data["image"])
                    states.append(data["state"])
                    if i < len(step_files) - 1:
                        actions.append(data["action"])
                def _stack_with_pad(values, target_len, pad_value=0.0):
                    if values:
                        arr = np.stack(values, axis=0).astype(np.float32)
                    else:
                        arr = np.zeros((0, self.action_dim), dtype=np.float32)
                    if arr.shape[0] < target_len:
                        pad = np.full(
                            (target_len - arr.shape[0], arr.shape[1] if arr.ndim > 1 else 1),
                            pad_value,
                            dtype=np.float32,
                        )
                        arr = np.concatenate([arr, pad], axis=0)
                    elif arr.shape[0] > target_len:
                        arr = arr[:target_len]
                    return arr

                states_arr = _stack_with_pad(states, self.frames_per_clip, pad_value=self.default_state_value)
                actions_arr = _stack_with_pad(
                    actions, self.frames_per_clip - 1 if self.frames_per_clip > 0 else 0, pad_value=0.0
                )
                extrinsics = np.zeros_like(states_arr, dtype=np.float32)
                indices = np.arange(start_idx, start_idx + stride * self.frames_per_clip, stride, dtype=np.int64)

                buffer = np.stack(images, axis=0)
                if self.transform is not None:
                    buffer = self.transform(buffer)

                return buffer, actions_arr, states_arr, extrinsics, indices
            except Exception:
                # Randomly resample if a window fails to load.
                index = np.random.randint(len(self))
                if attempt == max_retries - 1:
                    raise

    def _expand_data_paths(self, data_paths: Sequence[str]) -> List[str]:
        expanded: List[str] = []
        for path in data_paths:
            candidates = []
            if glob.has_magic(path):
                candidates = glob.glob(path)
            else:
                candidates = [path]
            if not candidates:
                raise FileNotFoundError(f"Dataset path does not exist: {path}")
            for candidate in candidates:
                if not os.path.exists(candidate):
                    raise FileNotFoundError(f"Dataset path does not exist: {candidate}")
                path = candidate
                if os.path.isdir(path):
                    expanded.append(path)
                elif os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as handle:
                        for line in handle:
                            candidate = line.strip()
                            if candidate:
                                expanded.append(candidate)
                else:
                    raise ValueError(f"Unsupported dataset path: {path}")
        return expanded

    def _register_path(self, directory: str):
        # If the directory contains subdirectories with steps, register each subdirectory.
        step_files = self._discover_steps(directory)
        if step_files:
            self._add_episode(directory, step_files)
            return

        for entry in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, entry)
            if not os.path.isdir(full_path):
                continue
            sub_steps = self._discover_steps(full_path)
            if sub_steps:
                self._add_episode(full_path, sub_steps)

    def _discover_steps(self, directory: str) -> List[str]:
        json_files = []
        for entry in os.listdir(directory):
            if entry.endswith(".json"):
                json_files.append(os.path.join(directory, entry))
        if not json_files:
            return []
        json_files.sort(key=self._extract_step_index)
        if len(json_files) < self.frames_per_clip * self.frame_stride:
            return []
        return json_files

    @staticmethod
    def _extract_step_index(path: str) -> int:
        match = re.search(r"(\d+)(?=\.json$)", os.path.basename(path))
        return int(match.group(0)) if match else 0

    def _add_episode(self, root: str, step_files: List[str]):
        max_start = len(step_files) - (self.frames_per_clip - 1) * self.frame_stride
        if max_start <= 0:
            return
        episode_idx = len(self.episodes)
        self.episodes.append(_Episode(root=root, step_files=step_files))
        for start in range(0, max_start):
            self.indices.append((episode_idx, start))

    def _infer_action_dim(self) -> Optional[int]:
        for episode in self.episodes:
            for step_path in episode.step_files:
                try:
                    with open(step_path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    action = payload.get("action")
                    if isinstance(action, list):
                        return len(action)
                except Exception:
                    continue
        return None

    def _ensure_state_keys(self, info_dict: Dict[str, float]) -> List[str]:
        if self.state_keys is None:
            self.state_keys = sorted(info_dict.keys())
        return self.state_keys

    def _load_step(self, json_path: str, root: str) -> Dict[str, np.ndarray]:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        img_rel_path = data.get("observation_image_path")
        if img_rel_path is None:
            raise ValueError(f"Missing observation_image_path in {json_path}")
        image_path = os.path.join(root, img_rel_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image, dtype=np.uint8)

        action = data.get("action")
        action_vec = self._pad_vector(action, self.action_dim)

        info = data.get("info", {})
        keys = self._ensure_state_keys(info)
        state_vec = np.array([info.get(k, self.default_state_value) for k in keys], dtype=np.float32)
        state_vec = self._pad_vector(state_vec, self.action_dim)

        return {
            "image": image_array,
            "action": action_vec,
            "state": state_vec,
        }

    def _pad_vector(self, values, target_dim: int) -> np.ndarray:
        if values is None:
            values = []
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size > target_dim:
            arr = arr[:target_dim]
        if arr.size < target_dim:
            padded = np.full((target_dim,), self.default_state_value, dtype=np.float32)
            padded[: arr.size] = arr
            arr = padded
        return arr.astype(np.float32)
