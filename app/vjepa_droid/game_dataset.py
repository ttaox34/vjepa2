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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch.utils.data
from PIL import Image


def _resolve_path(base: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (base / path).resolve()


class ActionMapper:
    def __init__(
        self,
        mapping_per_source: List[List[int]],
        raw_dim: int,
        global_dim: int,
        reduction: str = "and",
    ):
        self.raw_dim = raw_dim
        self.global_dim = global_dim
        self.reduction = reduction.lower()
        if self.reduction not in {"and", "or", "mean"}:
            raise ValueError(f"Unsupported reduction '{reduction}' (choose from 'and', 'or', 'mean')")

        if len(mapping_per_source) != raw_dim:
            raise ValueError("Mapping length does not match raw_dim")

        self.target_groups: List[List[int]] = [[] for _ in range(global_dim)]
        self.source_to_targets: List[List[int]] = [list(targets) for targets in mapping_per_source]
        for src_idx, targets in enumerate(mapping_per_source):
            for tgt in targets:
                if tgt < 0:
                    continue
                if tgt >= global_dim:
                    raise ValueError(f"Target index {tgt} exceeds global_dim {global_dim}")
                self.target_groups[tgt].append(src_idx)

    @staticmethod
    def _normalize_mapping_entry(entry) -> List[int]:
        if entry is None:
            return []
        if isinstance(entry, int):
            return [entry]
        if isinstance(entry, list):
            return [int(v) for v in entry if isinstance(v, (int, np.integer))]
        raise TypeError(f"Unsupported mapping entry type: {type(entry)}")

    @classmethod
    def from_file(cls, path: Path, global_dim: Optional[int]) -> "ActionMapper":
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            mapping_entries = data.get("map_to") or data.get("mapping")
            if mapping_entries is None:
                raise ValueError(f"Mapping file {path} must contain 'map_to' list")
            mapping_per_source = [cls._normalize_mapping_entry(entry) for entry in mapping_entries]
            raw_dim = int(data.get("raw_dim", len(mapping_per_source)))
            reduction = data.get("reduction", "and")
            file_global_dim = data.get("global_dim")
        elif isinstance(data, list):
            mapping_per_source = [cls._normalize_mapping_entry(entry) for entry in data]
            raw_dim = len(mapping_per_source)
            reduction = "and"
            file_global_dim = None
        else:
            raise ValueError(f"Unsupported mapping format in {path}")

        if raw_dim != len(mapping_per_source):
            raise ValueError(f"raw_dim {raw_dim} does not match mapping length {len(mapping_per_source)}")

        global_dim = file_global_dim if file_global_dim is not None else global_dim
        if global_dim is None:
            raise ValueError(f"global_dim must be provided either in config or mapping file {path}")

        return cls(mapping_per_source, raw_dim=raw_dim, global_dim=global_dim, reduction=reduction)

    def map(self, raw_action: np.ndarray) -> np.ndarray:
        if raw_action.size < self.raw_dim:
            raise ValueError(f"Raw action length {raw_action.size} < expected raw_dim {self.raw_dim}")

        mapped = np.zeros((self.global_dim,), dtype=np.float32)
        if self.reduction == "and":
            reducer = np.min
        elif self.reduction == "or":
            reducer = np.max
        else:  # mean
            reducer = np.mean

        for target_idx, sources in enumerate(self.target_groups):
            if not sources:
                continue
            values = raw_action[sources]
            mapped[target_idx] = float(reducer(values))

        return mapped


@dataclass
class _Episode:
    step_files: List[str]
    mapper: Optional[ActionMapper]


class RetroGameDataset(torch.utils.data.Dataset):
    """Retro dataset supporting manifests and per-game action remapping."""

    def __init__(
        self,
        data_paths: Optional[Sequence[str]] = None,
        frames_per_clip: int = 16,
        frame_stride: int = 1,
        transform=None,
        action_dim: Optional[int] = None,
        state_keys: Optional[Sequence[str]] = None,
        default_state_value: float = 0.0,
        manifest_paths: Optional[Sequence[str]] = None,
        action_mappings: Optional[Sequence[Optional[str]]] = None,
        include_returns: bool = False,
    ):
        if frames_per_clip < 2:
            raise ValueError("frames_per_clip must be >= 2 for action-conditioned training.")

        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.frame_stride = max(1, frame_stride)
        self.default_state_value = default_state_value
        self.state_keys = list(state_keys) if state_keys is not None else None
        self.include_returns = include_returns

        self.global_action_dim: Optional[int] = action_dim
        self.episodes: List[_Episode] = []
        self.indices: List[Tuple[int, int]] = []
        self.discounted_returns: Dict[str, float] = {}

        dataset_paths = list(data_paths) if data_paths else []
        manifest_paths = list(manifest_paths) if manifest_paths else []
        sources_count = max(len(dataset_paths), len(manifest_paths))
        if action_mappings is not None and len(action_mappings) not in {0, sources_count}:
            raise ValueError(
                "Length of action_mappings must match the number of datasets/manifest entries (or be omitted)."
            )

        mapper_list: List[Optional[ActionMapper]] = []
        if action_mappings:
            for mapping_path in action_mappings:
                if mapping_path is None:
                    mapper_list.append(None)
                    continue
                mapper = ActionMapper.from_file(Path(mapping_path), global_dim=self.global_action_dim)
                if self.global_action_dim is None:
                    self.global_action_dim = mapper.global_dim
                elif self.global_action_dim != mapper.global_dim:
                    raise ValueError(
                        f"Configured action_dim {self.global_action_dim} does not match mapping global_dim {mapper.global_dim}"
                    )
                mapper_list.append(mapper)
        else:
            mapper_list = [None] * sources_count

        if manifest_paths:
            if dataset_paths and len(dataset_paths) != len(manifest_paths):
                raise ValueError("When both datasets and manifest_paths are provided, they must have the same length.")
            for idx, manifest_path in enumerate(manifest_paths):
                mapper = mapper_list[idx] if mapper_list else None
                self._register_manifest(Path(manifest_path), mapper)
        else:
            if not dataset_paths:
                raise ValueError("Must provide dataset paths when manifest_paths is empty.")
            for idx, raw_path in enumerate(dataset_paths):
                mapper = mapper_list[idx] if mapper_list else None
                for directory in self._expand_data_path(raw_path):
                    self._register_path(directory, mapper)

        if not self.episodes:
            raise ValueError("No valid retro game trajectories found. Check dataset paths and file layout.")

        if self.global_action_dim is None:
            # Fall back to raw dimension inferred from first episode.
            self.global_action_dim = self._infer_raw_dim()

        self.action_dim = self.global_action_dim

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
                rewards: List[float] = []
                returns: List[float] = []

                for i, step_path in enumerate(step_files):
                    data = self._load_step(step_path, episode.mapper)
                    images.append(data["image"])
                    states.append(data["state"])
                    if i < len(step_files) - 1:
                        actions.append(data["action"])
                    rewards.append(data["reward"])
                    returns.append(data.get("discounted_return", 0.0))

                def _stack(values: List[np.ndarray], target_len: int) -> np.ndarray:
                    if values:
                        arr = np.stack(values, axis=0).astype(np.float32)
                    else:
                        arr = np.zeros((0, self.global_action_dim), dtype=np.float32)
                    if arr.shape[0] < target_len:
                        pad = np.zeros((target_len - arr.shape[0], self.global_action_dim), dtype=np.float32)
                        arr = np.concatenate([arr, pad], axis=0)
                    elif arr.shape[0] > target_len:
                        arr = arr[:target_len]
                    return arr

                def _stack_rewards(values: List[float], target_len: int) -> np.ndarray:
                    if values:
                        arr = np.array(values, dtype=np.float32)
                    else:
                        arr = np.zeros((0,), dtype=np.float32)
                    if arr.shape[0] < target_len:
                        pad = np.zeros((target_len - arr.shape[0],), dtype=np.float32)
                        arr = np.concatenate([arr, pad], axis=0)
                    elif arr.shape[0] > target_len:
                        arr = arr[:target_len]
                    return arr

                states_arr = _stack(states, self.frames_per_clip)
                actions_arr = _stack(actions, self.frames_per_clip - 1 if self.frames_per_clip > 0 else 0)
                rewards_arr = _stack_rewards(rewards, self.frames_per_clip)
                returns_arr = _stack_rewards(returns, self.frames_per_clip)
                extrinsics = np.zeros_like(states_arr, dtype=np.float32)
                indices = np.arange(start_idx, start_idx + stride * self.frames_per_clip, stride, dtype=np.int64)

                buffer = np.stack(images, axis=0)
                if self.transform is not None:
                    buffer = self.transform(buffer)

                if self.include_returns:
                    return buffer, actions_arr, states_arr, extrinsics, rewards_arr, returns_arr, indices
                return buffer, actions_arr, states_arr, extrinsics, rewards_arr, indices
            except Exception:
                index = np.random.randint(len(self))
                if attempt == max_retries - 1:
                    raise

    # ------------------------------------------------------------------
    # Episode registration helpers
    # ------------------------------------------------------------------
    def _expand_data_path(self, path: str) -> List[str]:
        candidates: List[str] = []
        if glob.has_magic(path):
            candidates = glob.glob(path)
        else:
            candidates = [path]

        expanded: List[str] = []
        for candidate in candidates:
            candidate_path = Path(candidate)
            if not candidate_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {candidate}")
            if candidate_path.is_dir():
                expanded.append(str(candidate_path.resolve()))
            elif candidate_path.is_file():
                with candidate_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        candidate_dir = line.strip()
                        if candidate_dir:
                            expanded.append(str(Path(candidate_dir).resolve()))
            else:
                raise ValueError(f"Unsupported dataset path: {candidate}")
        return expanded

    def _register_path(self, directory: str, mapper: Optional[ActionMapper]):
        step_files = self._discover_steps(directory)
        if step_files:
            self._add_episode(step_files, mapper)
            return

        for entry in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, entry)
            if not os.path.isdir(full_path):
                continue
            sub_steps = self._discover_steps(full_path)
            if sub_steps:
                self._add_episode(sub_steps, mapper)

    def _register_manifest(self, manifest_path: Path, mapper: Optional[ActionMapper]):
        base = manifest_path.parent
        current: List[str] = []
        last_episode_id = None

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not record.get("keep", True):
                    continue

                json_path = record.get("json_path")
                if json_path is None:
                    continue
                json_path = str(_resolve_path(base, json_path))
                self.discounted_returns[json_path] = float(record.get("discounted_return", 0.0))

                episode_id = record.get("episode_id")
                terminated = bool(record.get("terminated", False))

                if current and episode_id is not None and last_episode_id is not None and episode_id != last_episode_id:
                    self._add_episode(current, mapper)
                    current = []

                if episode_id is not None:
                    last_episode_id = episode_id

                current.append(json_path)
                if terminated:
                    self._add_episode(current, mapper)
                    current = []
                    last_episode_id = None

        if current:
            self._add_episode(current, mapper)

    def _add_episode(self, step_files: List[str], mapper: Optional[ActionMapper]):
        max_start = len(step_files) - (self.frames_per_clip - 1) * self.frame_stride
        if max_start <= 0:
            return
        episode_idx = len(self.episodes)
        self.episodes.append(_Episode(step_files=step_files, mapper=mapper))
        for start in range(0, max_start):
            self.indices.append((episode_idx, start))

    def _discover_steps(self, directory: str) -> List[str]:
        json_files = [os.path.join(directory, entry) for entry in os.listdir(directory) if entry.endswith(".json")]
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

    def _infer_raw_dim(self) -> int:
        for episode in self.episodes:
            mapper = episode.mapper
            if mapper is not None:
                return mapper.global_dim
            for step_path in episode.step_files:
                with open(step_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                action = payload.get("action")
                if isinstance(action, list):
                    return len(action)
        raise ValueError("Unable to infer action dimension from dataset.")

    # ------------------------------------------------------------------
    # Step loading
    # ------------------------------------------------------------------
    def _load_step(self, json_path: str, mapper: Optional[ActionMapper]) -> Dict[str, np.ndarray]:
        json_file = Path(json_path)
        with json_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        image_path = data.get("observation_image_path")
        if image_path is None:
            raise ValueError(f"Missing observation_image_path in {json_path}")
        image_path = _resolve_path(json_file.parent, image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image, dtype=np.uint8)

        raw_action = np.asarray(data.get("action", []), dtype=np.float32)
        reward_val = float(data.get("reward", 0.0))
        if mapper is not None:
            action_vec = mapper.map(raw_action)
            self.global_action_dim = mapper.global_dim
        else:
            if self.global_action_dim is None:
                self.global_action_dim = raw_action.size
            action_vec = np.zeros((self.global_action_dim,), dtype=np.float32)
            limit = min(self.global_action_dim, raw_action.size)
            action_vec[:limit] = raw_action[:limit]

        state_vec = np.zeros((self.global_action_dim,), dtype=np.float32)

        discounted_return = float(self.discounted_returns.get(str(json_file), 0.0))

        return {
            "image": image_array,
            "action": action_vec,
            "state": state_vec,
            "reward": reward_val,
            "discounted_return": discounted_return,
        }
