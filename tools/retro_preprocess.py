# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from app.vjepa_droid.game_dataset import ActionMapper

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            pass

        def close(self):
            pass


def find_step_directories(root: Path) -> List[Path]:
    dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(name.endswith(".json") for name in filenames):
            dirs.append(Path(dirpath))
    return sorted(dirs)


def iter_steps(directory: Path) -> Iterable[Path]:
    json_files = [p for p in directory.iterdir() if p.suffix == ".json" and p.name.startswith("step_")]
    json_files.sort(key=lambda p: int(p.stem.split("_")[-1]) if "_" in p.stem else p.stem)
    return json_files


def compute_metrics(image_path: Path, downsample: int, dark_pixel_threshold: float) -> Tuple[float, float, float]:
    img = Image.open(image_path).convert("RGB")
    if downsample > 0:
        img.thumbnail((downsample, downsample))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    mean = float(gray.mean() * 255.0)
    std = float(gray.std() * 255.0)
    dark_ratio = float((gray < dark_pixel_threshold / 255.0).mean())
    return mean, std, dark_ratio


def action_is_zero(action: Sequence[float], epsilon: float = 1e-9) -> bool:
    return all(abs(float(v)) <= epsilon for v in action)


def parse_args():
    parser = argparse.ArgumentParser(description="Retro dataset preprocessing tool.")
    parser.add_argument("--data-root", nargs="+", required=True, help="Directories containing step_*.json files.")
    parser.add_argument("--output", required=True, help="Path to manifest JSONL to write.")
    parser.add_argument("--dark-mean", type=float, default=35.0, help="Mean intensity threshold (0-255).")
    parser.add_argument("--dark-std", type=float, default=12.0, help="Std threshold (0-255).")
    parser.add_argument("--dark-ratio", type=float, default=0.985, help="Fraction of dark pixels to flag frame.")
    parser.add_argument("--dark-pixel-threshold", type=float, default=20.0, help="Pixel value considered dark (0-255).")
    parser.add_argument("--downsample", type=int, default=64, help="Thumbnail size for metric computation.")
    parser.add_argument("--drop-zero-action-dark", action="store_true", help="If set, drop dark frames even when action is zero.")
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--stats-output", type=str, default=None, help="Optional JSON file for summary stats.")
    parser.add_argument("--num-workers", type=int, default=1, help="Thread pool size for image/JSON decoding.")
    parser.add_argument(
        "--keep-map-output",
        type=str,
        default=None,
        help="Optional JSON file to store {json_path: keep_bool} mapping.",
    )
    parser.add_argument(
        "--action-mapping",
        type=str,
        default=None,
        help="Optional action mapping JSON (raw->global) to drop frames that use ignored buttons.",
    )
    parser.add_argument(
        "--disable-action-filter",
        action="store_true",
        help="Skip dropping frames based on action mapping even if one is provided.",
    )
    parser.add_argument(
        "--none-index",
        type=int,
        default=0,
        help="Global index representing a 'none' action in the mapping. Raw buttons mapped only here trigger drops when active.",
    )
    parser.add_argument(
        "--action-epsilon",
        type=float,
        default=1e-6,
        help="Tolerance when deciding if a raw action component should be considered non-zero.",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="Discount factor used when precomputing episode returns.",
    )
    parser.add_argument(
        "--value-map-output",
        type=str,
        default=None,
        help="Optional JSON file to store per-frame discounted returns {json_path: value}.",
    )
    parser.add_argument(
        "--min-episode-length",
        type=int,
        default=0,
        help="Minimum number of kept frames required to keep an episode. Shorter segments are discarded.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars during preprocessing.",
    )
    return parser.parse_args()


def process_step(
    json_path: Path,
    downsample: int,
    dark_pixel_threshold: float,
) -> Optional[dict]:
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        print(f"[retro_preprocess] failed to load {json_path}: {exc}")
        return None

    png_rel = metadata.get("observation_image_path")
    if png_rel is None:
        return None
    image_path = Path(png_rel)
    if not image_path.is_absolute():
        image_path = json_path.parent / png_rel
    if not image_path.exists():
        print(f"[retro_preprocess] missing image {image_path}")
        return None

    mean, std, dark_ratio = compute_metrics(image_path, downsample, dark_pixel_threshold)
    raw_action = metadata.get("action", [])
    zero_action = action_is_zero(raw_action)
    terminated = bool(metadata.get("terminated") or metadata.get("truncated"))
    reward = float(metadata.get("reward", 0.0))

    return {
        "json_path": json_path,
        "mean": mean,
        "std": std,
        "dark_ratio": dark_ratio,
        "zero_action": zero_action,
        "terminated": terminated,
        "raw_action": raw_action,
        "reward": reward,
    }


def build_drop_indices(mapper: ActionMapper, none_index: int) -> Tuple[int, ...]:
    drop_list: List[int] = []
    for raw_idx, targets in enumerate(mapper.source_to_targets):
        valid_targets = [t for t in targets if t >= 0]
        if not valid_targets or all(t == none_index for t in valid_targets):
            drop_list.append(raw_idx)
    return tuple(drop_list)


def main():
    args = parse_args()
    roots = [Path(p).expanduser().resolve() for p in args.data_root]
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapper: Optional[ActionMapper] = None
    drop_indices: Tuple[int, ...] = ()
    if args.action_mapping:
        mapping_path = Path(args.action_mapping).expanduser().resolve()
        mapper = ActionMapper.from_file(mapping_path, global_dim=None)
        drop_indices = build_drop_indices(mapper, args.none_index)
        print(
            f"[retro_preprocess] action mapping loaded ({mapper.raw_dim}->{mapper.global_dim}); "
            f"{len(drop_indices)} raw indices flagged as drop-if-active."
        )
        if args.disable_action_filter:
            drop_indices = ()
            print("[retro_preprocess] action filtering disabled; no frames will be dropped due to action mapping.")

    stats: Dict[str, int] = {
        "frames_seen": 0,
        "frames_kept": 0,
        "frames_dropped_dark": 0,
        "frames_dropped_action": 0,
        "episodes_written": 0,
        "episodes_dropped_short": 0,
    }

    keep_map = {} if args.keep_map_output else None
    value_map = {} if args.value_map_output else None

    episode_id = 0
    true_episode_buffer: List[Dict[str, object]] = []

    def mark_keep(path_str: str, keep: bool):
        if keep_map is not None:
            keep_map[path_str] = keep

    with output_path.open("w", encoding="utf-8") as outfile:
        for root in roots:
            directories = find_step_directories(root)
            for directory in directories:
                json_files = list(iter_steps(directory))
                if not json_files:
                    continue

                true_episode_buffer = []
                current_segment: List[dict] = []
                pending_segments: List[List[dict]] = []

                def close_current_segment(force_terminate_last: bool = False):
                    nonlocal current_segment, pending_segments
                    if not current_segment:
                        return
                    if force_terminate_last and not current_segment[-1]["terminated"]:
                        current_segment[-1]["terminated"] = True
                        current_segment[-1]["forced_termination"] = True
                    pending_segments.append(current_segment)
                    current_segment = []

                def finalize_true_episode():
                    nonlocal pending_segments, true_episode_buffer, episode_id
                    if not pending_segments:
                        true_episode_buffer = []
                        return

                    returns_map: Dict[int, float] = {}
                    future = 0.0
                    for record in reversed(true_episode_buffer):
                        reward_val = float(record["reward"])
                        future = reward_val + args.discount * future
                        entry = record.get("entry")
                        if entry is not None:
                            returns_map[id(entry)] = future

                    for segment in pending_segments:
                        episode_length = len(segment)
                        if episode_length < args.min_episode_length:
                            stats["episodes_dropped_short"] += 1
                            for entry in segment:
                                mark_keep(entry["json_path"], False)
                            continue

                        if not segment[-1]["terminated"]:
                            segment[-1]["terminated"] = True

                        for entry in segment:
                            entry["episode_id"] = episode_id
                            value = float(returns_map.get(id(entry), 0.0))
                            entry["discounted_return"] = value
                            if value_map is not None:
                                value_map[entry["json_path"]] = value
                            outfile.write(json.dumps(entry) + "\n")
                            mark_keep(entry["json_path"], True)
                        stats["frames_kept"] += episode_length
                        stats["episodes_written"] += 1
                        episode_id += 1

                    pending_segments = []
                    true_episode_buffer = []

                def iter_results():
                    if args.num_workers and args.num_workers > 1:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                            yield from executor.map(
                                lambda p: process_step(p, args.downsample, args.dark_pixel_threshold),
                                json_files,
                            )
                    else:
                        for json_path in json_files:
                            yield process_step(json_path, args.downsample, args.dark_pixel_threshold)

                with tqdm(
                    total=len(json_files),
                    desc=str(directory),
                    disable=args.no_progress or len(json_files) == 0,
                ) as progress:
                    for result in iter_results():
                        progress.update(1)
                        if result is None:
                            continue
                        json_path = result["json_path"]
                        mean = result["mean"]
                        std = result["std"]
                        dark_ratio = result["dark_ratio"]
                        zero_action = result["zero_action"]
                        terminated_flag = result["terminated"]
                        raw_action = result["raw_action"]
                        reward = float(result["reward"])

                        stats["frames_seen"] += 1
                        # if stats["frames_seen"] % args.log_every == 0:
                        #     print(
                        #         f"[retro_preprocess] processed {stats['frames_seen']} frames "
                        #         f"(kept={stats['frames_kept']}, dropped_dark={stats['frames_dropped_dark']}, "
                        #         f"dropped_action={stats['frames_dropped_action']})"
                        #     )

                        json_path_str = str(json_path)
                        dark_frame = (mean <= args.dark_mean and std <= args.dark_std) or (dark_ratio >= args.dark_ratio)
                        drop_dark = dark_frame and (args.drop_zero_action_dark or not zero_action)
                        drop_action = False
                        if not drop_dark and drop_indices:
                            values = list(raw_action) if isinstance(raw_action, list) else []
                            for idx in drop_indices:
                                if idx < len(values) and abs(float(values[idx])) > args.action_epsilon:
                                    drop_action = True
                                    break

                        if drop_dark or drop_action:
                            mark_keep(json_path_str, False)
                            if drop_dark:
                                stats["frames_dropped_dark"] += 1
                            if drop_action:
                                stats["frames_dropped_action"] += 1
                            if current_segment:
                                close_current_segment(force_terminate_last=True)
                            true_episode_buffer.append({"json_path": json_path_str, "reward": reward, "entry": None})
                            if drop_dark:
                                finalize_true_episode()
                            continue

                        entry = {
                            "json_path": json_path_str,
                            "episode_id": None,
                            "keep": True,
                            "terminated": bool(terminated_flag),
                            "forced_termination": False,
                        }
                        if not args.drop_zero_action_dark:
                            entry["zero_action_dark_allowed"] = bool(zero_action and dark_frame)

                        current_segment.append(entry)
                        true_episode_buffer.append({"json_path": json_path_str, "reward": reward, "entry": entry})

                        if entry["terminated"]:
                            close_current_segment()
                            finalize_true_episode()

                close_current_segment(force_terminate_last=True)
                finalize_true_episode()

    print(
        f"[retro_preprocess] done. seen={stats['frames_seen']} kept={stats['frames_kept']} "
        f"dropped_dark={stats['frames_dropped_dark']} dropped_action={stats['frames_dropped_action']} "
        f"episodes={stats['episodes_written']} dropped_short={stats['episodes_dropped_short']}"
    )
    if args.stats_output:
        stats_path = Path(args.stats_output).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    if keep_map is not None:
        keep_path = Path(args.keep_map_output).expanduser().resolve()
        keep_path.parent.mkdir(parents=True, exist_ok=True)
        with keep_path.open("w", encoding="utf-8") as f:
            json.dump(keep_map, f)

    if value_map is not None:
        value_path = Path(args.value_map_output).expanduser().resolve()
        value_path.parent.mkdir(parents=True, exist_ok=True)
        with value_path.open("w", encoding="utf-8") as f:
            json.dump(value_map, f)


if __name__ == "__main__":
    main()
