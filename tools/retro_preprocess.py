# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def action_is_zero(action: List[float], epsilon: float = 1e-9) -> bool:
    return all(abs(v) <= epsilon for v in action)


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
    zero_action = action_is_zero(metadata.get("action", []))
    terminated = bool(metadata.get("terminated") or metadata.get("truncated"))

    return {
        "json_path": json_path,
        "mean": mean,
        "std": std,
        "dark_ratio": dark_ratio,
        "zero_action": zero_action,
        "terminated": terminated,
    }


def main():
    args = parse_args()
    roots = [Path(p).expanduser().resolve() for p in args.data_root]
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "frames_seen": 0,
        "frames_kept": 0,
        "frames_dropped_dark": 0,
        "episodes_written": 0,
    }

    keep_map = {} if args.keep_map_output else None

    episode_id = 0
    prev_entry = None
    force_new_episode = False

    def flush_prev(out_file):
        nonlocal prev_entry, stats
        if prev_entry is not None:
            out_file.write(json.dumps(prev_entry) + "\n")
            if prev_entry.get("terminated"):
                stats["episodes_written"] += 1
            prev_entry = None

    with output_path.open("w", encoding="utf-8") as outfile:
        for root in roots:
            directories = find_step_directories(root)
            for directory in directories:
                json_files = list(iter_steps(directory))
                if not json_files:
                    continue
                force_new_episode = False

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

                for result in iter_results():
                    if result is None:
                        continue
                    json_path = result["json_path"]
                    mean = result["mean"]
                    std = result["std"]
                    dark_ratio = result["dark_ratio"]
                    zero_action = result["zero_action"]
                    terminated_flag = result["terminated"]

                    stats["frames_seen"] += 1
                    if stats["frames_seen"] % args.log_every == 0:
                        print(
                            f"[retro_preprocess] processed {stats['frames_seen']} frames "
                            f"(kept={stats['frames_kept']}, dropped_dark={stats['frames_dropped_dark']})"
                        )

                    dark_frame = (mean <= args.dark_mean and std <= args.dark_std) or (dark_ratio >= args.dark_ratio)
                    drop_dark = dark_frame and (args.drop_zero_action_dark or not zero_action)
                    keep_flag = not drop_dark
                    if keep_map is not None:
                        keep_map[str(json_path)] = keep_flag
                    if drop_dark:
                        stats["frames_dropped_dark"] += 1
                        if prev_entry is not None:
                            prev_entry["terminated"] = True
                            flush_prev(outfile)
                        else:
                            force_new_episode = True
                        continue

                    if force_new_episode:
                        episode_id += 1
                        force_new_episode = False

                    entry = {
                        "json_path": str(json_path),
                        "episode_id": episode_id,
                        "keep": True,
                        "terminated": False,
                    }
                    if not args.drop_zero_action_dark:
                        entry["zero_action_dark_allowed"] = zero_action and dark_frame
                    if terminated_flag:
                        entry["terminated"] = True
                        force_new_episode = True

                    if prev_entry is not None:
                        flush_prev(outfile)
                    prev_entry = entry
                    stats["frames_kept"] += 1

                if prev_entry is not None:
                    prev_entry["terminated"] = True
                    flush_prev(outfile)
                    episode_id += 1

        if prev_entry is not None:
            prev_entry["terminated"] = True
            flush_prev(outfile)

    print(
        f"[retro_preprocess] done. seen={stats['frames_seen']} kept={stats['frames_kept']} "
        f"dropped_dark={stats['frames_dropped_dark']} episodes={stats['episodes_written']}"
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


if __name__ == "__main__":
    main()
