# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pprint
from pathlib import Path

import torch
import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger

logger = get_logger(force=True)

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, required=True, help="Path to the training config YAML.")
parser.add_argument(
    "--folder",
    type=str,
    default=None,
    help="Optional override for the `folder` field in the YAML (handy for sweeps).",
)


def main():
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    with open(args.fname, "r") as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)

    if args.folder is not None:
        params["folder"] = args.folder

    folder = Path(params["folder"])
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        params_path = folder / "params-pretrain.yaml"
        with params_path.open("w") as f:
            yaml.dump(params, f)
        pprint.PrettyPrinter(indent=4).pprint(params)

    init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running on rank {rank} / {world_size} (local_rank={local_rank})")

    app_main(params["app"], args=params)


if __name__ == "__main__":
    main()
