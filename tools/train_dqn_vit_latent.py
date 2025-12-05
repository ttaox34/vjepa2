#!/usr/bin/env python3
"""
Multi-game Retro DQN where the observation stream is encoded on the fly by a pretrained ViT encoder.
This keeps the DQN hyper-parameters identical to Train_MultiGame_DQN.py, but replaces the visual
input with latent vectors from a ViT-L checkpoint.
"""

import argparse
import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import gymnasium as gym
import numpy as np
import retro
import torch
import torch.nn as nn
import yaml
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.dqn.policies import DQNPolicy

from app.vjepa.utils import init_video_model
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IdentityExtractor(BaseFeaturesExtractor):
    """Features extractor that passes vector observations through unchanged."""

    def __init__(self, observation_space: gym.spaces.Box, input_dim: Optional[int] = None):
        if input_dim is None:
            input_dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim=input_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class ViTLatentEncoder:
    """Utility that loads a ViT checkpoint and exposes a frame->latent helper."""

    def __init__(
        self,
        cfg_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        frames_per_clip: int = 2,
        latent_pool: str = "mean",
    ):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.device = torch.device(device)
        self.frames_per_clip = frames_per_clip
        self.latent_pool = latent_pool

        crop_size = self.cfg["data"]["crop_size"]
        patch_size = self.cfg["data"]["patch_size"]
        tubelet_size = self.cfg["data"]["tubelet_size"]
        self.crop_size = crop_size
        self.tokens_per_frame = int((crop_size // patch_size) ** 2)
        self.mean = (
            torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1, 1)
        )
        self.std = (
            torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1, 1)
        )

        encoder, _ = init_video_model(
            device=self.device,
            patch_size=patch_size,
            max_num_frames=self.frames_per_clip,
            tubelet_size=tubelet_size,
            model_name=self.cfg["model"]["model_name"],
            crop_size=crop_size,
            pred_depth=self.cfg["model"]["pred_depth"],
            pred_num_heads=self.cfg["model"].get("pred_num_heads"),
            pred_embed_dim=self.cfg["model"]["pred_embed_dim"],
            use_sdpa=self.cfg["meta"].get("use_sdpa", False),
            use_silu=self.cfg["model"].get("use_silu", False),
            use_pred_silu=self.cfg["model"].get("use_pred_silu", False),
            wide_silu=self.cfg["model"].get("wide_silu", True),
            use_rope=self.cfg["model"].get("use_rope", True),
            use_activation_checkpointing=False,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"], strict=False)
        encoder.eval()
        if hasattr(encoder, "backbone"):
            self.encoder = encoder.backbone
        else:
            self.encoder = encoder
        self.tubelet_size = tubelet_size
        if hasattr(self.encoder, "embed_dim"):
            self.embed_dim = self.encoder.embed_dim
        else:
            raise AttributeError("Unable to infer encoder embed_dim from the loaded ViT model.")

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """frames: numpy array (T,H,W,C)"""
        tensor = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)
            .to(self.device, dtype=torch.float32)
        )  # [T, C, H, W]
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
        tensor = tensor / 255.0
        tensor = (tensor - self.mean) / self.std
        with torch.no_grad():
            tokens = self.encoder(tensor)  # shape [B, num_patches, embed_dim]
        num_temporal = tokens.shape[1] // self.tokens_per_frame
        tokens = tokens.view(1, num_temporal, self.tokens_per_frame, self.embed_dim)
        target_tokens = tokens[:, -1, :, :]
        if self.latent_pool == "mean":
            latent = target_tokens.mean(dim=1)
        else:
            latent = target_tokens.flatten(1)
        return latent.squeeze(0).detach().cpu().numpy().astype(np.float32)


class ViTLatentWrapper(gym.ObservationWrapper):
    """Wraps an env to replace raw frames with ViT encoder latents."""

    def __init__(
        self,
        env: gym.Env,
        encoder_cfg: Dict[str, Any],
    ):
        super().__init__(env)
        self.encoder = ViTLatentEncoder(**encoder_cfg)
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.encoder.frames_per_clip)
        if encoder_cfg.get("latent_pool", "mean") == "mean":
            dim = self.encoder.embed_dim
        else:
            dim = self.encoder.embed_dim * self.encoder.tokens_per_frame
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(dim,),
            dtype=np.float32,
        )

    def observation(self, obs):
        self.buffer.append(obs)
        while len(self.buffer) < self.encoder.frames_per_clip:
            self.buffer.append(obs)
        frames = np.stack(list(self.buffer), axis=0)
        latent = self.encoder.encode(frames)
        return latent

    def reset(self, **kwargs):
        self.buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self.buffer.append(obs)
        return self.observation(obs), info


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, n: int = 4, stickprob: float = 0.25):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, action):
        terminated = False
        truncated = False
        total_reward = 0.0
        for i in range(self.n):
            if self.curac is None:
                self.curac = action
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = action
            elif i == 1:
                self.curac = action

            if self.supports_want_render and i < self.n - 1:
                obs, reward, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                obs, reward, terminated, truncated, info = self.env.step(self.curac)

            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class MultiBinaryToDiscreteWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.MultiBinary):
            raise TypeError("MultiBinaryToDiscreteWrapper expects a MultiBinary action space.")
        self.original_action_space = env.action_space
        self.action_space = gym.spaces.Discrete(2 ** self.original_action_space.n)

    def step(self, action):
        binary = np.array(
            [(action >> i) & 1 for i in range(self.original_action_space.n)],
            dtype=np.int8,
        )
        return self.env.step(binary)


def make_retro_env(game: str, state: Optional[str], max_episode_steps: int) -> gym.Env:
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game=game, state=state, render_mode="rgb_array")
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if isinstance(env.action_space, gym.spaces.MultiBinary):
        env = MultiBinaryToDiscreteWrapper(env)
    return env


def build_env_fn(
    game: str,
    state: Optional[str],
    seed: int,
    rank: int,
    max_episode_steps: int,
    encoder_cfg: Dict[str, Any],
) -> Callable[[], gym.Env]:
    def _init():
        env = make_retro_env(game=game, state=state, max_episode_steps=max_episode_steps)
        env = ViTLatentWrapper(env, encoder_cfg=encoder_cfg)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vector_envs(
    game: str,
    state: Optional[str],
    seed: int,
    max_episode_steps: int,
    n_envs: int,
    vec_env_type: str,
    encoder_cfg: Dict[str, Any],
):
    env_fns = [
        build_env_fn(game, state, seed, rank, max_episode_steps, encoder_cfg)
        for rank in range(n_envs)
    ]
    if vec_env_type == "subproc":
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    return env


def train_single_game(game: str, args_dict: dict, gpu_queue, job_idx: int):
    gpu_id = gpu_queue.get()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        args = argparse.Namespace(**args_dict)
        local_seed = args.seed + job_idx * 1000
        set_random_seed(local_seed)

        save_dir = os.path.join(args.save_root, game)
        os.makedirs(save_dir, exist_ok=True)
        tensorboard_dir = os.path.join(save_dir, "tensorboard")

        encoder_cfg = {
            "cfg_path": args.vit_config,
            "checkpoint_path": args.vit_checkpoint,
            "device": args.encoder_device,
            "frames_per_clip": args.vit_frames_per_clip,
            "latent_pool": args.vit_latent_pool,
        }
        env = make_vector_envs(
            game=game,
            state=args.state,
            seed=local_seed,
            max_episode_steps=args.max_episode_steps,
            n_envs=args.n_envs,
            vec_env_type=args.vec_env_type,
            encoder_cfg=encoder_cfg,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=save_dir,
            name_prefix=f"dqn_{game}",
            save_replay_buffer=args.save_replay_buffer,
            save_vecnormalize=False,
        )

        obs_dim = int(np.prod(env.observation_space.shape))
        policy_kwargs = dict(
            net_arch=[512, 256],
            features_extractor_class=IdentityExtractor,
            features_extractor_kwargs={"input_dim": obs_dim},
        )

        model = DQN(
            policy=DQNPolicy,
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=1.0,
            gamma=0.99,
            train_freq=(args.train_freq, "step"),
            gradient_steps=args.gradient_steps,
            target_update_interval=args.target_update,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            exploration_fraction=args.exploration_fraction,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_dir,
            device=args.device_type,
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            log_interval=10,
            callback=[checkpoint_callback],
        )

        final_model_path = os.path.join(save_dir, f"dqn_{game}_final.zip")
        model.save(final_model_path)
        env.close()
        torch.cuda.empty_cache()
        return game, final_model_path
    finally:
        gpu_queue.put(gpu_id)


def parse_games(args) -> List[str]:
    games: List[str] = []
    if args.games_file:
        with open(args.games_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(line)
    if args.games:
        games.extend([g.strip() for g in args.games.split(",") if g.strip()])
    if not games:
        raise ValueError("No games specified. Provide --games-file or --games.")
    return games


def parse_gpu_ids(devices: str) -> List[str]:
    ids = [d.strip() for d in devices.split(",") if d.strip()]
    if not ids:
        raise ValueError("No GPU ids detected from --devices.")
    return ids


def main():
    parser = argparse.ArgumentParser(description="Multi-game Retro DQN with ViT latents")
    parser.add_argument("--games-file", type=str, help="Text file with one Retro game name per line.")
    parser.add_argument("--games", type=str, help="Comma-separated list of games.")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=12_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=4_500)
    parser.add_argument("--vec-env-type", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--save-root", type=str, default="multi_game_dqn_vit")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--devices", type=str, default="0", help="Comma-separated GPU ids.")
    parser.add_argument("--procs-per-gpu", type=int, default=1)
    parser.add_argument("--device-type", type=str, default="cuda")
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.18)
    parser.add_argument("--vit-config", type=str, required=True, help="ViT config YAML.")
    parser.add_argument("--vit-checkpoint", type=str, required=True, help="ViT checkpoint path (encoder key expected).")
    parser.add_argument("--vit-frames-per-clip", type=int, default=2)
    parser.add_argument("--vit-latent-pool", choices=["mean", "flatten"], default="mean")
    parser.add_argument("--encoder-device", type=str, default="cpu", help="Device used for ViT inference inside envs.")
    args = parser.parse_args()

    games = parse_games(args)
    gpu_ids = parse_gpu_ids(args.devices)
    os.makedirs(args.save_root, exist_ok=True)

    args_dict = vars(args)
    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu_id in gpu_ids:
        for _ in range(args.procs_per_gpu):
            gpu_queue.put(gpu_id)

    max_workers = len(gpu_ids) * args.procs_per_gpu
    futures = []
    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
        for idx, game in enumerate(games):
            futures.append(executor.submit(train_single_game, game, args_dict, gpu_queue, idx))

        for future in as_completed(futures):
            try:
                game, model_path = future.result()
                print(f"[DONE] {game} -> {model_path}")
            except Exception as exc:
                print(f"[ERROR] Training failed: {exc}")


if __name__ == "__main__":
    main()
