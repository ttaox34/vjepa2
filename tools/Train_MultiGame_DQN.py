#!/usr/bin/env python3
"""
Train many Retro games with SB3 DQN in parallel, automatically distributing
jobs across multiple GPUs. Each game gets its own model/checkpoints.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from typing import Any, Callable, List, Optional

import gymnasium as gym
import numpy as np
import retro
import torch as th
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.dqn.policies import DQNPolicy

# Placeholder list – fill with the 200 Retro game names before launching training.
GAMES_TO_TRAIN: List[str] = [
    # "Game1",
    # "Game2",
]


class DuelingQNetwork(BasePolicy):
    """Q-network with dueling architecture."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            net_arch = [256, 256]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)
        self.value_net = nn.Sequential(
            *create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        )
        self.advantage_net = nn.Sequential(
            *create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        )

    def forward(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        value = self.value_net(features)
        advantage = self.advantage_net(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def _predict(self, observation, deterministic: bool = True):
        q_values = self(observation)
        return q_values.argmax(dim=1).reshape(-1)


class DuelingCnnPolicy(DQNPolicy):
    """CnnPolicy variant that builds a dueling Q-network."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.spaces.Discrete,
        lr_schedule,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


class StochasticFrameSkip(gym.Wrapper):
    """Randomized frame skip similar to DeepMind settings."""

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
    """Convert MultiBinary actions (e.g., up to 12 buttons) into a Discrete space."""

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


def make_retro_env(
    *,
    game: str,
    state: Optional[str],
    max_episode_steps: int,
) -> gym.Env:
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game=game, state=state, render_mode="rgb_array")
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if isinstance(env.action_space, gym.spaces.MultiBinary):
        env = MultiBinaryToDiscreteWrapper(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def build_env_fn(
    game: str,
    state: Optional[str],
    seed: int,
    rank: int,
    max_episode_steps: int,
) -> Callable[[], gym.Env]:
    def _init():
        env = make_retro_env(
            game=game,
            state=state,
            max_episode_steps=max_episode_steps,
        )
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
    frame_stack: int,
    vec_env_type: str,
):
    env_fns = [
        build_env_fn(game, state, seed, rank, max_episode_steps) for rank in range(n_envs)
    ]
    if vec_env_type == "subproc":
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=frame_stack)
    env = VecTransposeImage(env)
    return env


def train_single_game(game: str, args_dict: dict, gpu_queue, job_idx: int):
    gpu_id = gpu_queue.get()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        import torch  # noqa: WPS433

        args = argparse.Namespace(**args_dict)
        local_seed = args.seed + job_idx * 1000
        set_random_seed(local_seed)

        save_dir = os.path.join(args.save_root, game)
        os.makedirs(save_dir, exist_ok=True)
        tensorboard_dir = os.path.join(save_dir, "tensorboard")

        env = make_vector_envs(
            game=game,
            state=args.state,
            seed=local_seed,
            max_episode_steps=args.max_episode_steps,
            n_envs=args.n_envs,
            frame_stack=args.frame_stack,
            vec_env_type=args.vec_env_type,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=save_dir,
            name_prefix=f"dqn_{game}",
            save_replay_buffer=args.save_replay_buffer,
            save_vecnormalize=False,
        )

        model = DQN(
            policy=DuelingCnnPolicy,
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
            policy_kwargs=dict(
                net_arch=[512, 256],
            ),
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
        games = list(GAMES_TO_TRAIN)
    if not games:
        raise ValueError("No games specified. Provide --games-file, --games, or populate GAMES_TO_TRAIN.")
    return games


def parse_gpu_ids(devices: str) -> List[str]:
    ids = [d.strip() for d in devices.split(",") if d.strip()]
    if not ids:
        raise ValueError("No GPU ids detected from --devices.")
    return ids


def main():
    parser = argparse.ArgumentParser(description="Multi-game Retro DQN trainer")
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
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=4_500)
    parser.add_argument("--vec-env-type", choices=["subproc", "dummy"], default="subproc", help="Use DummyVecEnv to reduce OS processes.")
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--save-replay-buffer", action="store_true", help="Store replay buffers with checkpoints.")
    parser.add_argument("--save-root", type=str, default="multi_game_dqn_models")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--devices", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids.")
    parser.add_argument("--procs-per-gpu", type=int, default=1, help="Concurrent processes allowed per GPU.")
    parser.add_argument("--device-type", type=str, default="cuda", help="SB3 device string ('cuda' or 'auto').")
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.18)
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
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, game in enumerate(games):
            futures.append(
                executor.submit(train_single_game, game, args_dict, gpu_queue, idx)
            )

        for future in as_completed(futures):
            try:
                game, model_path = future.result()
                print(f"[DONE] {game} -> {model_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Training failed: {exc}")


if __name__ == "__main__":
    main()
