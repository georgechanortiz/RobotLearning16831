# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a model-based controller on the Go2 walking task."""

import argparse
import csv
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an MPC-style MBRL agent on an Isaac Lab task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Flat-Unitree-Go2-train-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for training.")
parser.add_argument("--buffer_capacity", type=int, default=300000, help="Replay buffer capacity.")
parser.add_argument("--seed_steps", type=int, default=25, help="Initial random environment steps before planning.")
parser.add_argument("--train_steps", type=int, default=400, help="Number of environment interaction steps.")
parser.add_argument("--updates_per_step", type=int, default=8, help="Model updates after each environment step.")
parser.add_argument("--batch_size", type=int, default=4096, help="Replay batch size.")
parser.add_argument("--hidden_dim", type=int, default=512, help="Dynamics ensemble hidden dimension.")
parser.add_argument("--model_depth", type=int, default=3, help="Number of hidden layers per ensemble member.")
parser.add_argument("--ensemble_size", type=int, default=5, help="Number of dynamics models in the ensemble.")
parser.add_argument("--horizon", type=int, default=10, help="Planning horizon in environment steps.")
parser.add_argument("--candidates", type=int, default=256, help="Candidate action sequences for CEM.")
parser.add_argument("--elites", type=int, default=32, help="Elite sequences kept each CEM iteration.")
parser.add_argument("--cem_iterations", type=int, default=4, help="CEM refinement iterations.")
parser.add_argument("--discount", type=float, default=0.99, help="Planning discount factor.")
parser.add_argument("--lr", type=float, default=3e-4, help="Dynamics model learning rate.")
parser.add_argument("--eval_interval", type=int, default=10, help="Steps between console/log summaries.")
parser.add_argument("--save_interval", type=int, default=50, help="Steps between checkpoints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import FP16831.tasks  # noqa: F401
from FP16831.mbrl import CEMPlanner, DynamicsEnsemble, ReplayBuffer


@dataclass
class TrainState:
    env_steps: int = 0
    gradient_updates: int = 0
    episodes_finished: int = 0
    best_mean_return: float = float("-inf")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_obs(obs: object, device: torch.device) -> torch.Tensor:
    if isinstance(obs, dict):
        if "policy" in obs:
            obs = obs["policy"]
        elif "obs" in obs and isinstance(obs["obs"], dict) and "policy" in obs["obs"]:
            obs = obs["obs"]["policy"]
        else:
            raise KeyError(f"Unsupported observation dictionary keys: {list(obs.keys())}")
    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return obs.float().view(obs.shape[0], -1)


def to_tensor(x: object, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def get_action_bounds(action_space: gym.Space, device: torch.device, action_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return -torch.ones(action_dim, device=device), torch.ones(action_dim, device=device)

    low_t = torch.as_tensor(low, dtype=torch.float32, device=device).view(-1)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device).view(-1)
    low_t = torch.where(torch.isfinite(low_t), low_t, -torch.ones_like(low_t))
    high_t = torch.where(torch.isfinite(high_t), high_t, torch.ones_like(high_t))
    return low_t, high_t


def random_actions(batch_size: int, action_low: torch.Tensor, action_high: torch.Tensor) -> torch.Tensor:
    return action_low + torch.rand((batch_size, action_low.numel()), device=action_low.device) * (action_high - action_low)


def make_log_dir() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join("logs", "mbrl", f"go2_walk_{timestamp}"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    return log_dir


def append_metrics(csv_path: str, row: dict[str, float | int]) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    set_seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = torch.device(env.unwrapped.device)
    log_dir = make_log_dir()
    metrics_path = os.path.join(log_dir, "metrics.csv")

    obs, _ = env.reset()
    obs = flatten_obs(obs, device)
    num_envs = obs.shape[0]
    action_dim = int(np.prod(env.action_space.shape))
    obs_dim = obs.shape[-1]

    action_low, action_high = get_action_bounds(env.action_space, device, action_dim)

    replay = ReplayBuffer(args_cli.buffer_capacity, obs_dim=obs_dim, action_dim=action_dim)
    model = DynamicsEnsemble(
        obs_dim=obs_dim,
        action_dim=action_dim,
        ensemble_size=args_cli.ensemble_size,
        hidden_dim=args_cli.hidden_dim,
        depth=args_cli.model_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_cli.lr)
    model.eval()
    planner = CEMPlanner(
        model=model,
        action_low=action_low,
        action_high=action_high,
        horizon=args_cli.horizon,
        candidates=args_cli.candidates,
        elites=args_cli.elites,
        iterations=args_cli.cem_iterations,
        discount=args_cli.discount,
    )

    train_state = TrainState()
    episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.float32, device=device)
    recent_returns: list[float] = []
    recent_lengths: list[float] = []
    latest_losses = {"loss": 0.0, "delta_loss": 0.0, "reward_loss": 0.0, "continue_loss": 0.0}

    with open(os.path.join(log_dir, "config.txt"), "w", encoding="utf-8") as f:
        for key, value in sorted(vars(args_cli).items()):
            f.write(f"{key}: {value}\n")

    while simulation_app.is_running() and train_state.env_steps < args_cli.train_steps:
        with torch.inference_mode():
            if train_state.env_steps < args_cli.seed_steps or len(replay) < args_cli.batch_size:
                actions = random_actions(obs.shape[0], action_low, action_high)
            else:
                actions = planner.plan(obs)

            next_obs_raw, rewards, terminated, truncated, _ = env.step(actions)
            next_obs = flatten_obs(next_obs_raw, device)
            rewards = to_tensor(rewards, device).float().view(-1, 1)
            terminated = to_tensor(terminated, device).bool().view(-1, 1)
            truncated = to_tensor(truncated, device).bool().view(-1, 1)
            done = terminated | truncated
            continues = (~done).float()

            replay.add_batch(
                obs.detach().cpu(),
                actions.detach().cpu(),
                rewards.detach().cpu(),
                next_obs.detach().cpu(),
                continues.detach().cpu(),
            )

            episode_returns += rewards.squeeze(-1)
            episode_lengths += 1
            if done.any():
                done_mask = done.squeeze(-1)
                recent_returns.extend(episode_returns[done_mask].detach().cpu().tolist())
                recent_lengths.extend(episode_lengths[done_mask].detach().cpu().tolist())
                train_state.episodes_finished += int(done_mask.sum().item())
                episode_returns[done_mask] = 0.0
                episode_lengths[done_mask] = 0.0

            obs = next_obs
            train_state.env_steps += 1

        if len(replay) >= args_cli.batch_size:
            model.train()
            for _ in range(args_cli.updates_per_step):
                batch = replay.sample(args_cli.batch_size, device=device)
                loss, metrics = model.loss(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                latest_losses = metrics
                train_state.gradient_updates += 1
            model.eval()

        if train_state.env_steps % args_cli.eval_interval == 0:
            mean_return = float(np.mean(recent_returns[-100:])) if recent_returns else 0.0
            mean_length = float(np.mean(recent_lengths[-100:])) if recent_lengths else 0.0
            train_state.best_mean_return = max(train_state.best_mean_return, mean_return)
            row = {
                "env_steps": train_state.env_steps,
                "gradient_updates": train_state.gradient_updates,
                "episodes_finished": train_state.episodes_finished,
                "buffer_size": len(replay),
                "mean_return_100": mean_return,
                "mean_length_100": mean_length,
                "best_mean_return": train_state.best_mean_return,
                **latest_losses,
            }
            append_metrics(metrics_path, row)
            print(
                "[MBRL] "
                f"step={train_state.env_steps} "
                f"buffer={len(replay)} "
                f"episodes={train_state.episodes_finished} "
                f"return100={mean_return:.3f} "
                f"len100={mean_length:.2f} "
                f"loss={latest_losses['loss']:.4f}"
            )

        if train_state.env_steps % args_cli.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, "checkpoints", f"model_{train_state.env_steps:05d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_state": asdict(train_state),
                    "args": vars(args_cli),
                },
                checkpoint_path,
            )

    final_path = os.path.join(log_dir, "checkpoints", "model_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_state": asdict(train_state),
            "args": vars(args_cli),
        },
        final_path,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
