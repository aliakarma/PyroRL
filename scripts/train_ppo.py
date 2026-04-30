#!/usr/bin/env python3
"""
PPO Sanity Check Training Script for PyroRL

This script trains a PPO agent on PyroRL (California or Saudi calibration)
to verify that RL can learn meaningful behavior. This is a sanity check before
scaling to cloud training, not an optimization benchmark.

Usage:
    python scripts/train_ppo.py --calibration california --timesteps 50000
    python scripts/train_ppo.py --calibration saudi --timesteps 100000
"""

import argparse
import importlib
import csv
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Ensure PyroRL is importable
import sys

# Add parent directory to path if needed
repo_root = Path(__file__).parent.parent

# Prefer loading the local source tree even if an older `pyrorl` is installed.
local_pkg_root = repo_root / "pyrorl"
local_env_path = local_pkg_root / "pyrorl" / "envs" / "pyrorl.py"
if local_env_path.exists():
    sys.path.insert(0, str(local_pkg_root))
    spec = importlib.util.spec_from_file_location("pyrorl_local_env", str(local_env_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load local env module from {local_env_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PyroRLEnv = module.PyroRLEnv
else:
    # Fallback to installed package import.
    PyroRLEnv = importlib.import_module("pyrorl.pyrorl.envs").PyroRLEnv


def linear_lr_schedule(
    start_lr: float = 3e-4, end_lr: float = 1e-4
):
    """
    Stable-Baselines3 learning rate schedule.

    progress_remaining goes from 1.0 (start) to 0.0 (end).
    """

    def schedule(progress_remaining: float) -> float:
        progress_remaining = float(np.clip(progress_remaining, 0.0, 1.0))
        return end_lr + (start_lr - end_lr) * progress_remaining

    return schedule


class TrainingDiagnosticsCallback(BaseCallback):
    """Track episode rewards, actions, and fire counts during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.fire_counts: list[float] = []
        self.new_ignitions: list[float] = []
        self.action_counts: Counter[int] = Counter()

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            flat_actions = np.asarray(actions).reshape(-1)
            for action in flat_actions:
                self.action_counts[int(action)] += 1

        infos = self.locals.get("infos") or []
        if infos:
            info = infos[0]
            if "fire_cells" in info:
                self.fire_counts.append(float(info["fire_cells"]))
            if "new_ignitions" in info:
                self.new_ignitions.append(float(info["new_ignitions"]))

            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_rewards.append(float(episode_info["r"]))
                self.episode_lengths.append(int(episode_info["l"]))
                if len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    recent_lengths = self.episode_lengths[-10:]
                    recent_fire = self.fire_counts[-100:] if self.fire_counts else []
                    recent_ign = self.new_ignitions[-100:] if self.new_ignitions else []
                    mean_fire = float(np.mean(recent_fire)) if recent_fire else 0.0
                    mean_ign = float(np.mean(recent_ign)) if recent_ign else 0.0
                    print(
                        f"[diagnostics] episodes={len(self.episode_rewards)} "
                        f"mean_reward_last10={np.mean(recent_rewards):.2f} "
                        f"mean_length_last10={np.mean(recent_lengths):.1f} "
                        f"mean_fire_cells={mean_fire:.1f} "
                        f"mean_new_ignitions={mean_ign:.2f}"
                    )

        return True

    def _on_training_end(self) -> None:
        if self.action_counts:
            total_actions = sum(self.action_counts.values())
            action_summary = ", ".join(
                f"{action}:{count / total_actions:.2%}"
                for action, count in sorted(self.action_counts.items())
            )
            print(f"Action distribution: {action_summary}")

        if self.fire_counts:
            print(f"Mean fire cells during training: {np.mean(self.fire_counts):.2f}")


class EarlyStopAndBestModelCallback(BaseCallback):
    """
    - Track moving average episode reward (window).
    - Save best model whenever moving average improves.
    - Stop early if moving average drops > drop_frac from best.
    - Every eval_every_timesteps, run evaluation episodes and update best model if improved.
    """

    def __init__(
        self,
        save_dir: str,
        env_factory,
        seed: int,
        window: int = 50,
        min_timesteps_before_stop: int = 50_000,
        abs_drop_threshold: float = 20.0,
        patience_episodes: int = 300,
        eval_every_timesteps: int = 10_000,
        eval_episodes: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.env_factory = env_factory
        self.seed = int(seed)
        self.window = int(window)
        self.min_timesteps_before_stop = int(min_timesteps_before_stop)
        self.abs_drop_threshold = float(abs_drop_threshold)
        self.patience_episodes = int(patience_episodes)
        self.eval_every_timesteps = int(eval_every_timesteps)
        self.eval_episodes = int(eval_episodes)

        self.episode_rewards: list[float] = []
        self.best_eval_reward: float = -1e9
        self.best_train_ma: float = -np.inf
        self.best_timestep: int = 0
        self.best_model_path = os.path.join(save_dir, "best_model.zip")
        self.last_eval_timestep: int = 0
        self.last_improve_episode_idx: int = 0

    def _on_step(self) -> bool:
        # Periodic evaluation (timestep-based)
        if (
            self.eval_every_timesteps > 0
            and (int(self.num_timesteps) - int(self.last_eval_timestep))
            >= self.eval_every_timesteps
        ):
            self.last_eval_timestep = int(self.num_timesteps)
            if self.model is not None:
                eval_rewards, _ = run_rollouts(
                    self.env_factory,
                    lambda obs, env: int(
                        np.asarray(self.model.predict(obs, deterministic=True)[0]).item()
                    ),
                    episodes=self.eval_episodes,
                    seed=self.seed + 9000 + self.last_eval_timestep,
                )
                eval_avg = float(np.mean(eval_rewards)) if eval_rewards else float("-inf")
                eval_std = float(np.std(eval_rewards)) if eval_rewards else float("inf")
                print(
                    f"[eval] timesteps={self.last_eval_timestep} "
                    f"avg_reward={eval_avg:.2f} ± {eval_std:.2f} over {self.eval_episodes} episodes"
                )
                print(
                    f"[best-update] eval={eval_avg:.2f}, best={self.best_eval_reward:.2f}"
                )
                if eval_avg > self.best_eval_reward:
                    self.best_eval_reward = eval_avg
                    self.best_timestep = int(self.last_eval_timestep)
                    self.model.save(self.best_model_path)
                    # Reset patience on improvement (episode-count based patience).
                    self.last_improve_episode_idx = len(self.episode_rewards)
                    print(
                        f"[best] source=eval best_eval_reward={self.best_eval_reward:.2f} ± {eval_std:.2f} "
                        f"timestep={self.best_timestep} -> saved {self.best_model_path}"
                    )

        infos = self.locals.get("infos") or []
        if not infos:
            return True

        info = infos[0]
        episode_info = info.get("episode")
        if episode_info is None:
            return True

        self.episode_rewards.append(float(episode_info["r"]))
        ep_idx = len(self.episode_rewards)
        if ep_idx < self.window:
            return True

        moving_avg = float(np.mean(self.episode_rewards[-self.window :]))
        improved = moving_avg > self.best_train_ma + 1e-8
        if improved:
            self.best_train_ma = moving_avg
            # We no longer save the model based on training MA.
            # We track best_train_ma solely for early stopping logic.
            self.last_improve_episode_idx = ep_idx
        else:
            # Avoid premature stopping: require minimum training timesteps AND patience.
            if int(self.num_timesteps) >= self.min_timesteps_before_stop:
                episodes_since_improve = ep_idx - self.last_improve_episode_idx
                if episodes_since_improve >= self.patience_episodes:
                    # Stop only on a real collapse (absolute drop, not percentage).
                    if np.isfinite(self.best_train_ma):
                        if moving_avg < (self.best_train_ma - self.abs_drop_threshold):
                            print(
                                f"[early-stop] collapse detected: moving_avg < best_train_ma - {self.abs_drop_threshold:.0f} "
                                f"(best_train_ma={self.best_train_ma:.2f}, "
                                f"current_ma={moving_avg:.2f} at t={int(self.num_timesteps)}, "
                                f"episodes_since_improve={episodes_since_improve})"
                            )
                            return False

        return True


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_default_environment(calibration: str = "california"):
    """Create a smaller environment that is easier for PPO to learn on."""
    num_rows, num_cols = 10, 10

    populated_areas = np.array([[5, 5]])

    paths = np.array(
        [
            [[4, 5], [3, 5], [2, 5], [1, 5], [0, 5]],
        ],
        dtype=object,
    )

    paths_to_pops = {
        0: [[5, 5]],
    }

    env_kwargs = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "populated_areas": populated_areas,
        "paths": paths,
        "paths_to_pops": paths_to_pops,
        "custom_fire_locations": np.array([[4, 4]]),
        "fire_propagation_rate": 0.06,
        "calibration": calibration,
        # Keep episode horizon fixed so PPO sees stable trajectories.
        "terminate_on_population_loss": False,
        # Fire-centric dense reward for initial learnability.
        "reward_weights": {
            "fire_delta": 1.0,
            "burning_cells": 0.20,
            "new_ignitions": 0.10,
            "newly_burned_population": 0.0,
            "burning_population": 0.0,
            "finished_evac": 0.0,
        },
    }
    if calibration == "saudi":
        env_kwargs["wind_speed"] = 6.0
        env_kwargs["wind_angle"] = np.deg2rad(135.0)

    env = PyroRLEnv(**env_kwargs)
    return env


def run_rollouts(env_factory, policy_fn, episodes: int, seed: int):
    rewards = []
    lengths = []

    for episode in range(episodes):
        set_seeds(seed + episode)
        env = env_factory()
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        episode_length = 0

        while not (terminated or truncated):
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            episode_length += 1

        rewards.append(total_reward)
        lengths.append(episode_length)
        env.close()

    return rewards, lengths


def train_ppo(
    calibration: str,
    timesteps: int,
    seed: int,
    save_path: str,
    verbose: int = 1,
):
    """
    Train PPO agent on PyroRL environment.
    
    Args:
        calibration: "california" or "saudi"
        timesteps: Total timesteps to train
        seed: Random seed
        save_path: Directory to save checkpoints
        verbose: Verbosity level for PPO training
    """
    
    # Set seeds
    set_seeds(seed)
    
    # Create directories
    Path(save_path).mkdir(parents=True, exist_ok=True)
    log_dir = f"logs/{calibration}_ppo"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PPO SANITY CHECK - {calibration.upper()} CALIBRATION")
    print(f"{'='*70}")
    print(f"Timesteps:    {timesteps:,}")
    print(f"Seed:         {seed}")
    print(f"Save path:    {save_path}")
    print(f"Log dir:      {log_dir}")
    print(f"{'='*70}\n")
    
    # Create environment with monitoring
    env = create_default_environment(calibration=calibration)
    env = Monitor(
        env,
        log_dir,
        info_keywords=(
            "fire_cells",
            "burning_population",
            "newly_burned",
            "fire_delta",
            "new_ignitions",
        ),
    )
    
    # Print environment info
    print(f"Environment created: {calibration}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space:      {env.action_space}")
    print()
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_lr_schedule(start_lr=3e-4, end_lr=1e-4),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        device="auto",
        seed=seed,
        tensorboard_log=log_dir,
    )
    
    print("PPO Model Configuration:")
    print(f"  Learning rate: linear(3e-4 -> 1e-4)")
    print(f"  n_steps:       2048")
    print(f"  batch_size:    64")
    print(f"  gamma:         0.99")
    print(f"  n_epochs:      10")
    print(f"  ent_coef:      0.005")
    print()

    diagnostics = TrainingDiagnosticsCallback(verbose=0)
    early_stop = EarlyStopAndBestModelCallback(
        save_dir=save_path,
        env_factory=lambda: create_default_environment(calibration=calibration),
        seed=seed,
        window=50,
        min_timesteps_before_stop=50_000,
        abs_drop_threshold=20.0,
        patience_episodes=300,
        eval_every_timesteps=10_000,
        eval_episodes=20,
        verbose=1 if verbose else 0,
    )
    
    # Train
    print("Starting training...")
    print("-" * 70)
    
    model.learn(total_timesteps=timesteps, callback=[diagnostics, early_stop])
    
    print("-" * 70)
    print("Training complete!\n")

    if os.path.exists(early_stop.best_model_path):
        print(f"Best eval reward: {early_stop.best_eval_reward:.2f}")
        print(f"Best timestep:    {early_stop.best_timestep}")
        print(f"Best source:      eval")
        print(f"Best model path:  {early_stop.best_model_path}\n")
        model = PPO.load(early_stop.best_model_path, env=env, device="auto")
    else:
        print("Warning: best_model.zip not found; using last policy.\n")
    
    # Extract rewards from monitor
    episode_rewards = []
    episode_lengths = []
    
    monitor_file = os.path.join(log_dir, "monitor.csv")
    if os.path.exists(monitor_file):
        try:
            with open(monitor_file, newline="", encoding="utf-8") as monitor_handle:
                reader = csv.DictReader(line for line in monitor_handle if not line.startswith("#"))
                for row in reader:
                    if "r" in row and "l" in row:
                        episode_rewards.append(float(row["r"]))
                        episode_lengths.append(int(float(row["l"])))
        except Exception as e:
            print(f"Warning: Could not read monitor file: {e}")
    
    # Print training summary
    if episode_rewards:
        print("TRAINING SUMMARY")
        print("-" * 70)
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Reward range:   [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")
        
        if len(episode_rewards) >= 10:
            last_10_avg = np.mean(episode_rewards[-10:])
            first_10_avg = np.mean(episode_rewards[:10])
            print(f"First 10 avg:   {first_10_avg:.2f}")
            print(f"Last 10 avg:    {last_10_avg:.2f}")
            improvement = last_10_avg - first_10_avg
            print(f"Improvement:    {improvement:+.2f}")
        else:
            avg_reward = np.mean(episode_rewards)
            print(f"Average reward: {avg_reward:.2f}")
        
        print(f"Avg episode length: {np.mean(episode_lengths):.1f} steps")
        print("-" * 70)
        print()

    def make_eval_env():
        return create_default_environment(calibration=calibration)

    baseline_episodes = 10
    random_rewards, random_lengths = run_rollouts(
        make_eval_env,
        lambda obs, env: env.action_space.sample(),
        episodes=baseline_episodes,
        seed=seed + 1000,
    )

    ppo_rewards, ppo_lengths = run_rollouts(
        make_eval_env,
        lambda obs, env: int(np.asarray(model.predict(obs, deterministic=True)[0]).item()),
        episodes=baseline_episodes,
        seed=seed + 1000,
    )

    print("BASELINE COMPARISON")
    print("-" * 70)
    print(
        f"Random policy: avg reward={np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}, "
        f"avg length={np.mean(random_lengths):.1f}"
    )
    print(
        f"PPO policy:    avg reward={np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}, "
        f"avg length={np.mean(ppo_lengths):.1f}"
    )
    print("-" * 70)
    print()
    
    # Final saved artifact is best_model.zip (captured during training).
    
    # Evaluation
    print("EVALUATION (20 PPO episodes)")
    print("-" * 70)
    
    eval_rewards = []
    eval_lengths = []
    
    for episode in range(20):
        eval_env = make_eval_env()
        obs, _ = eval_env.reset(seed=seed + 3000 + episode)
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            action_value = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = eval_env.step(action_value)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        print(f"  Episode {episode+1}: reward={episode_reward:7.2f}, length={episode_length:4d}")
        eval_env.close()
    
    print("-" * 70)
    print(f"Eval avg reward: {np.mean(eval_rewards):7.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Eval avg length: {np.mean(eval_lengths):7.1f} ± {np.std(eval_lengths):.1f}")
    print()
    
    # Plot reward curve
    if episode_rewards:
        plot_path = os.path.join(log_dir, f"{calibration}_reward_curve.png")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Episode rewards
        ax1.plot(episode_rewards, alpha=0.5, label="Episode reward")
        if len(episode_rewards) > 20:
            window = 20
            smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(episode_rewards)), smoothed, linewidth=2, label=f"Smoothed (window={window})")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title(f"{calibration.upper()} - Episode Rewards")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Episode lengths
        ax2.plot(episode_lengths, alpha=0.5, color='orange', label="Episode length")
        if len(episode_lengths) > 20:
            window = 20
            smoothed = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(episode_lengths)), smoothed, linewidth=2, color='darkorange', label=f"Smoothed (window={window})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Length (steps)")
        ax2.set_title(f"{calibration.upper()} - Episode Lengths")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Reward curve saved to: {plot_path}\n")
        plt.close()
    
    # Close environment
    env.close()
    
    print(f"{'='*70}")
    print("PPO SANITY CHECK COMPLETE")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="PPO sanity check training for PyroRL"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="california",
        choices=["california", "saudi"],
        help="Environment calibration mode",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level for PPO training (0-2)",
    )
    
    args = parser.parse_args()
    
    train_ppo(
        calibration=args.calibration,
        timesteps=args.timesteps,
        seed=args.seed,
        save_path=args.save_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
