#!/usr/bin/env python3
"""
PPO Training Script for PyroRL

This script trains a PPO agent on PyroRL (California or Saudi calibration).
It supports longer training runs, checkpointing, and logging.

Usage:
    python scripts/train_ppo.py --calibration california --timesteps 300000
"""

import argparse
import csv
import os
import random
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add repo root to sys.path so the nested package resolves correctly
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv


def linear_lr_schedule(initial_value: float = 1e-4):
    """
    Stable-Baselines3 learning rate schedule (linear decay).
    """
    def schedule(progress_remaining: float) -> float:
        progress_remaining = float(np.clip(progress_remaining, 0.0, 1.0))
        return initial_value * progress_remaining
    return schedule


class TrainingDiagnosticsCallback(BaseCallback):
    """Track episode rewards, lengths, and log to CSV."""

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'reward', 'length', 'moving_avg_reward'])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        if infos:
            info = infos[0]
            episode_info = info.get("episode")
            if episode_info is not None:
                reward = float(episode_info["r"])
                length = int(episode_info["l"])
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                
                moving_avg = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else reward
                
                with open(self.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.num_timesteps, reward, length, moving_avg])
        return True


class DegradationWarningCallback(BaseCallback):
    """
    Logs a warning if the moving average reward drops significantly from its peak,
    but does NOT stop training.
    """
    def __init__(self, window_size: int = 50, drop_threshold: float = 40.0, min_timesteps: int = 50000, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.drop_threshold = drop_threshold
        self.min_timesteps = min_timesteps
        self.best_moving_avg = -np.inf
        self.episode_rewards: list[float] = []
        self.warned_recently = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        if infos:
            info = infos[0]
            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_rewards.append(float(episode_info["r"]))
                
                if len(self.episode_rewards) >= self.window_size:
                    moving_avg = float(np.mean(self.episode_rewards[-self.window_size:]))
                    if moving_avg > self.best_moving_avg:
                        self.best_moving_avg = moving_avg
                        self.warned_recently = False
                    elif moving_avg < self.best_moving_avg - self.drop_threshold:
                        if self.num_timesteps >= self.min_timesteps and not self.warned_recently:
                            if self.verbose > 0:
                                print(f"\n[WARNING] Policy degradation detected at timestep {self.num_timesteps}!")
                                print(f"Moving avg ({moving_avg:.2f}) dropped by > {self.drop_threshold} from peak ({self.best_moving_avg:.2f}).")
                                print("Training will continue to full timesteps.")
                            self.warned_recently = True
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
        [[[4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]],
        dtype=object,
    )
    paths_to_pops = {0: [[5, 5]]}

    env_kwargs = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "populated_areas": populated_areas,
        "paths": paths,
        "paths_to_pops": paths_to_pops,
        "custom_fire_locations": np.array([[4, 4]]),
        "fire_propagation_rate": 0.06,
        "calibration": calibration,
        "terminate_on_population_loss": False,
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


def format_timesteps(timesteps: int) -> str:
    if timesteps >= 1000 and timesteps % 1000 == 0:
        return f"{timesteps // 1000}k"
    return str(timesteps)


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
    save_name: str,
    log_dir: str,
    eval_freq: int,
):
    # Set seeds
    set_seeds(seed)
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PPO TRAINING - {calibration.upper()} CALIBRATION")
    print(f"{'='*70}")
    print(f"Timesteps:    {timesteps:,}")
    print(f"Seed:         {seed}")
    print(f"Log dir:      {log_dir}")
    print(f"Eval freq:    {eval_freq}")
    print(f"{'='*70}\n")
    
    # Create environment with monitoring
    env = create_default_environment(calibration=calibration)
    env.reset(seed=seed)
    env = Monitor(env, log_dir)
    
    eval_env = create_default_environment(calibration=calibration)
    eval_env.reset(seed=seed + 1000)
    eval_env = Monitor(eval_env, log_dir)
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_lr_schedule(1e-4),
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="auto",
        seed=seed,
        tensorboard_log=log_dir,
    )
    
    csv_log_path = os.path.join(log_dir, f"{calibration}_training_curve.csv")
    diagnostics = TrainingDiagnosticsCallback(log_path=csv_log_path, verbose=0)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints",
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    
    degradation_callback = DegradationWarningCallback(verbose=1)
    
    # Train
    print("Starting training...")
    print("-" * 70)
    
    model.learn(total_timesteps=timesteps, callback=[diagnostics, eval_callback, degradation_callback])
    
    print("-" * 70)
    print("Training complete!\n")

    # Rename best model
    best_model_path = os.path.join("checkpoints", "best_model.zip")
    renamed_best_model_path = os.path.join("checkpoints", f"ppo_{calibration}_best.zip")
    if os.path.exists(best_model_path):
        if os.path.exists(renamed_best_model_path):
            os.remove(renamed_best_model_path)
        shutil.move(best_model_path, renamed_best_model_path)
        print(f"Saved best model to {renamed_best_model_path}")
        # Load best model for evaluation
        model = PPO.load(renamed_best_model_path, env=env, device="auto")
    else:
        print("Warning: best model not found.")

    # Save final model
    timestep_str = format_timesteps(timesteps)
    final_model_name = f"ppo_{calibration}_{timestep_str}"
    if save_name:
        final_model_name += f"_{save_name}"
    final_model_path = os.path.join("checkpoints", f"{final_model_name}.zip")
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}\n")

    # Print training summary
    print(f"{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total timesteps: {timesteps}")
    if os.path.exists(csv_log_path):
        try:
            with open(csv_log_path, 'r') as f:
                reader = csv.DictReader(f)
                rewards = [float(row['reward']) for row in reader]
                if rewards:
                    best_reward = max(rewards)
                    final_reward = rewards[-1]
                    print(f"Best reward:     {best_reward:.2f}")
                    print(f"Final reward:    {final_reward:.2f}")
                    if len(rewards) >= 10:
                        first_10_avg = np.mean(rewards[:10])
                        last_10_avg = np.mean(rewards[-10:])
                        improvement = last_10_avg - first_10_avg
                        print(f"Improvement:     {improvement:+.2f} (from first 10 to last 10 episodes)")
        except Exception as e:
            print(f"Could not read training curve: {e}")
    print(f"{'='*70}\n")

    def make_eval_env():
        return create_default_environment(calibration=calibration)

    baseline_episodes = 10
    random_rewards, random_lengths = run_rollouts(
        make_eval_env,
        lambda obs, env: env.action_space.sample(),
        episodes=baseline_episodes,
        seed=seed + 2000,
    )

    ppo_rewards, ppo_lengths = run_rollouts(
        make_eval_env,
        lambda obs, env: int(np.asarray(model.predict(obs, deterministic=True)[0]).item()),
        episodes=baseline_episodes,
        seed=seed + 2000,
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
    
    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="PPO training for PyroRL"
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
        default=300000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="",
        help="Optional model name",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/ppo/",
        help="Directory for logs",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="Evaluation frequency",
    )
    
    args = parser.parse_args()
    
    train_ppo(
        calibration=args.calibration,
        timesteps=args.timesteps,
        seed=args.seed,
        save_name=args.save_name,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
    )


if __name__ == "__main__":
    main()
