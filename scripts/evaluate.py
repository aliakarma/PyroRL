#!/usr/bin/env python3
"""
PPO Evaluation Script for PyroRL

Evaluate a trained PPO model on a given environment and report performance.
This script does NOT train anything — it only runs inference.

Usage:
    python scripts/evaluate.py --model checkpoints/best_model.zip --env california
    python scripts/evaluate.py --model checkpoints/best_model.zip --env saudi --episodes 100
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Import PyroRLEnv
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_default_environment(calibration: str = "california"):
    """Create the same environment used during training."""
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path: str, calibration: str, episodes: int, seed: int):
    """Load a trained PPO model and evaluate it."""

    # Validate model file
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    set_seeds(seed)

    print(f"\n{'='*70}")
    print(f"PPO EVALUATION - {calibration.upper()} ENVIRONMENT")
    print(f"{'='*70}")
    print(f"Model:      {model_path}")
    print(f"Env:        {calibration}")
    print(f"Episodes:   {episodes}")
    print(f"Seed:       {seed}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    model = PPO.load(model_path, device="auto")
    print("Model loaded successfully.\n")

    # Run evaluation episodes
    print(f"Running {episodes} evaluation episodes (deterministic=True)...")
    print("-" * 70)

    all_rewards = []
    all_lengths = []

    for ep in range(episodes):
        env = create_default_environment(calibration=calibration)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_value = int(np.asarray(action).item())
            obs, reward, terminated, truncated, _ = env.step(action_value)
            done = terminated or truncated
            total_reward += float(reward)
            episode_length += 1

        all_rewards.append(total_reward)
        all_lengths.append(episode_length)
        print(f"  Episode {ep+1:3d}/{episodes}: reward={total_reward:8.2f}, length={episode_length:4d}")
        env.close()

    print("-" * 70)

    # Compute statistics
    rewards = np.array(all_rewards)
    lengths = np.array(all_lengths)

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))
    median_reward = float(np.median(rewards))

    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))

    # Print results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Mean reward:   {mean_reward:8.2f}")
    print(f"  Std reward:    {std_reward:8.2f}")
    print(f"  Min reward:    {min_reward:8.2f}")
    print(f"  Max reward:    {max_reward:8.2f}")
    print(f"  Median reward: {median_reward:8.2f}")
    print()
    print(f"  Mean length:   {mean_length:8.1f}")
    print(f"  Std length:    {std_length:8.1f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model on PyroRL"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["california", "saudi"],
        help="Environment calibration to evaluate on",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        calibration=args.env,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
