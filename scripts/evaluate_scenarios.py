#!/usr/bin/env python3
"""
Statistical Evaluation Engine for PyroRL RL Experiments (Phase D — Slice 2)

Runs multiple evaluation episodes with a trained PPO model, collects rewards,
and reports mean, std, min, max, and 95% confidence intervals suitable for
research reporting.

Usage:
    python scripts/evaluate_scenarios.py --model checkpoints/best_model.zip --calibration saudi --scenario high_wind
    python scripts/evaluate_scenarios.py --model checkpoints/best_model.zip --calibration california --episodes 50
"""

import argparse
import csv
import importlib
import os
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Import PyroRLEnv — add repo root to sys.path so nested package resolves
# ---------------------------------------------------------------------------
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AVAILABLE_SCENARIOS = [
    "high_wind",
    "low_fuel",
    "oasis_cluster",
    "multi_ignition",
    "narrow_corridor",
]


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_eval_environment(calibration: str = "california", scenario: str | None = None):
    """Create the same environment used during training, with optional scenario overlay.

    Uses the identical map layout and reward configuration as ``train_ppo.py``
    so that evaluation results are directly comparable to training metrics.
    """
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

    if scenario is not None:
        env_kwargs["scenario"] = scenario

    env = PyroRLEnv(**env_kwargs)
    return env


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    model_path: str,
    calibration: str,
    scenario: str | None,
    episodes: int,
    seed: int,
    save_csv: bool,
):
    """Load a trained PPO model and evaluate it over multiple episodes."""

    # Validate model file
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    scenario_label = scenario if scenario else "none"

    # Header
    print(f"\n{'=' * 50}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"  Model:       {model_path}")
    print(f"  Calibration: {calibration}")
    print(f"  Scenario:    {scenario_label}")
    print(f"  Episodes:    {episodes}")
    print(f"  Seed:        {seed}")
    print(f"{'=' * 50}\n")

    # Load model
    print("Loading model...")
    model = PPO.load(model_path, device="auto")
    print("Model loaded successfully.\n")

    # Run episodes
    print(f"Running {episodes} evaluation episodes (deterministic=True)...")
    print("-" * 50)

    all_rewards: list[float] = []
    all_lengths: list[int] = []

    for ep in range(episodes):
        set_seeds(seed + ep)
        env = create_eval_environment(calibration=calibration, scenario=scenario)
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
        print(f"  Episode {ep + 1:3d}/{episodes}: reward={total_reward:8.2f}, length={episode_length:4d}")
        env.close()

    print("-" * 50)

    # -----------------------------------------------------------------------
    # Compute statistics
    # -----------------------------------------------------------------------
    rewards = np.array(all_rewards)
    lengths = np.array(all_lengths)

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))
    median_reward = float(np.median(rewards))

    n = len(rewards)
    ci_95 = 1.96 * (std_reward / np.sqrt(n))

    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"  Scenario:      {scenario_label}")
    print(f"  Calibration:   {calibration}")
    print(f"  Episodes:      {episodes}")
    print()
    print(f"  Mean reward:   {mean_reward:8.2f}")
    print(f"  Std reward:    {std_reward:8.2f}")
    print(f"  95% CI:        ± {ci_95:.2f}")
    print(f"  Min reward:    {min_reward:8.2f}")
    print(f"  Max reward:    {max_reward:8.2f}")
    print(f"  Median reward: {median_reward:8.2f}")
    print()
    print(f"  Mean length:   {mean_length:8.1f}")
    print(f"  Std length:    {std_length:8.1f}")
    print(f"{'=' * 50}\n")

    # -----------------------------------------------------------------------
    # Optionally save CSV
    # -----------------------------------------------------------------------
    if save_csv:
        log_dir = os.path.join("logs")
        os.makedirs(log_dir, exist_ok=True)
        csv_filename = f"eval_{calibration}_{scenario_label}.csv"
        csv_path = os.path.join(log_dir, csv_filename)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode_id", "reward", "length"])
            for i, (r, l) in enumerate(zip(all_rewards, all_lengths)):
                writer.writerow([i + 1, f"{r:.4f}", l])

        print(f"Results saved to: {csv_path}\n")

    return mean_reward, std_reward, ci_95


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Statistical evaluation engine for PyroRL RL experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained PPO model (.zip file)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        required=True,
        choices=["california", "saudi"],
        help="Environment calibration mode",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=AVAILABLE_SCENARIOS,
        help="Optional scenario overlay (e.g. high_wind, low_fuel)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of evaluation episodes (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        default=True,
        help="Save per-episode results to CSV (default: True)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV saving",
    )

    args = parser.parse_args()
    save_csv = args.save_csv and not args.no_csv

    run_evaluation(
        model_path=args.model,
        calibration=args.calibration,
        scenario=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
        save_csv=save_csv,
    )


if __name__ == "__main__":
    main()
