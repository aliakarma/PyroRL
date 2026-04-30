#!/usr/bin/env python3
"""
Scenario Matrix Experiment Runner for PyroRL (Phase D — Slice 3)

Evaluates California-trained and Saudi-trained PPO models across all scenarios,
producing a structured comparison table and CSV suitable for research reporting.

Usage:
    python scripts/run_scenario_matrix.py \\
        --ca_model checkpoints/best_model.zip \\
        --sa_model checkpoints/best_model.zip

    python scripts/run_scenario_matrix.py \\
        --ca_model checkpoints/ppo_california.zip \\
        --sa_model checkpoints/ppo_saudi.zip \\
        --episodes 50 --seed 0
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Import PyroRLEnv
# ---------------------------------------------------------------------------
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv

# ---------------------------------------------------------------------------
# Scenario list
# ---------------------------------------------------------------------------

SCENARIOS = [
    "high_wind",
    "low_fuel",
    "oasis_cluster",
    "multi_ignition",
    "narrow_corridor",
]

CALIBRATIONS = ["california", "saudi"]


# ---------------------------------------------------------------------------
# Helpers (duplicated from evaluate_scenarios.py to keep this self-contained)
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_eval_environment(calibration: str = "california", scenario: str | None = None):
    """Create the same environment used during training, with optional scenario.

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
# Silent evaluation (no per-episode printing)
# ---------------------------------------------------------------------------

def evaluate_silent(
    model: PPO,
    calibration: str,
    scenario: str | None,
    episodes: int,
    seed: int,
) -> dict:
    """Run evaluation episodes and return a statistics dict.

    Returns:
        dict with keys: mean, std, ci95, min, max, median, rewards
    """
    all_rewards: list[float] = []

    for ep in range(episodes):
        set_seeds(seed + ep)
        env = create_eval_environment(calibration=calibration, scenario=scenario)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_value = int(np.asarray(action).item())
            obs, reward, terminated, truncated, _ = env.step(action_value)
            done = terminated or truncated
            total_reward += float(reward)

        all_rewards.append(total_reward)
        env.close()

    rewards = np.array(all_rewards)
    n = len(rewards)
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    ci95 = 1.96 * (std / np.sqrt(n))

    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
        "rewards": all_rewards,
    }


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------

def run_matrix(
    ca_model_path: str,
    sa_model_path: str,
    episodes: int,
    seed: int,
):
    """Evaluate both models across all scenarios and print a results table."""

    # Validate model files
    for label, path in [("CA", ca_model_path), ("SA", sa_model_path)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} model not found: {path}")
            sys.exit(1)

    # Header
    print(f"\n{'=' * 78}")
    print("SCENARIO EVALUATION MATRIX")
    print(f"{'=' * 78}")
    print(f"  CA model:  {ca_model_path}")
    print(f"  SA model:  {sa_model_path}")
    print(f"  Episodes:  {episodes} per cell")
    print(f"  Seed:      {seed}")
    print(f"{'=' * 78}\n")

    # Load models
    print("Loading CA model...", end=" ")
    ca_model = PPO.load(ca_model_path, device="auto")
    print("OK")
    print("Loading SA model...", end=" ")
    sa_model = PPO.load(sa_model_path, device="auto")
    print("OK\n")

    # Evaluate baseline (no scenario) first
    print("Evaluating baseline (no scenario)...")
    baseline_configs = []
    for cal_label, cal_name, model in [("CA", "california", ca_model), ("SA", "saudi", sa_model)]:
        print(f"  {cal_label} model on {cal_name} baseline...", end=" ", flush=True)
        stats = evaluate_silent(model, cal_name, None, episodes, seed)
        baseline_configs.append((cal_label, cal_name, stats))
        print(f"mean={stats['mean']:8.2f} ± {stats['ci95']:.2f}")

    print()

    # Evaluate all scenarios
    results: list[dict] = []

    total_cells = len(SCENARIOS) * 2
    cell_idx = 0

    for scenario in SCENARIOS:
        row: dict = {"scenario": scenario}

        for cal_label, cal_name, model in [("CA", "california", ca_model), ("SA", "saudi", sa_model)]:
            cell_idx += 1
            print(
                f"  [{cell_idx}/{total_cells}] {cal_label} model × {scenario}...",
                end=" ",
                flush=True,
            )
            stats = evaluate_silent(model, cal_name, scenario, episodes, seed)
            row[f"{cal_label.lower()}_mean"] = stats["mean"]
            row[f"{cal_label.lower()}_std"] = stats["std"]
            row[f"{cal_label.lower()}_ci95"] = stats["ci95"]
            row[f"{cal_label.lower()}_min"] = stats["min"]
            row[f"{cal_label.lower()}_max"] = stats["max"]
            row[f"{cal_label.lower()}_rewards"] = stats["rewards"]
            print(f"mean={stats['mean']:8.2f} ± {stats['ci95']:.2f}")

        row["drop"] = row["sa_mean"] - row["ca_mean"]
        results.append(row)

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("SCENARIO EVALUATION MATRIX - RESULTS")
    print(f"{'=' * 78}")
    print()

    # Baselines
    print("  BASELINES (no scenario)")
    print(f"  {'-' * 60}")
    for cal_label, cal_name, stats in baseline_configs:
        print(
            f"    {cal_label} on {cal_name:12s}:  "
            f"mean={stats['mean']:8.2f}  std={stats['std']:6.2f}  "
            f"95% CI=±{stats['ci95']:5.2f}"
        )
    print()

    # Scenario table
    header = f"  {'Scenario':<18s} {'CA Mean':>10s} {'CA CI':>8s} {'SA Mean':>10s} {'SA CI':>8s} {'Drop':>10s}"
    print(header)
    print(f"  {'-' * 68}")

    for row in results:
        drop_str = f"{row['drop']:+.2f}"
        ca_ci_str = f"±{row['ca_ci95']:.2f}"
        sa_ci_str = f"±{row['sa_ci95']:.2f}"
        print(
            f"  {row['scenario']:<18s} "
            f"{row['ca_mean']:>10.2f} "
            f"{ca_ci_str:>8s} "
            f"{row['sa_mean']:>10.2f} "
            f"{sa_ci_str:>8s} "
            f"{drop_str:>10s}"
        )

    print(f"  {'-' * 68}")
    print()

    # Summary statistics across scenarios
    ca_means = [r["ca_mean"] for r in results]
    sa_means = [r["sa_mean"] for r in results]
    drops = [r["drop"] for r in results]

    print(f"  Average CA mean across scenarios:  {np.mean(ca_means):8.2f}")
    print(f"  Average SA mean across scenarios:  {np.mean(sa_means):8.2f}")
    print(f"  Average drop (SA - CA):            {np.mean(drops):+8.2f}")
    print(f"{'=' * 78}\n")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "scenario_matrix.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "ca_mean", "ca_std", "ca_ci95", "ca_min", "ca_max",
            "sa_mean", "sa_std", "sa_ci95", "sa_min", "sa_max",
            "drop",
        ])
        for row in results:
            writer.writerow([
                row["scenario"],
                f"{row['ca_mean']:.4f}", f"{row['ca_std']:.4f}", f"{row['ca_ci95']:.4f}",
                f"{row['ca_min']:.4f}", f"{row['ca_max']:.4f}",
                f"{row['sa_mean']:.4f}", f"{row['sa_std']:.4f}", f"{row['sa_ci95']:.4f}",
                f"{row['sa_min']:.4f}", f"{row['sa_max']:.4f}",
                f"{row['drop']:.4f}",
            ])

    print(f"Results saved to: {csv_path}")

    # Save per-episode detail CSV
    detail_csv_path = os.path.join(log_dir, "scenario_matrix_detail.csv")
    with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "model", "episode_id", "reward"])
        for row in results:
            for i, r in enumerate(row["ca_rewards"]):
                writer.writerow([row["scenario"], "CA", i + 1, f"{r:.4f}"])
            for i, r in enumerate(row["sa_rewards"]):
                writer.writerow([row["scenario"], "SA", i + 1, f"{r:.4f}"])

    print(f"Detail saved to:  {detail_csv_path}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run full scenario matrix evaluation for PyroRL"
    )
    parser.add_argument(
        "--ca_model",
        type=str,
        required=True,
        help="Path to California-trained PPO model (.zip)",
    )
    parser.add_argument(
        "--sa_model",
        type=str,
        required=True,
        help="Path to Saudi-trained PPO model (.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Episodes per scenario-model cell (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    run_matrix(
        ca_model_path=args.ca_model,
        sa_model_path=args.sa_model,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
