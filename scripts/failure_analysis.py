#!/usr/bin/env python3
"""
Failure Mode Analysis for PyroRL (Phase D - Slice 4)

Analyzes WHY a model fails under different scenarios by tracking per-step
metrics (fire spread, action distribution, suppression effectiveness) and
classifying failure types with rule-based heuristics.

Usage:
    python scripts/failure_analysis.py \\
        --model checkpoints/ppo_california.zip \\
        --calibration california \\
        --episodes 20

    python scripts/failure_analysis.py \\
        --model checkpoints/ppo_california.zip \\
        --calibration california \\
        --scenarios high_wind oasis_cluster
"""

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter
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
# Constants
# ---------------------------------------------------------------------------
ALL_SCENARIOS = [
    "high_wind",
    "low_fuel",
    "oasis_cluster",
    "multi_ignition",
    "narrow_corridor",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_eval_environment(calibration: str = "california", scenario=None):
    """Same layout as train_ppo.py for comparability."""
    num_rows, num_cols = 10, 10
    populated_areas = np.array([[5, 5]])
    paths = np.array([[[4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]], dtype=object)
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
    if scenario is not None:
        env_kwargs["scenario"] = scenario

    return PyroRLEnv(**env_kwargs)


# ---------------------------------------------------------------------------
# Detailed episode runner
# ---------------------------------------------------------------------------
def run_detailed_episode(model, calibration, scenario, seed):
    """Run one episode and collect per-step diagnostics.

    Returns a dict with episode-level aggregated metrics.
    """
    set_seeds(seed)
    env = create_eval_environment(calibration=calibration, scenario=scenario)
    obs, _ = env.reset(seed=seed)

    done = False
    total_reward = 0.0
    step_count = 0

    fire_cells_series = []
    new_ignitions_total = 0
    fire_delta_total = 0.0
    action_counts = Counter()
    suppression_attempts = 0  # steps where fire_delta < 0 (fire shrank)
    fire_grew_steps = 0       # steps where fire_delta > 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_value = int(np.asarray(action).item())
        obs, reward, terminated, truncated, info = env.step(action_value)
        done = terminated or truncated

        total_reward += float(reward)
        step_count += 1

        fire_cells_series.append(info.get("fire_cells", 0))
        new_ignitions_total += info.get("new_ignitions", 0)
        fire_delta = info.get("fire_delta", 0.0)
        fire_delta_total += fire_delta
        action_counts[action_value] += 1

        if fire_delta < -0.01:
            suppression_attempts += 1
        if fire_delta > 0.01:
            fire_grew_steps += 1

    env.close()

    # Compute action entropy (base-2)
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        probs = [c / total_actions for c in action_counts.values()]
        action_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(max(len(action_counts), 2))
        normalized_entropy = action_entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        action_entropy = 0.0
        normalized_entropy = 0.0

    # Suppression success rate
    suppression_rate = suppression_attempts / step_count if step_count > 0 else 0.0
    fire_growth_rate = fire_grew_steps / step_count if step_count > 0 else 0.0

    # Fire acceleration: compare first half vs second half
    mid = len(fire_cells_series) // 2
    first_half_avg = float(np.mean(fire_cells_series[:mid])) if mid > 0 else 0.0
    second_half_avg = float(np.mean(fire_cells_series[mid:])) if mid > 0 else 0.0

    return {
        "reward": total_reward,
        "steps": step_count,
        "mean_fire_cells": float(np.mean(fire_cells_series)) if fire_cells_series else 0.0,
        "max_fire_cells": int(np.max(fire_cells_series)) if fire_cells_series else 0,
        "total_new_ignitions": new_ignitions_total,
        "fire_delta_total": fire_delta_total,
        "action_entropy": action_entropy,
        "normalized_entropy": normalized_entropy,
        "action_dist": dict(action_counts),
        "suppression_rate": suppression_rate,
        "fire_growth_rate": fire_growth_rate,
        "first_half_fire": first_half_avg,
        "second_half_fire": second_half_avg,
    }


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------
def classify_failure(scenario, metrics):
    """Rule-based failure mode classification.

    Args:
        scenario: scenario name
        metrics: dict of aggregated metrics across episodes

    Returns:
        (failure_type, explanation) tuple
    """
    failures = []

    # 1. Rapid fire spread => underestimates spread
    if metrics["mean_fire_growth_rate"] > 0.60:
        failures.append(("underestimates spread", "fire grows on >60% of steps"))

    # 2. High normalized entropy => policy near-random
    if metrics["mean_normalized_entropy"] > 0.90:
        failures.append(("policy collapse", "action distribution near-uniform (entropy >0.90)"))

    # 3. Low suppression rate => actions not reducing fire
    if metrics["mean_suppression_rate"] < 0.10:
        failures.append(("action mismatch", "suppression effective on <10% of steps"))

    # 4. High reward variance => environment sensitivity
    if metrics["reward_std"] > 60:
        failures.append(("environment sensitivity", f"reward std={metrics['reward_std']:.1f} (high variance)"))

    # 5. Fire accelerates in second half => loses control
    if metrics["mean_second_half_fire"] > metrics["mean_first_half_fire"] * 1.5:
        ratio = metrics["mean_second_half_fire"] / max(metrics["mean_first_half_fire"], 0.01)
        failures.append(("loses control", f"fire {ratio:.1f}x larger in second half"))

    # Scenario-specific checks
    if scenario == "high_wind" and metrics["mean_fire_cells"] > 15:
        failures.append(("wind misprediction", "agent ignores wind-driven spread dynamics"))

    if scenario == "oasis_cluster" and metrics["mean_max_fire_cells"] > 30:
        failures.append(("fuel misestimation", "fails to handle dense fuel clusters"))

    if scenario == "multi_ignition" and metrics["mean_total_ignitions"] > 100:
        failures.append(("multi-front overwhelm", "cannot manage simultaneous fire fronts"))

    if not failures:
        if metrics["reward_mean"] > -20:
            return ("none", "model performs adequately")
        failures.append(("general underperformance", "no specific failure pattern detected"))

    # Return the most severe failure (first match)
    return failures[0]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_failure_analysis(model_path, calibration, scenarios, episodes, seed):
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    print(f"\n{'=' * 72}")
    print("FAILURE MODE ANALYSIS")
    print(f"{'=' * 72}")
    print(f"  Model:       {model_path}")
    print(f"  Calibration: {calibration}")
    print(f"  Scenarios:   {', '.join(scenarios)}")
    print(f"  Episodes:    {episodes}")
    print(f"  Seed:        {seed}")
    print(f"{'=' * 72}\n")

    print("Loading model...", end=" ")
    model = PPO.load(model_path, device="auto")
    print("OK\n")

    # Also run baseline (no scenario)
    all_configs = [("baseline", None)] + [(s, s) for s in scenarios]
    all_results = []

    for label, scenario in all_configs:
        print(f"  Analyzing {label}...", end=" ", flush=True)

        ep_metrics = []
        for ep in range(episodes):
            m = run_detailed_episode(model, calibration, scenario, seed + ep)
            ep_metrics.append(m)

        # Aggregate
        agg = {
            "scenario": label,
            "reward_mean": float(np.mean([m["reward"] for m in ep_metrics])),
            "reward_std": float(np.std([m["reward"] for m in ep_metrics])),
            "mean_fire_cells": float(np.mean([m["mean_fire_cells"] for m in ep_metrics])),
            "mean_max_fire_cells": float(np.mean([m["max_fire_cells"] for m in ep_metrics])),
            "mean_total_ignitions": float(np.mean([m["total_new_ignitions"] for m in ep_metrics])),
            "mean_normalized_entropy": float(np.mean([m["normalized_entropy"] for m in ep_metrics])),
            "mean_suppression_rate": float(np.mean([m["suppression_rate"] for m in ep_metrics])),
            "mean_fire_growth_rate": float(np.mean([m["fire_growth_rate"] for m in ep_metrics])),
            "mean_first_half_fire": float(np.mean([m["first_half_fire"] for m in ep_metrics])),
            "mean_second_half_fire": float(np.mean([m["second_half_fire"] for m in ep_metrics])),
        }

        # Classify
        failure_type, explanation = classify_failure(label, agg)
        agg["failure_type"] = failure_type
        agg["explanation"] = explanation
        all_results.append(agg)

        ci95 = 1.96 * agg["reward_std"] / np.sqrt(episodes)
        print(f"reward={agg['reward_mean']:8.2f} +/-{ci95:.2f}  failure={failure_type}")

    # -----------------------------------------------------------------------
    # Print detailed results
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("FAILURE MODE ANALYSIS - RESULTS")
    print(f"{'=' * 72}\n")

    # Metrics table
    hdr = f"  {'Scenario':<18s} {'Reward':>8s} {'Fire':>6s} {'Ignit':>6s} {'Suppr%':>7s} {'Entropy':>8s} {'GrowR':>6s}"
    print(hdr)
    print(f"  {'-' * 64}")

    for r in all_results:
        print(
            f"  {r['scenario']:<18s} "
            f"{r['reward_mean']:>8.1f} "
            f"{r['mean_fire_cells']:>6.1f} "
            f"{r['mean_total_ignitions']:>6.1f} "
            f"{r['mean_suppression_rate']*100:>6.1f}% "
            f"{r['mean_normalized_entropy']:>8.3f} "
            f"{r['mean_fire_growth_rate']*100:>5.1f}%"
        )

    print(f"  {'-' * 64}\n")

    # Failure classification table
    print(f"  {'Scenario':<18s} {'Failure Type':<26s} {'Explanation'}")
    print(f"  {'-' * 64}")

    for r in all_results:
        print(f"  {r['scenario']:<18s} {r['failure_type']:<26s} {r['explanation']}")

    print(f"  {'-' * 64}")
    print(f"{'=' * 72}\n")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "failure_modes.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "reward_mean", "reward_std",
            "mean_fire_cells", "mean_max_fire_cells", "mean_total_ignitions",
            "mean_suppression_rate", "mean_normalized_entropy", "mean_fire_growth_rate",
            "failure_type", "explanation",
        ])
        for r in all_results:
            writer.writerow([
                r["scenario"],
                f"{r['reward_mean']:.4f}", f"{r['reward_std']:.4f}",
                f"{r['mean_fire_cells']:.4f}", f"{r['mean_max_fire_cells']:.4f}",
                f"{r['mean_total_ignitions']:.4f}",
                f"{r['mean_suppression_rate']:.4f}", f"{r['mean_normalized_entropy']:.4f}",
                f"{r['mean_fire_growth_rate']:.4f}",
                r["failure_type"], r["explanation"],
            ])

    print(f"Results saved to: {csv_path}\n")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Failure mode analysis for PyroRL RL experiments"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model (.zip)")
    parser.add_argument(
        "--calibration", type=str, required=True,
        choices=["california", "saudi"], help="Environment calibration",
    )
    parser.add_argument(
        "--scenarios", type=str, nargs="*", default=None,
        choices=ALL_SCENARIOS, help="Scenarios to analyze (default: all)",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per scenario (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    scenarios = args.scenarios if args.scenarios else ALL_SCENARIOS

    run_failure_analysis(
        model_path=args.model,
        calibration=args.calibration,
        scenarios=scenarios,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
