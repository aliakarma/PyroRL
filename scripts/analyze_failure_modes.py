#!/usr/bin/env python3
"""
Failure Mode Analysis System for PyroRL (Phase D — Slice 4)

Analyzes WHY a trained model fails under different scenarios by tracking
detailed metrics (fire spread, action entropy, suppression effectiveness)
and classifying the failure using a rule-based engine.

Usage:
    python scripts/analyze_failure_modes.py --model checkpoints/best_model.zip --calibration saudi
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.stats
import torch
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Import PyroRLEnv
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv

SCENARIOS = [
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


def create_eval_environment(calibration: str = "saudi", scenario: str | None = None):
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

    env = PyroRLEnv(**env_kwargs)
    return env


def evaluate_with_metrics(model: PPO, calibration: str, scenario: str, episodes: int, seed: int) -> dict:
    all_rewards = []
    all_mean_fire_cells = []
    all_total_new_ignitions = []
    all_action_entropies = []
    all_suppression_success_rates = []

    for ep in range(episodes):
        set_seeds(seed + ep)
        env = create_eval_environment(calibration=calibration, scenario=scenario)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        
        total_reward = 0.0
        fire_cells_history = []
        total_new_ignitions = 0
        actions_taken = []
        
        suppression_attempts = 0
        suppression_successes = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_value = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action_value)
            done = terminated or truncated
            
            total_reward += float(reward)
            
            # Track metrics
            fire_cells_history.append(info.get("fire_cells", 0))
            total_new_ignitions += info.get("new_ignitions", 0)
            actions_taken.append(action_value)
            
            # Suppression effectiveness: action taken wasn't the last (do_nothing) 
            # and fire_delta > 0 (meaning fire cells decreased)
            # The action space has num_actions + 1 actions, where the last is do_nothing
            do_nothing_action = env.action_space.n - 1
            if action_value != do_nothing_action:
                suppression_attempts += 1
                if info.get("fire_delta", 0) > 0:
                    suppression_successes += 1

        env.close()

        all_rewards.append(total_reward)
        all_mean_fire_cells.append(np.mean(fire_cells_history) if fire_cells_history else 0)
        all_total_new_ignitions.append(total_new_ignitions)
        
        # Action entropy
        action_counts = Counter(actions_taken)
        probs = [count / len(actions_taken) for count in action_counts.values()]
        entropy = scipy.stats.entropy(probs)
        all_action_entropies.append(entropy)
        
        # Suppression success rate
        rate = (suppression_successes / suppression_attempts) if suppression_attempts > 0 else 0.0
        all_suppression_success_rates.append(rate)

    return {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_fire_cells": np.mean(all_mean_fire_cells),
        "mean_new_ignitions": np.mean(all_total_new_ignitions),
        "mean_action_entropy": np.mean(all_action_entropies),
        "mean_suppression_success_rate": np.mean(all_suppression_success_rates),
    }

def classify_failure(metrics: dict) -> tuple[str, str]:
    """Rule-based failure classification."""
    reward = metrics["mean_reward"]
    std_reward = metrics["std_reward"]
    ignitions = metrics["mean_new_ignitions"]
    entropy = metrics["mean_action_entropy"]
    suppression_rate = metrics["mean_suppression_success_rate"]
    
    # If the policy does well, no major failure
    if reward > -50:
        return "none", "policy succeeds"

    # 1. Action mismatch: trying to suppress but failing
    if suppression_rate < 0.05 and entropy > 0.1: # entropy > 0.1 ensures it's not just spamming "do nothing" always without trying
        return "action mismatch", "suppression ineffective"
        
    # 2. Policy collapse: taking mostly random actions or collapsing to a single useless action
    # If entropy is very high, it's acting randomly. If entropy is very low, it's stuck.
    # Note: entropy of uniform distribution for 6 actions is ~1.79
    if entropy > 1.5:
        return "policy collapse", "actions are random"
    if entropy < 0.05 and suppression_rate == 0.0:
        # Stuck doing nothing or doing one action that never succeeds
        return "policy collapse", "mode collapse to single ineffective action"

    # 3. Underestimates spread: fire grows out of control
    if ignitions > 150:
        return "underestimates spread", "fire grows rapidly"

    # 4. Environment sensitivity: high variance
    if std_reward > 50:
        return "environment sensitivity", "high variance in outcomes"

    # Default fallback
    return "unknown", "complex interaction of factors"


def main():
    parser = argparse.ArgumentParser(description="Failure Mode Analysis System")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model")
    parser.add_argument("--calibration", type=str, default="saudi", choices=["california", "saudi"])
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="logs", help="Base output directory")
    
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    print("Loading model...", end=" ", flush=True)
    model = PPO.load(args.model, device="auto")
    print("OK\n")

    print(f"========================================")
    print(f"FAILURE MODE ANALYSIS")
    print(f"=====================")
    print(f"  Model:       {args.model}")
    print(f"  Calibration: {args.calibration}")
    print(f"  Episodes:    {args.episodes}")
    print(f"========================================\n")

    results = []

    for scenario in SCENARIOS:
        print(f"Analyzing {scenario}...", end=" ", flush=True)
        metrics = evaluate_with_metrics(model, args.calibration, scenario, args.episodes, args.seed)
        failure_type, explanation = classify_failure(metrics)
        
        results.append({
            "scenario": scenario,
            "failure_type": failure_type,
            "explanation": explanation,
            **metrics
        })
        print(f"[{failure_type}]")

    print(f"\n=========================================================================")
    print(f"FAILURE MODE ANALYSIS - RESULTS")
    print(f"=========================================================================")
    header = f"{'Scenario':<18s} {'Failure Type':<25s} {'Explanation':<30s}"
    print(header)
    print("-" * 75)
    
    for r in results:
        print(f"{r['scenario']:<18s} {r['failure_type']:<25s} {r['explanation']:<30s}")
        
    print("-" * 75)

    # Save CSV
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "failure_modes.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "failure_type", "explanation", "mean_reward", "std_reward", 
            "mean_fire_cells", "mean_new_ignitions", "mean_action_entropy", "mean_suppression_success_rate"
        ])
        for r in results:
            writer.writerow([
                r["scenario"], r["failure_type"], r["explanation"], 
                f"{r['mean_reward']:.2f}", f"{r['std_reward']:.2f}",
                f"{r['mean_fire_cells']:.2f}", f"{r['mean_new_ignitions']:.2f}",
                f"{r['mean_action_entropy']:.4f}", f"{r['mean_suppression_success_rate']:.4f}"
            ])

    print(f"\nDetailed metrics saved to: {csv_path}")

if __name__ == "__main__":
    main()
