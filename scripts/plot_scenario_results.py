#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot PyroRL scenario evaluation results")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory containing scenario_matrix.csv")
    args = parser.parse_args()

    print(f"[INFO] Output directory: {args.output_dir}")

    from pathlib import Path
    out_dir = Path(args.output_dir)
    csv_path = out_dir / "scenario_matrix.csv"
    
    assert csv_path.exists(), f"Error: {csv_path} not found. Please run run_scenario_matrix.py first."

    df = pd.read_csv(csv_path)
    
    scenarios = df['scenario'].tolist()
    ca_means = df['ca_mean'].values
    ca_ci = df['ca_ci95'].values
    sa_means = df['sa_mean'].values
    sa_ci = df['sa_ci95'].values
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use clean, distinct colors
    ax.bar(x - width/2, ca_means, width, yerr=ca_ci, label='CA Model (Baseline)', color='#2b83ba', capsize=4, alpha=0.9)
    ax.bar(x + width/2, sa_means, width, yerr=sa_ci, label='SA Model (Arid/Shamal)', color='#d7191c', capsize=4, alpha=0.9)
    
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Scenario Performance Comparison (Distribution Shift)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='lower left')
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_scenario_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] {out_path}")

if __name__ == "__main__":
    main()
