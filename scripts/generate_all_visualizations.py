#!/usr/bin/env python3
"""
Automated Multi-Scenario Visualization Generator for PyroRL

Generates side-by-side comparison GIFs, key-frame PNGs, and captions
for every registered scenario. Outputs are organized into structured
subdirectories under logs/<scenario>/.

Usage:
    python scripts/generate_all_visualizations.py \
        --ca_model checkpoints/ppo_california_best.zip \
        --sa_model checkpoints/ppo_saudi_best.zip

    python scripts/generate_all_visualizations.py \
        --ca_model checkpoints/ppo_california_best.zip \
        --sa_model checkpoints/ppo_saudi_best.zip \
        --annotate --seed 0
"""

import argparse
import os
import sys
import time
from pathlib import Path
import shutil

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.visualize_episode import SCENARIOS, run_compare


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparative visualizations for all PyroRL scenarios",
    )
    parser.add_argument(
        "--ca_model", required=True, help="Path to California-trained model (.zip)",
    )
    parser.add_argument(
        "--sa_model", required=True, help="Path to Saudi-trained model (.zip)",
    )
    parser.add_argument(
        "--annotate", action="store_true", help="Enable annotation overlays",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    temp_dir = Path("temp_frames")
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PYRORL — AUTOMATED VISUALIZATION GENERATOR")
    print("=" * 70)
    print(f"  CA model:  {args.ca_model}")
    print(f"  SA model:  {args.sa_model}")
    print(f"  Annotate:  {args.annotate}")
    print(f"  Seed:      {args.seed}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print("=" * 70)

    t0 = time.time()
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n[{i}/{len(SCENARIOS)}] Processing {scenario}...")
        
        output_dir = Path("logs") / scenario
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_compare(
            ca_model_path=args.ca_model,
            sa_model_path=args.sa_model,
            scenario=scenario,
            output_dir=output_dir,
            temp_dir=temp_dir,
            annotate=args.annotate,
            seed=args.seed
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — {len(SCENARIOS)} scenarios processed in {elapsed:.1f}s")
    print(f"  Output directory: logs/")
    print(f"{'=' * 70}")

    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
