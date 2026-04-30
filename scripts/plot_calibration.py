#!/usr/bin/env python
"""
Plot California vs Saudi calibration trajectories from CSV output.

Usage:
    python scripts/plot_calibration.py results.csv
    python scripts/plot_calibration.py results.csv --output plot.png
"""

import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print(
        "ERROR: matplotlib and pandas required for plotting.\n"
        "  pip install matplotlib pandas"
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Plot calibration comparison.")
    parser.add_argument("csv", type=str, help="Path to summary CSV from test_calibration.py")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for plot (e.g., plot.png). Default: display.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 6))

    step = df["step"]
    calif_mean = df["california_mean"]
    calif_std = df["california_std"]
    saudi_mean = df["saudi_mean"]
    saudi_std = df["saudi_std"]

    ax.plot(step, calif_mean, "b-", linewidth=2, label="California (mean)")
    ax.fill_between(
        step,
        calif_mean - calif_std,
        calif_mean + calif_std,
        alpha=0.2,
        color="blue",
        label="California (± std)",
    )

    ax.plot(step, saudi_mean, "r-", linewidth=2, label="Saudi (mean)")
    ax.fill_between(
        step,
        saudi_mean - saudi_std,
        saudi_mean + saudi_std,
        alpha=0.2,
        color="red",
        label="Saudi (± std)",
    )

    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Fire cells", fontsize=12)
    ax.set_title("PyroRL Calibration Comparison: California vs Saudi", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
