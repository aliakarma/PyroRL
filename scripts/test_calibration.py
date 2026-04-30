import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from pyrorl.envs import PyroRLEnv
    from pyrorl.envs.map_helpers.create_map_info import generate_map_info
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "pyrorl"))
    from pyrorl.envs import PyroRLEnv
    from pyrorl.envs.map_helpers.create_map_info import generate_map_info


def build_env_inputs(num_rows: int, num_cols: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    populated_areas, paths, paths_to_pops = generate_map_info(
        num_rows,
        num_cols,
        num_populated_areas=5,
        save_map=False,
    )

    center_row = num_rows // 2
    center_col = num_cols // 2
    custom_fire_locations = np.array(
        [[center_row, center_col], [center_row, min(center_col + 1, num_cols - 1)]]
    )

    return populated_areas, paths, paths_to_pops, custom_fire_locations


def rollout(calibration: str, env_inputs: tuple, steps: int, debug: bool):
    populated_areas, paths, paths_to_pops, custom_fire_locations = env_inputs
    env = PyroRLEnv(
        num_rows=20,
        num_cols=20,
        populated_areas=populated_areas,
        paths=paths,
        paths_to_pops=paths_to_pops,
        custom_fire_locations=custom_fire_locations,
        calibration=calibration,
        debug=debug,
    )
    env.reset()

    fire_counts = []
    for _ in range(steps):
        state = env.unwrapped.fire_env.get_state()
        fire_counts.append(int(state[0].sum()))
        action = env.action_space.n - 1
        env.step(action)

    return fire_counts


def compute_stats(traces: list[list[int]]):
    arr = np.array(traces)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    avg = mean.mean()
    return mean, std, avg


def compute_trajectory_stats(traces: list[list[int]]):
    arr = np.array(traces)
    peaks = [max(t) if t else 0 for t in traces]
    peak_times = [t.index(max(t)) if t and max(t) > 0 else -1 for t in traces]
    finals = [t[-1] if t else 0 for t in traces]

    return {
        "peak_mean": float(np.mean(peaks)),
        "peak_std": float(np.std(peaks)),
        "peak_time_mean": float(np.mean([t for t in peak_times if t >= 0])) if any(t >= 0 for t in peak_times) else 0.0,
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals)),
    }


def write_csv(
    csv_path: Path,
    calibration: str,
    seeds: list[int],
    traces: list[list[int]],
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "calibration", "seed", "fire_cells"])
        for seed, trace in zip(seeds, traces):
            for step, count in enumerate(trace):
                writer.writerow([step, calibration, seed, count])


def write_summary_csv(
    csv_path: Path,
    calibration_results: dict,
    seeds: list[int],
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "california_mean", "california_std", "saudi_mean", "saudi_std"])
        california_mean = calibration_results["california"]["mean"]
        california_std = calibration_results["california"]["std"]
        saudi_mean = calibration_results["saudi"]["mean"]
        saudi_std = calibration_results["saudi"]["std"]
        for step in range(len(california_mean)):
            writer.writerow([
                step,
                f"{california_mean[step]:.2f}",
                f"{california_std[step]:.2f}",
                f"{saudi_mean[step]:.2f}",
                f"{saudi_std[step]:.2f}",
            ])


def print_trajectory_sample(calibration: str, traces: list[list[int]], steps_to_show: int = 20):
    print(f"\n{calibration.upper()} first {steps_to_show} steps (per seed):")
    for seed_idx, trace in enumerate(traces):
        sample = trace[:steps_to_show]
        print(f"  seed {seed_idx}: {sample}")

    print(f"\n{calibration.upper()} last {steps_to_show} steps (per seed):")
    for seed_idx, trace in enumerate(traces):
        sample = trace[-steps_to_show:] if len(trace) >= steps_to_show else trace
        print(f"  seed {seed_idx}: {sample}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare California and Saudi calibration dynamics."
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    env_inputs_by_seed = [build_env_inputs(20, 20, seed) for seed in seeds]

    results = {}
    for calibration in ("california", "saudi"):
        traces = []
        for seed, env_inputs in zip(seeds, env_inputs_by_seed):
            trace = rollout(calibration, env_inputs, args.steps, args.debug)
            traces.append(trace)

        mean, std, avg = compute_stats(traces)
        stats = compute_trajectory_stats(traces)
        results[calibration] = {
            "traces": traces,
            "mean": mean,
            "std": std,
            "avg": avg,
            "stats": stats,
        }

        print(f"\n{'='*70}")
        print(f"{calibration.upper()} - Summary")
        print(f"{'='*70}")
        print_trajectory_sample(calibration, traces, steps_to_show=min(20, args.steps))
        print(f"\n{calibration.upper()} statistics:")
        print(f"  Average fire cells: {avg:.2f}")
        print(f"  Peak fire cells (mean ± std): {stats['peak_mean']:.1f} ± {stats['peak_std']:.1f}")
        print(f"  Time to peak (mean): {stats['peak_time_mean']:.1f} steps")
        print(f"  Final fire cells (mean ± std): {stats['final_mean']:.1f} ± {stats['final_std']:.1f}")

        if args.csv:
            csv_dir = Path(args.csv).parent
            cal_csv = csv_dir / f"{calibration}_trajectory.csv"
            write_csv(cal_csv, calibration, seeds, traces)
            print(f"\nWrote {calibration} trajectory to {cal_csv}")

    if args.csv:
        summary_csv = Path(args.csv)
        write_summary_csv(summary_csv, results, seeds)
        print(f"\nWrote combined summary to {summary_csv}")


if __name__ == "__main__":
    main()
