#!/usr/bin/env python3
"""
PyroRL Episode Visualization — Publication-Quality Renderer

Modes:
  Single:  --model X --calibration Y --scenario Z --out path.gif
  Compare: --compare --ca_model X --sa_model Y --scenario Z
  Batch:   --compare --all_scenarios --ca_model X --sa_model Y
  Annotate: add --annotate to any mode for overlay annotations

Outputs per scenario (in logs/<scenario>/ when batch, logs/ otherwise):
  - compare_<scenario>.gif           (animated side-by-side comparison)
  - compare_<scenario>_t{1,20,50,100}.png  (high-DPI key-frame snapshots)
  - caption.txt                      (auto-generated academic caption)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import imageio.v2 as imageio
from stable_baselines3 import PPO
from scipy import ndimage

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv
from pyrorl.pyrorl.envs.environment.environment import (
    FIRE_INDEX, FUEL_INDEX, POPULATED_INDEX, EVACUATING_INDEX, PATHS_INDEX,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SCENARIOS = [
    "high_wind", "low_fuel", "oasis_cluster", "multi_ignition", "narrow_corridor",
    "extreme_wind", "fuel_depletion", "random_terrain", "delayed_ignition", "dense_population",
]

CELL_COLORS = ['#06d6a0', '#ffd166', '#ef476f', '#073b4c', '#118ab2', '#bf9aca', '#555555']
CELL_CMAP = ListedColormap(CELL_COLORS)

KEY_FRAMES = [1, 20, 50, 100]

CAPTION_TEMPLATES = {
    "high_wind":        "Under elevated wind conditions, {ca_desc}. {sa_desc}.",
    "low_fuel":         "With reduced fuel density, {ca_desc}. {sa_desc}.",
    "oasis_cluster":    "Facing concentrated fuel clusters, {ca_desc}. {sa_desc}.",
    "multi_ignition":   "With multiple simultaneous ignition points, {ca_desc}. {sa_desc}.",
    "narrow_corridor":  "In a narrow high-fuel corridor, {ca_desc}. {sa_desc}.",
    "extreme_wind":     "Under extreme wind with severe gusts, {ca_desc}. {sa_desc}.",
    "fuel_depletion":   "With rapidly depleting fuel, {ca_desc}. {sa_desc}.",
    "random_terrain":   "On irregular terrain with unpredictable fire channels, {ca_desc}. {sa_desc}.",
    "delayed_ignition": "With a 15-step ignition delay, {ca_desc}. {sa_desc}.",
    "dense_population": "In a densely populated grid requiring active suppression, {ca_desc}. {sa_desc}.",
}

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def create_eval_environment(calibration: str = "california", scenario: str = None):
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
            "fire_delta": 1.0, "burning_cells": 0.20, "new_ignitions": 0.10,
            "newly_burned_population": 0.0, "burning_population": 0.0, "finished_evac": 0.0,
        },
        "skip": True,
    }

    if calibration == "saudi":
        env_kwargs["wind_speed"] = 6.0
        env_kwargs["wind_angle"] = np.deg2rad(135.0)

    if scenario is not None and scenario != "none":
        env_kwargs["scenario"] = scenario

    return PyroRLEnv(**env_kwargs)

# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def _build_grid(env):
    """Convert environment state to a categorical grid for visualization."""
    state = env.fire_env.get_state()
    finished = env.fire_env.get_finished_evacuating()
    rows, cols = state.shape[1], state.shape[2]

    # 0=Grass, 1=Path, 2=Fire, 3=Populated, 4=Evacuating, 5=Finished, 6=Burned
    grid = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if state[FUEL_INDEX][r][c] <= 0 and state[FIRE_INDEX][r][c] == 0:
                grid[r, c] = 6  # Burned out
            if state[PATHS_INDEX][r][c] > 0:
                grid[r, c] = 1
            if state[FIRE_INDEX][r][c] == 1:
                grid[r, c] = 2
            if state[POPULATED_INDEX][r][c] == 1:
                grid[r, c] = 3
            if state[EVACUATING_INDEX][r][c] > 0:
                grid[r, c] = 4
            if [r, c] in finished:
                grid[r, c] = 5
    return grid

# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _find_largest_fire_cluster(grid):
    """Return bounding box (r_min, r_max, c_min, c_max) of the largest fire cluster."""
    fire_mask = (grid == 2).astype(int)
    if fire_mask.sum() == 0:
        return None
    labeled, n_features = ndimage.label(fire_mask)
    if n_features == 0:
        return None
    largest = max(range(1, n_features + 1), key=lambda i: (labeled == i).sum())
    cluster = np.argwhere(labeled == largest)
    r_min, c_min = cluster.min(axis=0)
    r_max, c_max = cluster.max(axis=0)
    return r_min, r_max, c_min, c_max

# ---------------------------------------------------------------------------
# Axis drawing (shared by single and compare modes)
# ---------------------------------------------------------------------------

def _draw_axis(ax, env, grid, title_str, annotate=False, policy_label=None, scenario=None, step=None):
    """Draw a single grid onto an axis with all overlays."""
    rows, cols = grid.shape

    ax.imshow(grid, cmap=CELL_CMAP, vmin=0, vmax=6)
    ax.set_aspect("equal")

    # Thin white gridlines between cells
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Suppression hatches — reduced density and lower alpha for readability
    supp_mask = env.fire_env.suppression_mask
    for r in range(rows):
        for c in range(cols):
            if supp_mask[r, c] > 0:
                rect = Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    fill=False, hatch="//", edgecolor="black", alpha=0.35, linewidth=0,
                )
                ax.add_patch(rect)

    # Wind direction — thicker arrow, dark blue, labeled with angle + speed
    wind_angle = env.fire_env.wind_angle
    wind_speed = env.fire_env.wind_speed
    if wind_speed and wind_speed > 0 and wind_angle is not None:
        angle_deg = np.degrees(wind_angle) % 360
        dx = np.cos(wind_angle) * 1.2
        dy = np.sin(wind_angle) * 1.2
        ax.annotate(
            "", xy=(cols / 2 + dx, rows + 0.8 + dy), xytext=(cols / 2, rows + 0.8),
            arrowprops=dict(arrowstyle="-|>", color="#1a3a5c", lw=3),
        )
        ax.text(
            cols / 2 - 2.5, rows + 1.8,
            f"Wind Direction ({angle_deg:.0f}°, {wind_speed:.1f})",
            fontsize=9, color="#1a3a5c", fontweight="bold",
        )

    # Annotation overlays
    if annotate:
        # Largest fire cluster bounding box
        bbox = _find_largest_fire_cluster(grid)
        if bbox:
            r_min, r_max, c_min, c_max = bbox
            rect = Rectangle(
                (c_min - 0.6, r_min - 0.6),
                c_max - c_min + 1.2, r_max - r_min + 1.2,
                fill=False, edgecolor="#ef476f", linewidth=2, linestyle="--",
            )
            ax.add_patch(rect)

        # Star markers on population cells
        state = env.fire_env.get_state()
        pop_cells = np.argwhere(state[POPULATED_INDEX] == 1)
        for r, c in pop_cells:
            ax.plot(c, r, marker="*", color="#ffd166", markersize=14, markeredgecolor="black", markeredgewidth=0.5)

        # Text overlay box
        info_lines = []
        if policy_label:
            info_lines.append(f"Policy: {policy_label}")
        if step is not None:
            info_lines.append(f"Timestep: t={step}")
        if scenario:
            info_lines.append(f"Scenario: {scenario}")
        if info_lines:
            ax.text(
                0.02, 0.98, "\n".join(info_lines),
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ccc"),
            )

    ax.set_title(title_str, fontsize=13, fontweight="bold", pad=12)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def _get_legend_elements():
    """Consistent-order legend matching the colormap indices."""
    return [
        Patch(facecolor="#06d6a0", label="Unburned / Grass"),
        Patch(facecolor="#555555", label="Burned Out"),
        Patch(facecolor="#ef476f", label="Fire"),
        Patch(facecolor="#073b4c", label="Population"),
        Patch(facecolor="#ffd166", label="Path"),
        Patch(facecolor="#118ab2", label="Evacuating"),
        Patch(facecolor="#bf9aca", label="Finished Evac"),
        Patch(facecolor="none", hatch="//", edgecolor="black", label="Suppression Active"),
    ]

# ---------------------------------------------------------------------------
# Single-model rendering
# ---------------------------------------------------------------------------

def render_single(env, step, output_dir, title_str, annotate=False, policy_label=None, scenario=None,
                  save_key_frame=False, key_frame_path=None):
    grid = _build_grid(env)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    _draw_axis(ax, env, grid, title_str + f" (t={step})", annotate, policy_label, scenario, step)
    ax.legend(handles=_get_legend_elements(), loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11,
              borderaxespad=0.5, framealpha=0.9)
    plt.tight_layout()

    frame_path = os.path.join(output_dir, f"frame_{step:03d}.png")
    plt.savefig(frame_path, dpi=100)
    if save_key_frame and key_frame_path:
        plt.savefig(key_frame_path, dpi=200)

    plt.close()
    return frame_path

# ---------------------------------------------------------------------------
# Comparative rendering
# ---------------------------------------------------------------------------

def render_compare(env_ca, env_sa, step, output_dir, scenario, annotate=False,
                   save_key_frame=False, key_frame_path=None):
    grid_ca = _build_grid(env_ca)
    grid_sa = _build_grid(env_sa)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    _draw_axis(axes[0], env_ca, grid_ca, f"California Policy (t={step})",
               annotate, "California", scenario, step)
    _draw_axis(axes[1], env_sa, grid_sa, f"Saudi Policy (t={step})",
               annotate, "Saudi", scenario, step)

    fig.suptitle(f"Scenario: {scenario}", fontsize=18, fontweight="bold")
    axes[1].legend(handles=_get_legend_elements(), loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=11, borderaxespad=0.5, framealpha=0.9)
    plt.tight_layout()

    frame_path = Path(output_dir) / f"frame_{step:03d}.png"
    plt.savefig(frame_path, dpi=100)
    if save_key_frame and key_frame_path:
        plt.savefig(key_frame_path, dpi=200)

    plt.close()
    return frame_path

# ---------------------------------------------------------------------------
# Caption generator
# ---------------------------------------------------------------------------

def generate_caption(scenario, env_ca, env_sa, out_dir):
    """Write an auto-generated academic caption based on final environment states."""
    state_ca = env_ca.fire_env.get_state()
    state_sa = env_sa.fire_env.get_state()

    fire_ca = int(state_ca[FIRE_INDEX].sum())
    fire_sa = int(state_sa[FIRE_INDEX].sum())
    burned_ca = int(np.logical_and(state_ca[FUEL_INDEX] <= 0, state_ca[FIRE_INDEX] == 0).sum()) + fire_ca
    burned_sa = int(np.logical_and(state_sa[FUEL_INDEX] <= 0, state_sa[FIRE_INDEX] == 0).sum()) + fire_sa
    supp_ca = int(env_ca.fire_env.suppression_mask.sum() > 0)
    supp_sa = int(env_sa.fire_env.suppression_mask.sum() > 0)

    # Build descriptive fragments
    if burned_ca > burned_sa:
        ca_desc = f"the California-trained policy exhibited higher fire spread ({burned_ca} burned cells), indicating weaker containment"
        sa_desc = f"the Saudi-trained policy achieved better containment ({burned_sa} burned cells), demonstrating stronger generalization to this distribution shift"
    elif burned_sa > burned_ca:
        ca_desc = f"the California-trained policy contained fire more effectively ({burned_ca} burned cells)"
        sa_desc = f"the Saudi-trained policy experienced greater spread ({burned_sa} burned cells), suggesting reduced adaptability"
    else:
        ca_desc = f"the California-trained policy and Saudi-trained policy achieved comparable containment ({burned_ca} burned cells each)"
        sa_desc = "no significant behavioral divergence was observed"

    template = CAPTION_TEMPLATES.get(scenario, "Scenario: {scenario}. {ca_desc}. {sa_desc}.")
    caption = f"Scenario: {scenario}. " + template.format(ca_desc=ca_desc, sa_desc=sa_desc)

    caption_path = Path(out_dir) / "caption.txt"
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    print(f"  Caption saved to {caption_path}")

# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_single(args, temp_dir):
    model_name = os.path.basename(args.model).replace(".zip", "")
    title_str = f"Scenario: {args.scenario} | Model: {model_name}"

    print(f"Initializing {args.calibration} environment with scenario: {args.scenario}")
    env = create_eval_environment(args.calibration, args.scenario)

    print(f"Loading model from {args.model}")
    model = PPO.load(args.model, device="auto")

    obs, _ = env.reset(seed=args.seed)
    frames = []

    done = False
    step = 0
    while not done:
        save_key = step in KEY_FRAMES
        key_frame_path = None
        if save_key:
            key_frame_path = os.path.join("logs", f"frame_{args.scenario}_{model_name}_t{step}.png")

        frame_path = render_single(
            env, step, temp_dir, title_str,
            annotate=args.annotate, policy_label=model_name, scenario=args.scenario,
            save_key_frame=save_key, key_frame_path=key_frame_path,
        )
        frames.append(imageio.imread(frame_path))

        if step > 0:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        step += 1

    print(f"Episode finished in {step} steps. Generating GIF...")
    out_gif = args.out if args.out else f"logs/fire_{args.scenario}_{model_name}.gif"
    imageio.mimsave(out_gif, frames, duration=200, loop=0)
    print(f"Successfully saved visualization GIF to {out_gif}")


def run_compare(ca_model_path, sa_model_path, scenario, output_dir, temp_dir, annotate=False, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-'*60}")
    print(f"  Scenario: {scenario}")
    print(f"{'-'*60}")

    env_ca = create_eval_environment("california", scenario)
    env_sa = create_eval_environment("saudi", scenario)

    model_ca = PPO.load(ca_model_path, device="auto")
    model_sa = PPO.load(sa_model_path, device="auto")

    obs_ca, _ = env_ca.reset(seed=seed)
    obs_sa, _ = env_sa.reset(seed=seed)

    frames = []
    done_ca = False
    done_sa = False
    step = 0

    while not (done_ca and done_sa):
        save_key = step in KEY_FRAMES
        key_frame_path = None
        if save_key:
            key_frame_path = output_dir / f"compare_t{step}.png"

        frame_path = render_compare(
            env_ca, env_sa, step, temp_dir, scenario,
            annotate=annotate, save_key_frame=save_key, key_frame_path=key_frame_path,
        )
        frames.append(imageio.imread(frame_path))

        if step > 0:
            if not done_ca:
                action_ca, _ = model_ca.predict(obs_ca, deterministic=True)
                obs_ca, _, terminated_ca, truncated_ca, _ = env_ca.step(int(action_ca))
                done_ca = terminated_ca or truncated_ca

            if not done_sa:
                action_sa, _ = model_sa.predict(obs_sa, deterministic=True)
                obs_sa, _, terminated_sa, truncated_sa, _ = env_sa.step(int(action_sa))
                done_sa = terminated_sa or truncated_sa

        step += 1
        if step > 150:
            break

    print(f"  Episode finished in {step} steps. Generating GIF...")
    out_gif = output_dir / "compare.gif"
    imageio.mimsave(out_gif, frames, duration=200, loop=0)
    print(f"  GIF saved to {out_gif}")

    # Auto-generate caption
    generate_caption(scenario, env_ca, env_sa, output_dir)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyroRL Episode Visualization (Publication Quality)")
    parser.add_argument("--compare", action="store_true", help="Side-by-side CA vs SA comparison")
    parser.add_argument("--annotate", action="store_true", help="Enable annotation overlays")
    parser.add_argument("--ca_model", help="Path to California model (.zip)")
    parser.add_argument("--sa_model", help="Path to Saudi model (.zip)")
    parser.add_argument("--model", help="Path to model (.zip) for single mode")
    parser.add_argument("--calibration", choices=["california", "saudi"], help="Calibration for single mode")
    parser.add_argument("--scenario", required=True, help="Scenario to apply (required)")
    parser.add_argument("--out", help="Output gif path for single mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    if args.compare:
        if not args.ca_model or not args.sa_model:
            print("Error: --compare requires --ca_model and --sa_model")
            sys.exit(1)

        output_dir = Path("logs") / args.scenario
        run_compare(
            ca_model_path=args.ca_model,
            sa_model_path=args.sa_model,
            scenario=args.scenario,
            output_dir=output_dir,
            temp_dir=temp_dir,
            annotate=args.annotate,
            seed=args.seed
        )
    else:
        if not args.model or not args.calibration:
            print("Error: single mode requires --model and --calibration")
            sys.exit(1)
        run_single(args, temp_dir)

    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
