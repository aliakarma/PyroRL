"""
Scenario system for PyroRL evaluation experiments.

Each scenario modifies environment parameters AFTER the ``FireWorld`` is
constructed but BEFORE simulation begins, enabling controlled experimental
conditions without touching training logic.

Usage::

    from pyrorl.pyrorl.envs.environment.scenarios import apply_scenario

    # Called automatically by WildfireEvacuationEnv when scenario is set
    apply_scenario(env, "high_wind")

Available scenarios:
    - ``high_wind``       — increased wind magnitude and gust probability
    - ``low_fuel``        — reduced global fuel density
    - ``oasis_cluster``   — more fuel clusters with higher local peaks
    - ``multi_ignition``  — multiple initial fire ignition points
    - ``narrow_corridor`` — strip of high fuel surrounded by low fuel
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..pyrorl import WildfireEvacuationEnv

# Indices matching environment.py
FIRE_INDEX = 0
FUEL_INDEX = 1


# ─────────────────────────────────────────────────────────────────────────────
# Individual scenario implementations
# ─────────────────────────────────────────────────────────────────────────────

def _apply_high_wind(env: "WildfireEvacuationEnv") -> None:
    """Increase wind magnitude and gust probability.

    Modifies the underlying FireWorld's wind parameters to simulate
    extreme wind conditions (e.g. severe Shamal event).
    """
    fire_env = env.fire_env

    if fire_env.wind_model == "shamal":
        # Boost the Shamal base parameters for stronger persistent wind
        fire_env.wind_speed = max(fire_env.wind_speed or 0, 35.0)
        fire_env.wind_angle = fire_env.wind_angle or np.deg2rad(135.0)
    else:
        # For static/none wind models, set a strong static wind
        fire_env.wind_speed = 35.0
        fire_env.wind_angle = np.deg2rad(135.0)
        fire_env.wind_model = "static"

    # Re-apply wind transform with boosted speed
    from .environment_constant import linear_wind_transform
    import torch
    fire_env.fire_mask = torch.as_tensor(
        linear_wind_transform(fire_env.wind_speed, fire_env.wind_angle)
    )

    # Also update the wrapper-level wind for consistency across resets
    env.wind_speed = fire_env.wind_speed
    env.wind_angle = fire_env.wind_angle


def _apply_low_fuel(env: "WildfireEvacuationEnv") -> None:
    """Reduce global fuel density to simulate extremely arid conditions.

    Scales existing fuel map down by 40% and caps the maximum.
    """
    fire_env = env.fire_env
    fire_env.state_space[FUEL_INDEX] *= 0.6
    fire_env.state_space[FUEL_INDEX] = np.clip(
        fire_env.state_space[FUEL_INDEX], 0, 5.0
    )
    # Ensure burning cells still have enough fuel to burn
    fire_env._ensure_burning_cells_have_fuel()


def _apply_oasis_cluster(env: "WildfireEvacuationEnv") -> None:
    """Add extra dense fuel clusters (oasis-like vegetation patches).

    Creates 6-8 additional high-fuel clusters on top of the existing map,
    producing pockets of intense fire potential in an otherwise sparse landscape.
    """
    fire_env = env.fire_env
    num_rows, num_cols = fire_env.state_space.shape[1], fire_env.state_space.shape[2]

    num_clusters = max(6, int(min(num_rows, num_cols) / 1.5))
    cluster_radius = max(1.5, min(num_rows, num_cols) * 0.12)
    cluster_peak = 12.0

    rows = np.arange(num_rows)[:, None]
    cols = np.arange(num_cols)[None, :]

    for _ in range(num_clusters):
        cx = np.random.randint(0, num_rows)
        cy = np.random.randint(0, num_cols)
        dist2 = (rows - cx) ** 2 + (cols - cy) ** 2
        bump = cluster_peak * np.exp(-dist2 / (2 * (cluster_radius ** 2)))
        fire_env.state_space[FUEL_INDEX] += bump

    fire_env.state_space[FUEL_INDEX] = np.clip(
        fire_env.state_space[FUEL_INDEX], 0, None
    )


def _apply_multi_ignition(env: "WildfireEvacuationEnv") -> None:
    """Set multiple initial fire ignition points.

    Places 4 fires at spread-out positions across the grid to create
    a challenging multi-front scenario.
    """
    fire_env = env.fire_env
    num_rows, num_cols = fire_env.state_space.shape[1], fire_env.state_space.shape[2]

    # Clear existing fires
    fire_env.state_space[FIRE_INDEX] = 0

    # Place 4 fires in different quadrants to maximize challenge
    margin_r = max(1, num_rows // 5)
    margin_c = max(1, num_cols // 5)
    ignition_points = [
        (margin_r, margin_c),                               # top-left
        (margin_r, num_cols - 1 - margin_c),                # top-right
        (num_rows - 1 - margin_r, margin_c),                # bottom-left
        (num_rows - 1 - margin_r, num_cols - 1 - margin_c), # bottom-right
    ]

    for r, c in ignition_points:
        r = np.clip(r, 0, num_rows - 1)
        c = np.clip(c, 0, num_cols - 1)
        fire_env.state_space[FIRE_INDEX, r, c] = 1

    # Ensure all ignition cells have fuel
    fire_env._ensure_burning_cells_have_fuel()
    fire_env.prev_fire_cells = int(fire_env.state_space[FIRE_INDEX].sum())


def _apply_narrow_corridor(env: "WildfireEvacuationEnv") -> None:
    """Create a strip of high fuel surrounded by low fuel.

    Produces a narrow channel of dense vegetation running across the grid,
    forcing fire to propagate along a constrained corridor.
    """
    fire_env = env.fire_env
    num_rows, num_cols = fire_env.state_space.shape[1], fire_env.state_space.shape[2]

    # Set all fuel very low
    fire_env.state_space[FUEL_INDEX] = np.clip(
        np.random.normal(1.0, 0.5, (num_rows, num_cols)), 0.0, 2.0
    )

    # Create a high-fuel corridor (3 rows wide) through the center
    corridor_center = num_rows // 2
    corridor_half_width = 1
    row_lo = max(0, corridor_center - corridor_half_width)
    row_hi = min(num_rows, corridor_center + corridor_half_width + 1)

    fire_env.state_space[FUEL_INDEX, row_lo:row_hi, :] = np.clip(
        np.random.normal(10.0, 1.5, (row_hi - row_lo, num_cols)), 6.0, 15.0
    )

    # Ensure burning cells still have fuel
    fire_env._ensure_burning_cells_have_fuel()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario registry and public API
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_REGISTRY = {
    "high_wind": _apply_high_wind,
    "low_fuel": _apply_low_fuel,
    "oasis_cluster": _apply_oasis_cluster,
    "multi_ignition": _apply_multi_ignition,
    "narrow_corridor": _apply_narrow_corridor,
}

AVAILABLE_SCENARIOS = tuple(sorted(_SCENARIO_REGISTRY.keys()))


def apply_scenario(env: "WildfireEvacuationEnv", scenario_name: str) -> None:
    """Apply a named scenario to an initialised environment.

    Parameters
    ----------
    env : WildfireEvacuationEnv
        The gym wrapper (must already have ``fire_env`` constructed).
    scenario_name : str
        One of the keys in ``AVAILABLE_SCENARIOS``.

    Raises
    ------
    ValueError
        If *scenario_name* is not recognised.
    """
    fn = _SCENARIO_REGISTRY.get(scenario_name)
    if fn is None:
        valid = ", ".join(AVAILABLE_SCENARIOS)
        raise ValueError(
            f"Unknown scenario {scenario_name!r}. "
            f"Available scenarios: {valid}"
        )
    fn(env)

    if getattr(env, "debug", False):
        fire_env = env.fire_env
        fuel = fire_env.state_space[FUEL_INDEX]
        fire_cells = int(fire_env.state_space[FIRE_INDEX].sum())
        wind_spd = fire_env.wind_speed if fire_env.wind_speed is not None else 0.0
        print(
            f"[scenario] applied: {scenario_name} | "
            f"fuel min={fuel.min():.2f} mean={fuel.mean():.2f} max={fuel.max():.2f} | "
            f"fire_cells={fire_cells} | wind_speed={wind_spd:.1f}"
        )
