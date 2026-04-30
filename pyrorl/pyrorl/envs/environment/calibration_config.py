"""
Calibration configuration system for PyroRL environments.

Provides a structured ``EnvConfig`` dataclass and a ``get_config`` factory so
that every calibration-sensitive parameter lives in one inspectable place
instead of scattered ``if calibration == ...`` branches.

Usage::

    from pyrorl.envs.environment.calibration_config import get_config

    config = get_config("saudi")
    print(config.fuel_mean)   # 3.5
    print(config.wind_type)   # "shamal"
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EnvConfig:
    """All calibration-sensitive environment parameters."""

    # ── Fuel model ───────────────────────────────────────────────────────
    fuel_mean: float = 8.5
    fuel_stdev: float = 3.0
    fuel_burn_rate: float = 1.0

    # ── Fire propagation ─────────────────────────────────────────────────
    fire_propagation_rate: float = 0.094

    # ── Wind model ───────────────────────────────────────────────────────
    #   wind_type: "none" | "static" | "shamal"
    #   wind_params: type-specific parameters (e.g. base_speed for shamal)
    wind_type: str = "none"
    wind_params: Dict = field(default_factory=dict)

    # ── Terrain model ────────────────────────────────────────────────────
    #   terrain_type: "flat" | "dune"
    #   terrain_params: type-specific parameters for dune_profile()
    terrain_type: str = "flat"
    terrain_params: Dict = field(default_factory=dict)

    # ── Visibility / dust model ──────────────────────────────────────────
    #   visibility_radius: None means full observability
    visibility_radius: Optional[int] = None
    visibility_params: Dict = field(default_factory=lambda: {
        "dust_intensity": 0.0,
    })

    # ── Suppression tuning ───────────────────────────────────────────────
    suppression_strength: float = 0.55
    suppression_radius: int = 1
    suppression_extinguish_prob: float = 0.55

    # ── Oasis / clustered-fuel model ─────────────────────────────────────
    oasis_clusters: Dict = field(default_factory=lambda: {
        "enabled": False,
        "count_factor": 2.8,
        "radius_factor": 0.1,
        "peak": 8.0,
    })

    # ── Default reward weights ───────────────────────────────────────────
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "fire_delta": 1.0,
        "burning_cells": 0.10,
        "new_ignitions": 0.05,
        "newly_burned_population": 15.0,
        "burning_population": 3.0,
        "finished_evac": 8.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Pre-defined calibration profiles
# ─────────────────────────────────────────────────────────────────────────────

def _california_config() -> EnvConfig:
    """Mediterranean / chaparral wildfire profile."""
    return EnvConfig(
        # Fuel
        fuel_mean=8.5,
        fuel_stdev=3.0,
        fuel_burn_rate=1.0,
        # Fire
        fire_propagation_rate=0.094,
        # Wind
        wind_type="none",
        wind_params={},
        # Terrain
        terrain_type="flat",
        terrain_params={},
        # Visibility
        visibility_radius=None,
        visibility_params={"dust_intensity": 0.0},
        # Suppression
        suppression_strength=0.55,
        suppression_radius=1,
        suppression_extinguish_prob=0.55,
        # Oasis clusters disabled
        oasis_clusters={
            "enabled": False,
            "count_factor": 2.8,
            "radius_factor": 0.1,
            "peak": 8.0,
        },
        # Reward weights
        reward_weights={
            "fire_delta": 1.0,
            "burning_cells": 0.10,
            "new_ignitions": 0.05,
            "newly_burned_population": 15.0,
            "burning_population": 3.0,
            "finished_evac": 8.0,
        },
    )


def _saudi_config() -> EnvConfig:
    """Arid / desert wildfire profile with Shamal wind and dune terrain."""
    return EnvConfig(
        # Fuel — sparse background with clustered oasis patches
        fuel_mean=3.5,
        fuel_stdev=1.3,
        fuel_burn_rate=1.0,
        # Fire
        fire_propagation_rate=0.094,
        # Wind — Shamal regime (dynamic NW gusts)
        wind_type="shamal",
        wind_params={
            "base_speed": 22.0,
            "speed_stdev": 6.0,
            "base_angle_deg": 135.0,
            "angle_stdev_deg": 12.0,
            "gust_prob": 0.25,
            "gust_speed_range": (8.0, 18.0),
            "gust_angle_stdev_deg": 25.0,
        },
        # Terrain — dune ridges
        terrain_type="dune",
        terrain_params={
            "ridge_spacing": 12.0,
            "ridge_width": 3.5,
            "curvature": 0.02,
            "crescent_shift": 2.5,
            "amplitude": 1.0,
            "orientation_deg": 315.0,
            "corridor_gain": 0.45,
        },
        # Visibility — dust reduces observation radius
        visibility_radius=None,  # auto-computed from grid size
        visibility_params={"dust_intensity": 0.45},
        # Suppression — slightly stronger radius, weaker extinguish
        suppression_strength=0.6,
        suppression_radius=2,
        suppression_extinguish_prob=0.45,
        # Oasis clusters — patchy dense vegetation
        oasis_clusters={
            "enabled": True,
            "count_factor": 2.8,
            "radius_factor": 0.1,
            "peak": 8.0,
        },
        # Reward weights (same defaults)
        reward_weights={
            "fire_delta": 1.0,
            "burning_cells": 0.10,
            "new_ignitions": 0.05,
            "newly_burned_population": 15.0,
            "burning_population": 3.0,
            "finished_evac": 8.0,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

_CALIBRATION_REGISTRY: Dict[str, callable] = {
    "california": _california_config,
    "saudi": _saudi_config,
}


def get_config(calibration: str) -> EnvConfig:
    """
    Return the ``EnvConfig`` for the given calibration profile.

    Parameters
    ----------
    calibration : str
        One of ``"california"`` or ``"saudi"``.

    Returns
    -------
    EnvConfig
        A fresh copy of the config (safe to mutate).

    Raises
    ------
    ValueError
        If *calibration* is not a known profile.
    """
    factory = _CALIBRATION_REGISTRY.get(calibration)
    if factory is None:
        valid = ", ".join(sorted(_CALIBRATION_REGISTRY))
        raise ValueError(
            f"Unknown calibration {calibration!r}. Valid options: {valid}"
        )
    return factory()
""", "Description": "Structured EnvConfig dataclass with California and Saudi profiles, plus get_config() factory.", "IsArtifact": false, "Overwrite": false, "TargetFile": "c:\\Users\\Ali Akarma\\Desktop\\Github\\PyroRL\\pyrorl\\pyrorl\\envs\\environment\\calibration_config.py"}
