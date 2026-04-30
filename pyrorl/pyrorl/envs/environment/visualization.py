"""
Visualization helpers for PyroRL environments.

Standalone functions for generating paper-ready figures of fuel maps,
terrain profiles, and wind fields.  These are side-effect-free and
do **not** import or modify any environment state.

Usage::

    from pyrorl.envs.environment.visualization import (
        plot_fuel_map,
        plot_terrain,
        plot_wind_field,
    )

    plot_fuel_map(fuel_grid, title="Saudi Fuel Distribution", save_path="fuel.png")
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def plot_fuel_map(
    fuel_grid: np.ndarray,
    title: str = "Fuel Map",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 6),
    cmap: str = "YlGn",
) -> None:
    """Heatmap of the fuel distribution across the grid.

    Parameters
    ----------
    fuel_grid : np.ndarray
        2-D array of fuel values (rows × cols).
    title : str
        Figure title.
    save_path : str or None
        If given, save the figure to this path instead of showing it.
    figsize : tuple
        Figure dimensions in inches.
    cmap : str
        Matplotlib colormap name.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(fuel_grid, origin="upper", cmap=cmap, aspect="equal")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fuel level")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Fuel map saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_terrain(
    terrain_grid: np.ndarray,
    title: str = "Terrain Elevation",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 6),
    cmap: str = "terrain",
) -> None:
    """Contour / heatmap of terrain elevation (e.g. dune profile).

    Parameters
    ----------
    terrain_grid : np.ndarray
        2-D array of normalised elevation values (rows × cols).
    title : str
        Figure title.
    save_path : str or None
        If given, save the figure to this path.
    figsize : tuple
        Figure dimensions in inches.
    cmap : str
        Matplotlib colormap name.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(terrain_grid, origin="upper", cmap=cmap, aspect="equal")
    # Overlay contour lines for readability
    contour = ax.contour(
        terrain_grid, levels=8, colors="k", linewidths=0.5, alpha=0.4
    )
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.2f")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation (normalised)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Terrain map saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_wind_field(
    wind_speed: float,
    wind_angle: float,
    grid_shape: Tuple[int, int] = (10, 10),
    title: str = "Wind Field",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 6),
) -> None:
    """Quiver plot of a uniform wind vector across the grid.

    Parameters
    ----------
    wind_speed : float
        Magnitude of the wind (arbitrary units).
    wind_angle : float
        Direction of the wind in **radians** (0 = East, π/2 = North).
    grid_shape : tuple
        (rows, cols) of the grid to visualise.
    title : str
        Figure title.
    save_path : str or None
        If given, save the figure to this path.
    figsize : tuple
        Figure dimensions in inches.
    """
    import matplotlib.pyplot as plt

    rows, cols = grid_shape
    Y, X = np.mgrid[0:rows, 0:cols]
    U = np.full_like(X, wind_speed * np.cos(wind_angle), dtype=float)
    V = np.full_like(Y, wind_speed * np.sin(wind_angle), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(X, Y, U, V, scale=wind_speed * max(rows, cols) * 1.2, color="#2196F3")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(
        f"{title}  (speed={wind_speed:.1f}, angle={np.rad2deg(wind_angle):.1f}°)",
        fontsize=13,
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Wind field saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
