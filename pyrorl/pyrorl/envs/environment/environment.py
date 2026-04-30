"""
Environment for Wildfire Spread
"""

import numpy as np
import random
import torch
from typing import Optional, Any, Tuple, Dict, List

# For wind bias
from .environment_constant import set_fire_mask, linear_wind_transform
from .calibration_config import get_config

"""
Indices corresponding to each layer of state
"""
FIRE_INDEX = 0
FUEL_INDEX = 1
POPULATED_INDEX = 2
EVACUATING_INDEX = 3
PATHS_INDEX = 4


class FireWorld:
    """
    We represent the world as a 5 by n by m tensor:
    - n by m is the size of the grid world,
    - 5 represents each of the following:
        - [fire, fuel, populated_areas, evacuating, paths]
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        populated_areas: np.ndarray,
        paths: np.ndarray,
        paths_to_pops: dict,
        num_fire_cells: int = 2,
        custom_fire_locations: Optional[np.ndarray] = None,
        wind_speed: Optional[float] = None,
        wind_angle: Optional[float] = None,
        fuel_mean: Optional[float] = None,
        fuel_stdev: Optional[float] = None,
        fire_propagation_rate: Optional[float] = None,
        calibration: str = "california",
        fuel_burn_rate: Optional[float] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        terminate_on_population_loss: bool = True,
        debug: bool = False,
    ):
        """
        The constructor defines the state and action space, initializes the fires,
        and sets the paths and populated areas.
        - wind angle is in radians
        """
        # Assert that number of rows, columns, and fire cells are both positive
        if num_rows < 1:
            raise ValueError("Number of rows should be positive!")
        if num_cols < 1:
            raise ValueError("Number of rows should be positive!")
        if num_fire_cells < 1:
            raise ValueError("Number of fire cells should be positive!")

        # Calibration controls default parameters and models
        if calibration not in {"california", "saudi"}:
            raise ValueError("Calibration must be either 'california' or 'saudi'")
        self.calibration = calibration
        config = get_config(calibration)

        # Resolve defaults from config — user overrides take precedence
        if fuel_mean is None:
            fuel_mean = config.fuel_mean
        if fuel_stdev is None:
            fuel_stdev = config.fuel_stdev
        if fire_propagation_rate is None:
            fire_propagation_rate = config.fire_propagation_rate
        if fuel_burn_rate is None:
            fuel_burn_rate = config.fuel_burn_rate
        self.fuel_burn_rate = fuel_burn_rate
        self.debug = debug
        self.terminate_on_population_loss = terminate_on_population_loss

        # Check that populated areas are within the grid
        valid_populated_areas = (
            (populated_areas[:, 0] >= 0)
            & (populated_areas[:, 1] >= 0)
            & (populated_areas[:, 0] < num_rows)
            & (populated_areas[:, 1] < num_cols)
        )
        if np.any(~valid_populated_areas):
            raise ValueError("Populated areas are not valid with the grid dimensions")

        # Check that each path has squares within the grid
        valid_paths = [
            (
                (np.array(path)[:, 0] >= 0)
                & (np.array(path)[:, 1] >= 0)
                & (np.array(path)[:, 0] < num_rows)
                & (np.array(path)[:, 1] < num_cols)
            )
            for path in paths
        ]
        if np.any(~np.hstack(valid_paths)):
            raise ValueError("Pathed areas are not valid with the grid dimensions")

        # Define the state and action space
        self.reward = 0
        self.state_space = np.zeros([5, num_rows, num_cols])
        self.suppression_mask = np.zeros((num_rows, num_cols), dtype=np.float32)
        self.suppression_decay = 0.85
        self.suppression_strength = config.suppression_strength
        self.suppression_radius = config.suppression_radius
        # Besides reducing future spread probability, suppression can actively extinguish
        # existing burning cells in the treated zone. Without this, actions can be too
        # weak relative to stochastic spread for PPO to learn from.
        self.suppression_extinguish_prob = config.suppression_extinguish_prob
        self.last_finished_evac_count = 0
        self.last_new_ignitions = 0
        self.reward_weights = reward_weights or config.reward_weights
        self.last_reward_components = {
            "fire_delta": 0.0,
            "fire_cells": 0,
            "newly_burned": 0,
            "burning_population": 0,
            "finished_evac": 0,
            "new_ignitions": 0,
        }

        # Set up actions -- add extra action for doing nothing
        num_paths, num_actions = np.arange(len(paths)), 0
        for key in paths_to_pops:

            # First, check that path index actually exists
            if not np.isin(key, num_paths):
                raise ValueError("Key is not a valid index of a path!")

            # Then, check that each populated area exists
            areas = np.array(paths_to_pops[key])
            if np.any(~np.isin(areas, populated_areas)):
                raise ValueError("Corresponding populated area does not exist!")

            # Increment total number of actions to be taken
            for _ in range(len(paths_to_pops[key])):
                num_actions += 1
        self.actions = list(np.arange(num_actions + 1))

        # We want to remember which action index corresponds to which population center
        # and which path (because we just provide an array like [1,2,3,4,5,6,7]) which
        # would each be mapped to a given population area taking a given path
        self.action_to_pop_and_path: dict[Any, Optional[Tuple[Any, Any]]] = {
            self.actions[-1]: None
        }

        # Map each action to a populated area and path
        index = 0
        for path in paths_to_pops:
            for pop in paths_to_pops[path]:
                self.action_to_pop_and_path[index] = (pop, path)
                index += 1

        # State for the evacuation of populated areas
        self.evacuating_paths: Dict[int, list] = (
            {}
        )  # path_index : list of pop x,y indices that are evacuating [[x,y],[x,y],...]
        self.evacuating_timestamps = np.full((num_rows, num_cols), np.inf)

        # If the user specifies custom fire locations, set them
        self.num_fire_cells = num_fire_cells
        if custom_fire_locations is not None:

            # Check that populated areas are within the grid
            valid_fire_locations = (
                (custom_fire_locations[:, 0] >= 0)
                & (custom_fire_locations[:, 1] >= 0)
                & (custom_fire_locations[:, 0] < num_rows)
                & (custom_fire_locations[:, 1] < num_cols)
            )
            if np.any(~valid_fire_locations):
                raise ValueError(
                    "Populated areas are not valid with the grid dimensions"
                )

            # Only once valid, set them!
            fire_rows = custom_fire_locations[:, 0]
            fire_cols = custom_fire_locations[:, 1]
            self.state_space[FIRE_INDEX, fire_rows, fire_cols] = 1

        # Otherwise, randomly generate them
        else:
            for _ in range(self.num_fire_cells):
                self.state_space[
                    FIRE_INDEX,
                    random.randint(0, num_rows - 1),
                    random.randint(0, num_cols - 1),
                ] = 1

        # Initialize fuel levels
        self.state_space[FUEL_INDEX] = self._initialize_fuel_map(
            num_rows, num_cols, fuel_mean, fuel_stdev
        )
        self._ensure_burning_cells_have_fuel()

        # Initialize populated areas
        pop_rows, pop_cols = populated_areas[:, 0], populated_areas[:, 1]
        self.state_space[POPULATED_INDEX, pop_rows, pop_cols] = 1
        self.prev_fire_cells = int(self.state_space[FIRE_INDEX].sum())

        # Initialize self.paths
        self.paths: List[List[Any]] = []
        for path in paths:
            path_array = np.array(path)
            path_rows, path_cols = path_array[:, 0].astype(int), path_array[
                :, 1
            ].astype(int)
            self.state_space[PATHS_INDEX, path_rows, path_cols] += 1

            # Each path in self.paths is a list that records what the path is and
            # whether the path still exists (i.e. has not been destroyed by a fire)
            self.paths.append([np.zeros((num_rows, num_cols)), True])
            self.paths[-1][0][path_rows, path_cols] += 1

        # Set the timestep
        self.time_step = 0

        # Set terrain influence (dunes for Saudi calibration)
        self.terrain_spread_factor = None
        if calibration == "saudi":
            terrain = self._build_dune_terrain(num_rows, num_cols)
            self.terrain_spread_factor = self._build_terrain_spread_factor(terrain)

        # Set fire mask
        self.fire_mask = set_fire_mask(fire_propagation_rate)

        # Factor in wind speeds or Shamal wind regime
        self.wind_speed = wind_speed
        self.wind_angle = wind_angle
        self.wind_model = "none"
        if calibration == "saudi" and wind_speed is None and wind_angle is None:
            self.wind_model = "shamal"
            self._update_shamal_wind_mask()
        elif wind_speed is not None or wind_angle is not None:
            if wind_speed is None or wind_angle is None:
                raise TypeError(
                    "When setting wind details, "
                    "wind speed and wind angle must both be provided"
                )
            self.wind_model = "static"
            self.fire_mask = torch.as_tensor(
                linear_wind_transform(wind_speed, wind_angle)
            )
        else:
            self.fire_mask = torch.from_numpy(self.fire_mask)

        # Record which population cells have finished evacuating
        self.finished_evacuating_cells: list[list[int]] = []

        if self.debug and self.calibration == "saudi":
            self._log_fuel_stats("init")

    def _initialize_fuel_map(
        self, num_rows: int, num_cols: int, fuel_mean: float, fuel_stdev: float
    ) -> np.ndarray:
        base_fuel = np.random.normal(fuel_mean, fuel_stdev, (num_rows, num_cols))
        base_fuel = np.clip(base_fuel, 0, None)

        if self.calibration == "saudi":
            cluster_map = self._build_oasis_clusters(num_rows, num_cols)
            sparsity = np.random.random((num_rows, num_cols))
            background_scale = np.where(sparsity > 0.5, 0.8, 0.35)
            background = base_fuel * background_scale
            base_fuel = background + cluster_map

        return np.clip(base_fuel, 0, None)

    def _ensure_burning_cells_have_fuel(self) -> None:
        burning_cells = np.where(self.state_space[FIRE_INDEX] == 1)
        if burning_cells[0].size == 0:
            return

        min_fuel = max(1.0, self.fuel_burn_rate * 1.5)
        current_fuel = self.state_space[FUEL_INDEX, burning_cells[0], burning_cells[1]]
        self.state_space[FUEL_INDEX, burning_cells[0], burning_cells[1]] = np.maximum(
            current_fuel, min_fuel
        )

    def _apply_suppression_zone(
        self, center_row: int, center_col: int, radius: Optional[int] = None
    ) -> None:
        if radius is None:
            radius = self.suppression_radius

        row_min = max(0, center_row - radius)
        row_max = min(self.state_space.shape[1] - 1, center_row + radius)
        col_min = max(0, center_col - radius)
        col_max = min(self.state_space.shape[2] - 1, center_col + radius)
        self.suppression_mask[row_min : row_max + 1, col_min : col_max + 1] = np.maximum(
            self.suppression_mask[row_min : row_max + 1, col_min : col_max + 1],
            1.0,
        )

        # Active extinguishing: probabilistically clear burning cells in the treated zone.
        zone_fire = self.state_space[FIRE_INDEX, row_min : row_max + 1, col_min : col_max + 1]
        if np.any(zone_fire == 1):
            rng = np.random.random(zone_fire.shape)
            extinguish = (zone_fire == 1) & (rng < self.suppression_extinguish_prob)
            if np.any(extinguish):
                zone_fire = zone_fire.copy()
                zone_fire[extinguish] = 0
                self.state_space[FIRE_INDEX, row_min : row_max + 1, col_min : col_max + 1] = zone_fire

    def _apply_action_suppression(self, pop_cell: tuple, path_index: int) -> None:
        pop_row, pop_col = int(pop_cell[0]), int(pop_cell[1])
        self._apply_suppression_zone(pop_row, pop_col, radius=self.suppression_radius)

        path_rows, path_cols = np.where(self.paths[path_index][0] > 0)
        for row, col in zip(path_rows.tolist(), path_cols.tolist()):
            self._apply_suppression_zone(int(row), int(col), radius=1)

    def _build_oasis_clusters(self, num_rows: int, num_cols: int) -> np.ndarray:
        cluster_count = max(4, int(min(num_rows, num_cols) / 2.8))
        cluster_radius = max(2.0, min(num_rows, num_cols) * 0.1)
        cluster_peak = 8.0

        rows = np.arange(num_rows)[:, None]
        cols = np.arange(num_cols)[None, :]
        cluster_map = np.zeros((num_rows, num_cols), dtype=float)

        for _ in range(cluster_count):
            center_row = np.random.randint(0, num_rows)
            center_col = np.random.randint(0, num_cols)
            dist2 = (rows - center_row) ** 2 + (cols - center_col) ** 2
            cluster_map += cluster_peak * np.exp(
                -dist2 / (2 * (cluster_radius**2))
            )

        return cluster_map

    def _log_fuel_stats(self, stage: str) -> None:
        fuel = self.state_space[FUEL_INDEX]
        print(
            "[Saudi debug] stage="
            + stage
            + " fuel_min="
            + f"{fuel.min():.3f}"
            + " fuel_mean="
            + f"{fuel.mean():.3f}"
            + " fuel_max="
            + f"{fuel.max():.3f}"
        )

    def _log_spread_stats(
        self, step: int, spread: torch.Tensor, burning_before: int, burning_after: int
    ) -> None:
        if step % 5 != 0:
            return

        spread_min = float(spread.min().item())
        spread_mean = float(spread.mean().item())
        spread_max = float(spread.max().item())
        wind_speed = self.wind_speed if self.wind_speed is not None else 0.0
        wind_angle = self.wind_angle if self.wind_angle is not None else 0.0
        print(
            "[Saudi debug] step="
            + str(step)
            + " burning="
            + str(burning_before)
            + "->"
            + str(burning_after)
            + " spread_min="
            + f"{spread_min:.4f}"
            + " spread_mean="
            + f"{spread_mean:.4f}"
            + " spread_max="
            + f"{spread_max:.4f}"
            + " wind_speed="
            + f"{wind_speed:.2f}"
            + " wind_angle="
            + f"{wind_angle:.3f}"
        )

    @staticmethod
    def dune_profile(
        x: np.ndarray,
        y: np.ndarray,
        ridge_spacing: float = 12.0,
        ridge_width: float = 3.5,
        curvature: float = 0.02,
        crescent_shift: float = 2.5,
        amplitude: float = 1.0,
        orientation: float = np.deg2rad(315.0),
    ) -> np.ndarray:
        cos_a = np.cos(orientation)
        sin_a = np.sin(orientation)
        u = x * cos_a + y * sin_a
        v = -x * sin_a + y * cos_a

        phase = (u % ridge_spacing) - ridge_spacing / 2.0
        curved_phase = phase - curvature * (v**2)

        ridge = np.exp(-(curved_phase**2) / (2 * (ridge_width**2)))
        crescent = ridge - 0.6 * np.exp(
            -((curved_phase + crescent_shift) ** 2)
            / (2 * ((ridge_width * 0.9) ** 2))
        )
        return amplitude * np.maximum(crescent, 0.0)

    def _build_dune_terrain(self, num_rows: int, num_cols: int) -> np.ndarray:
        x_coords, y_coords = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
        elevation = self.dune_profile(x_coords, y_coords)
        elevation = (elevation - elevation.min()) / (
            elevation.max() - elevation.min() + 1e-6
        )
        return elevation

    def _build_terrain_spread_factor(
        self, elevation: np.ndarray, corridor_gain: float = 0.45
    ) -> torch.Tensor:
        elevation = (elevation - elevation.min()) / (
            elevation.max() - elevation.min() + 1e-6
        )
        factor = 1.0 + (1.0 - elevation) * corridor_gain
        return torch.from_numpy(factor.astype(np.float32))

    def _sample_shamal_wind(self) -> Tuple[float, float]:
        base_speed = 22.0
        speed = max(0.0, np.random.normal(base_speed, 6.0))
        angle = np.deg2rad(135.0) + np.random.normal(0.0, np.deg2rad(12.0))

        if np.random.random() < 0.25:
            speed += np.random.uniform(8.0, 18.0)
            angle += np.random.normal(0.0, np.deg2rad(25.0))

        return speed, angle

    def _update_shamal_wind_mask(self) -> None:
        speed, angle = self._sample_shamal_wind()
        self.wind_speed = speed
        self.wind_angle = angle
        self.fire_mask = torch.as_tensor(linear_wind_transform(speed, angle))

    def sample_fire_propogation(self):
        """
        Sample the next state of the wildfire model.
        """
        burning_before = int(self.state_space[FIRE_INDEX].sum())
        prev_fire = self.state_space[FIRE_INDEX].copy()
        if self.wind_model == "shamal":
            self._update_shamal_wind_mask()

        # Drops fuel level of enflamed cells
        self.state_space[FUEL_INDEX, self.state_space[FIRE_INDEX] == 1] -= (
            self.fuel_burn_rate
        )
        self.state_space[FUEL_INDEX, self.state_space[FUEL_INDEX] < 0] = 0

        if self.debug and self.calibration == "saudi":
            self._log_fuel_stats("step=" + str(self.time_step) + "_postburn")

        # Extinguishes cells that have run out of fuel
        self.state_space[FIRE_INDEX, self.state_space[FUEL_INDEX, :] <= 0] = 0

        # Runs kernel of neighborhing cells where each row
        # corresponds to the neighborhood of a cell
        torch_rep = torch.tensor(self.state_space[FIRE_INDEX]).unsqueeze(0)
        y = torch.nn.Unfold((5, 5), dilation=1, padding=2)
        z = y(torch_rep)

        # The relative importance of each neighboring cell is weighted
        z = z * self.fire_mask

        # Unenflamed cells are set to 1 to eliminate their role to the
        # fire spread equation
        z[z == 0] = 1
        z = z.prod(dim=0)
        z = 1 - z.reshape(self.state_space[FIRE_INDEX].shape)
        if self.terrain_spread_factor is not None:
            z = torch.clamp(z * self.terrain_spread_factor, min=0.0, max=1.0)

        if np.any(self.suppression_mask > 0):
            suppression_factor = 1.0 - self.suppression_strength * self.suppression_mask
            suppression_factor = np.clip(suppression_factor, 0.0, 1.0).astype(np.float32)
            z = torch.clamp(z * torch.from_numpy(suppression_factor), min=0.0, max=1.0)

        # From the probability of an ignition in z, new fire locations are
        # randomly generated
        prob_mask = torch.rand_like(z)
        new_fire = (z > prob_mask).float()
        new_ignitions = int(((np.array(new_fire) == 1) & (prev_fire == 0)).sum())

        # These new fire locations are added to the state
        self.state_space[FIRE_INDEX] = np.maximum(
            np.array(new_fire), self.state_space[FIRE_INDEX]
        )
        self.last_new_ignitions = new_ignitions

        if np.any(self.suppression_mask > 0):
            self.suppression_mask *= self.suppression_decay
            self.suppression_mask[self.suppression_mask < 1e-4] = 0.0

        if self.debug and self.calibration == "saudi":
            burning_after = int(self.state_space[FIRE_INDEX].sum())
            self._log_spread_stats(
                self.time_step, z, burning_before, burning_after
            )

    def update_paths_and_evactuations(self):
        """
        Performs three functions:
        1. Remove paths that been burned down by a fire
        2. Also stops evacuating any areas that were taking a burned down path
        3. Also decrements the evacuation timestamps
        """
        self.last_finished_evac_count = 0
        for i in range(len(self.paths)):
            # Decrement path counts and remove path if path is on fire
            if (
                self.paths[i][1]
                and np.sum(
                    np.logical_and(self.state_space[FIRE_INDEX], self.paths[i][0])
                )
                > 0
            ):
                self.state_space[PATHS_INDEX] -= self.paths[i][0]
                self.paths[i][1] = False

                # Stop evacuating an area if it was taking the removed path
                if i in self.evacuating_paths:
                    pop_centers = np.array(self.evacuating_paths[i])
                    pop_rows, pop_cols = pop_centers[:, 0], pop_centers[:, 1]

                    # Reset timestamp and evacuation index for populated areas
                    self.evacuating_timestamps[pop_rows, pop_cols] = np.inf
                    self.state_space[EVACUATING_INDEX, pop_rows, pop_cols] = 0
                    del self.evacuating_paths[i]

            # We need to decrement the evacuating paths timestamp
            elif i in self.evacuating_paths:

                # For the below, this code works for if multiple population centers
                # are taking the same path and finish at the same time, but if we have
                # it so that two population centers can't take the same
                # path it could probably be simplified
                pop_centers = np.array(self.evacuating_paths[i])
                pop_rows, pop_cols = pop_centers[:, 0], pop_centers[:, 1]
                self.evacuating_timestamps[pop_rows, pop_cols] -= 1
                done_evacuating = np.where(self.evacuating_timestamps == 0)

                self.state_space[
                    EVACUATING_INDEX, done_evacuating[0], done_evacuating[1]
                ] = 0
                self.state_space[
                    POPULATED_INDEX, done_evacuating[0], done_evacuating[1]
                ] = 0

                # Note that right now it is going to be vastly often the case that two
                # population cases don't finish evacuating along the same path at the
                # same time right now, so this is an extremely rare edge case, meaning
                # that most often this for loop will run for a single iteration
                done_evacuating = np.array([done_evacuating[0], done_evacuating[1]])
                done_evacuating = np.transpose(done_evacuating)
                for j in range(done_evacuating.shape[0]):
                    self.evacuating_paths[i].remove(list(done_evacuating[j]))

                    # This population center is done evacuating, so we can set its
                    # timestamp back to infinity (so we don't try to remove this
                    # from self.evacuating paths twice - was causing a bug)
                    update_row, update_col = (
                        done_evacuating[j, 0],
                        done_evacuating[j, 1],
                    )
                    self.evacuating_timestamps[update_row, update_col] = np.inf
                    self.finished_evacuating_cells.append([update_row, update_col])
                    self.last_finished_evac_count += 1

                # No more population centers are using this path, so we delete it
                if len(self.evacuating_paths[i]) == 0:
                    del self.evacuating_paths[i]

    def accumulate_reward(self):
        """
        Mark enflamed areas as no longer populated or evacuating and calculate reward.
        """
        # Get which populated_areas areas are on fire and evacuating
        populated_areas = np.where(self.state_space[POPULATED_INDEX] == 1)
        fire = self.state_space[FIRE_INDEX][populated_areas]
        evacuating = self.state_space[EVACUATING_INDEX][populated_areas]

        # Mark enflamed areas as no longer populated or evacuating
        enflamed_populated_areas = np.where(fire == 1)[0]
        enflamed_rows = populated_areas[0][enflamed_populated_areas]
        enflamed_cols = populated_areas[1][enflamed_populated_areas]

        # Depopulate enflamed areas and remove evacuations
        self.state_space[POPULATED_INDEX, enflamed_rows, enflamed_cols] = 0
        self.state_space[EVACUATING_INDEX, enflamed_rows, enflamed_cols] = 0

        current_fire_cells = int(self.state_space[FIRE_INDEX].sum())
        burning_population = int(
            np.logical_and(
                self.state_space[FIRE_INDEX] == 1,
                self.state_space[POPULATED_INDEX] == 1,
            ).sum()
        )
        fire_delta = self.prev_fire_cells - current_fire_cells
        w = self.reward_weights
        # Reward signal: dense, smooth, and action-correlated (via suppression affecting spread).
        # - Positive for reducing burning cells (fire_delta)
        # - Shaping term: negative proportional to current burning cells
        # - Penalty for newly ignited cells (spread pressure)
        # - Smaller population penalties (still important, but not allowed to swamp learning early)
        self.reward = (
            w["fire_delta"] * float(fire_delta)
            - w["burning_cells"] * float(current_fire_cells)
            - w["new_ignitions"] * float(self.last_new_ignitions)
            - w["newly_burned_population"] * float(len(enflamed_populated_areas))
            - w["burning_population"] * float(burning_population)
            + w["finished_evac"] * float(self.last_finished_evac_count)
        )
        self.prev_fire_cells = current_fire_cells
        self.last_reward_components = {
            "fire_delta": float(fire_delta),
            "fire_cells": current_fire_cells,
            "newly_burned": int(len(enflamed_populated_areas)),
            "burning_population": burning_population,
            "finished_evac": int(self.last_finished_evac_count),
            "new_ignitions": int(self.last_new_ignitions),
        }

    def advance_to_next_timestep(self):
        """
        Take three steps:
        1. Advance fire forward one timestep
        2. Update paths and evacuation
        3. Accumulate reward and document enflamed areas
        """
        self.sample_fire_propogation()
        self.update_paths_and_evactuations()
        self.accumulate_reward()
        self.time_step += 1

    def set_action(self, action: int):
        """
        Allow the agent to take an action within the action space.
        """
        # Check that there is an action to take
        if (
            action in self.action_to_pop_and_path
            and self.action_to_pop_and_path[action] is not None
        ):
            action_val = self.action_to_pop_and_path[action]
            if action_val is not None and len(action_val) > 0:
                pop_cell, path_index = action_val
                pop_cell_row, pop_cell_col = pop_cell[0], pop_cell[1]

                # Ensure that the path chosen and populated cell haven't burned down.
                # IMPORTANT: suppression should apply whenever action is taken so the
                # policy can continuously influence fire spread (not just once).
                if self.paths[path_index][1]:
                    # Start evacuation once (only if the population cell still exists).
                    if (
                        self.state_space[POPULATED_INDEX, pop_cell_row, pop_cell_col] == 1
                        and self.evacuating_timestamps[pop_cell_row, pop_cell_col] == np.inf
                    ):
                        if path_index in self.evacuating_paths:
                            self.evacuating_paths[path_index].append(pop_cell)
                        else:
                            self.evacuating_paths[path_index] = [pop_cell]
                        self.state_space[EVACUATING_INDEX, pop_cell_row, pop_cell_col] = 1
                        self.evacuating_timestamps[pop_cell_row, pop_cell_col] = 10

                    # Apply/refresh suppression every time the action is chosen,
                    # even if the population cell burned down (keeps action impactful).
                    self._apply_action_suppression(pop_cell, path_index)

    def get_state_utility(self) -> int:
        """
        Get the total amount of utility given a current state.
        """
        present_reward = self.reward
        self.reward = 0
        return present_reward

    def get_actions(self) -> list:
        """
        Get the set of actions available to the agent.
        """
        return self.actions

    def get_timestep(self) -> int:
        """
        Get current timestep of simulation
        """
        return self.time_step

    def get_state(self) -> np.ndarray:
        """
        Get the state space of the current configuration of the gridworld.
        """
        returned_state = np.copy(self.state_space)
        returned_state[PATHS_INDEX] = np.clip(returned_state[PATHS_INDEX], 0, 1)
        return returned_state

    def get_terminated(self) -> bool:
        """
        Get the status of the simulation.
        """
        if self.time_step >= 100:
            return True
        if self.terminate_on_population_loss:
            population_remaining = int(self.state_space[POPULATED_INDEX].sum())
            return population_remaining == 0
        return False

    def get_finished_evacuating(self) -> list:
        """
        Get the populated areas that are finished evacuating.
        """
        return self.finished_evacuating_cells
