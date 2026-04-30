"""
OpenAI Gym Environment Wrapper Class
"""

from pyrorl.envs.environment.environment import (
    FireWorld,
    FIRE_INDEX,
    POPULATED_INDEX,
    EVACUATING_INDEX,
)
from pyrorl.envs.environment.calibration_config import get_config
from pyrorl.envs.environment.scenarios import apply_scenario, AVAILABLE_SCENARIOS
import gymnasium as gym
from gymnasium import spaces
import imageio.v2 as imageio
import numpy as np
import os
import pygame
import shutil
from typing import Optional, Any, Tuple, List

# Constants for visualization
IMG_DIRECTORY = "grid_screenshots/"
FIRE_COLOR = pygame.Color("#ef476f")
POPULATED_COLOR = pygame.Color("#073b4c")
EVACUATING_COLOR = pygame.Color("#118ab2")
PATH_COLOR = pygame.Color("#ffd166")
GRASS_COLOR = pygame.Color("#06d6a0")
FINISHED_COLOR = pygame.Color("#BF9ACA")


class WildfireEvacuationEnv(gym.Env):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        populated_areas: np.ndarray,
        paths: np.ndarray,
        paths_to_pops: dict,
        custom_fire_locations: Optional[np.ndarray] = None,
        wind_speed: Optional[float] = None,
        wind_angle: Optional[float] = None,
        fuel_mean: Optional[float] = None,
        fuel_stdev: Optional[float] = None,
        fire_propagation_rate: Optional[float] = None,
        calibration: str = "california",
        scenario: Optional[str] = None,
        fuel_burn_rate: Optional[float] = None,
        visibility_radius: Optional[int] = None,
        dust_intensity: Optional[float] = None,
        visibility_center: Optional[Tuple[int, int]] = None,
        terminate_on_population_loss: bool = True,
        reward_weights: Optional[dict[str, float]] = None,
        debug: bool = False,
        skip: bool = False,
    ):
        """
        Set up the basic environment and its parameters.
        """
        # Save parameters and set up environment
        if calibration not in {"california", "saudi"}:
            raise ValueError("Calibration must be either 'california' or 'saudi'")
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.populated_areas = populated_areas
        self.paths = paths
        self.paths_to_pops = paths_to_pops
        self.custom_fire_locations = custom_fire_locations
        self.wind_speed = wind_speed
        self.wind_angle = wind_angle
        self.calibration = calibration
        config = get_config(calibration)
        if fuel_mean is None:
            fuel_mean = config.fuel_mean
        if fuel_stdev is None:
            fuel_stdev = config.fuel_stdev
        if fire_propagation_rate is None:
            fire_propagation_rate = config.fire_propagation_rate
        if fuel_burn_rate is None:
            fuel_burn_rate = config.fuel_burn_rate
        if dust_intensity is None:
            dust_intensity = config.visibility_params["dust_intensity"]
        if visibility_radius is None:
            visibility_radius = config.visibility_radius
            if visibility_radius is None and calibration == "saudi":
                visibility_radius = max(2, int(min(num_rows, num_cols) * 0.25))

        self.fuel_mean = fuel_mean
        self.fuel_stdev = fuel_stdev
        self.fire_propagation_rate = fire_propagation_rate
        self.fuel_burn_rate = fuel_burn_rate
        self.visibility_radius = visibility_radius
        self.dust_intensity = dust_intensity
        self.visibility_center = visibility_center
        self.terminate_on_population_loss = terminate_on_population_loss
        self.reward_weights = reward_weights
        self.debug = debug
        self.skip = skip
        self.scenario = scenario
        self.fire_env = FireWorld(
            num_rows,
            num_cols,
            populated_areas,
            paths,
            paths_to_pops,
            custom_fire_locations=custom_fire_locations,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            fuel_mean=fuel_mean,
            fuel_stdev=fuel_stdev,
            fire_propagation_rate=fire_propagation_rate,
            calibration=calibration,
            fuel_burn_rate=fuel_burn_rate,
            reward_weights=reward_weights,
            terminate_on_population_loss=terminate_on_population_loss,
            debug=debug,
        )

        # Apply scenario modifications (after FireWorld is constructed)
        if self.scenario is not None:
            apply_scenario(self, self.scenario)

        # Set up action space
        actions = self.fire_env.get_actions()
        self.action_space = spaces.Discrete(len(actions))

        # Set up observation space
        observations = self._apply_visibility(self.fire_env.get_state())
        self.observation_space = spaces.Box(
            low=0, high=200, shape=observations.shape, dtype=np.float64
        )

        # Create directory to store screenshots
        if os.path.exists(IMG_DIRECTORY) is False:
            os.mkdir(IMG_DIRECTORY)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.fire_env = FireWorld(
            self.num_rows,
            self.num_cols,
            self.populated_areas,
            self.paths,
            self.paths_to_pops,
            wind_speed=self.wind_speed,
            wind_angle=self.wind_angle,
            fuel_mean=self.fuel_mean,
            fuel_stdev=self.fuel_stdev,
            fire_propagation_rate=self.fire_propagation_rate,
            calibration=self.calibration,
            fuel_burn_rate=self.fuel_burn_rate,
            reward_weights=self.reward_weights,
            terminate_on_population_loss=self.terminate_on_population_loss,
            debug=self.debug,
        )

        # Apply scenario modifications on every reset
        if self.scenario is not None:
            apply_scenario(self, self.scenario)

        state_space = self._apply_visibility(self.fire_env.get_state())
        return state_space, {"": ""}

    def step(self, action: int) -> tuple:
        """
        Take a step and advance the environment after taking an action.
        """
        # Take the action and advance to the next timestep
        self.fire_env.set_action(action)
        self.fire_env.advance_to_next_timestep()

        # Gather observations and rewards
        observations = self._apply_visibility(self.fire_env.get_state())
        rewards = self.fire_env.get_state_utility()
        terminated = self.fire_env.get_terminated()
        state = self.fire_env.get_state()
        info = {
            "fire_cells": int(state[FIRE_INDEX].sum()),
            "burning_population": int(
                np.logical_and(
                    state[FIRE_INDEX] == 1,
                    state[POPULATED_INDEX] == 1,
                ).sum()
            ),
            "finished_evac": int(self.fire_env.last_reward_components.get("finished_evac", 0)),
            "newly_burned": int(self.fire_env.last_reward_components.get("newly_burned", 0)),
            "fire_delta": float(self.fire_env.last_reward_components.get("fire_delta", 0.0)),
            "new_ignitions": int(self.fire_env.last_reward_components.get("new_ignitions", 0)),
            "action": int(action),
        }
        return observations, rewards, terminated, False, info

    def _apply_visibility(self, state_space: np.ndarray) -> np.ndarray:
        if self.visibility_radius is None:
            return state_space

        rows, cols = state_space.shape[1], state_space.shape[2]
        effective_radius = self.visibility_radius
        if self.dust_intensity > 0:
            effective_radius = max(
                1, int(round(self.visibility_radius * (1.0 - self.dust_intensity)))
            )
        if effective_radius >= max(rows, cols):
            return state_space

        mask = np.zeros((rows, cols), dtype=bool)
        centers = self._visibility_centers(state_space)
        row_idx = np.arange(rows)[:, None]
        col_idx = np.arange(cols)[None, :]
        for center in centers:
            center_row, center_col = center
            dist2 = (row_idx - center_row) ** 2 + (col_idx - center_col) ** 2
            mask |= dist2 <= effective_radius**2

        masked_state = np.copy(state_space)
        masked_state[:, ~mask] = 0
        return masked_state

    def _visibility_centers(self, state_space: np.ndarray) -> List[Tuple[int, int]]:
        if self.visibility_center is not None:
            return [self.visibility_center]

        population_mask = np.logical_or(
            state_space[POPULATED_INDEX] == 1,
            state_space[EVACUATING_INDEX] == 1,
        )
        rows, cols = np.where(population_mask)
        if rows.size > 0:
            return list(zip(rows.tolist(), cols.tolist()))

        return [(state_space.shape[1] // 2, state_space.shape[2] // 2)]

    def render_hf(
        self, screen: pygame.Surface, font: pygame.font.Font
    ) -> pygame.Surface:
        """
        Set up header and footer
        """
        # Get width and height of the screen
        surface_width = screen.get_width()
        surface_height = screen.get_height()

        # Starting locations and timestep
        x_offset, y_offset = 0.05, 0.05
        timestep = self.fire_env.get_timestep()

        # Set title of the screen
        text = font.render("Timestep #: " + str(timestep), True, (0, 0, 0))
        screen.blit(text, (surface_width * x_offset, surface_height * y_offset))

        # Set initial grid squares and offsets
        grid_squares = [
            (GRASS_COLOR, "Grass"),
            (FIRE_COLOR, "Fire"),
            (POPULATED_COLOR, "Populated"),
            (EVACUATING_COLOR, "Evacuating"),
            (PATH_COLOR, "Path"),
            (FINISHED_COLOR, "Finished"),
        ]
        x_offset, y_offset = 0.2, 0.045

        # Iterate through, create the grid squares
        for i in range(len(grid_squares)):

            # Get the color and name, set in the screen
            (color, name) = grid_squares[i]
            pygame.draw.rect(
                screen,
                color,
                (surface_width * x_offset, surface_height * y_offset, 25, 25),
            )
            text = font.render(name, True, (0, 0, 0))
            screen.blit(
                text, (surface_width * x_offset + 35, surface_height * y_offset + 5)
            )

            # Adjust appropriate offset
            x_offset += 0.125

        return screen

    def render(self):
        """
        Render the environment
        """
        # Set up the state space
        state_space = self.fire_env.get_state()
        finished_evacuating = self.fire_env.get_finished_evacuating()
        (_, rows, cols) = state_space.shape

        # Get dimensions of the screen
        pygame.init()
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h

        # Set up screen and font
        surface_width = screen_width * 0.8
        surface_height = screen_height * 0.8
        screen = pygame.display.set_mode([surface_width, surface_height])
        font = pygame.font.Font(None, 25)

        # Set screen details
        screen.fill((255, 255, 255))
        pygame.display.set_caption("PyroRL")
        screen = self.render_hf(screen, font)

        # Calculation for square
        total_width = 0.85 * surface_width - 2 * (cols - 1)
        total_height = 0.85 * surface_height - 2 * (rows - 1)
        square_dim = min(int(total_width / cols), int(total_height / rows))

        # Calculate start x, start y
        start_x = surface_width - 2 * (cols - 1) - square_dim * cols
        start_y = (
            surface_height - 2 * (rows - 1) - square_dim * rows + 0.05 * surface_height
        )
        start_x /= 2
        start_y /= 2

        # Running the loop!
        running = True
        while running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    timestep = self.fire_env.get_timestep()
                    pygame.image.save(screen, IMG_DIRECTORY + str(timestep) + ".png")
                    running = False

            # Iterate through all of the squares
            # Note: try to vectorize?
            for x in range(cols):
                for y in range(rows):

                    # Set color of the square
                    color = GRASS_COLOR
                    if state_space[4][y][x] > 0:
                        color = PATH_COLOR
                    if state_space[0][y][x] == 1:
                        color = FIRE_COLOR
                    if state_space[2][y][x] == 1:
                        color = POPULATED_COLOR
                    if state_space[3][y][x] > 0:
                        color = EVACUATING_COLOR
                    if [y, x] in finished_evacuating:
                        color = FINISHED_COLOR

                    # Draw the square
                    # self.grid_dim = min(self.grid_width, self.grid_height)
                    square_rect = pygame.Rect(
                        start_x + x * (square_dim + 2),
                        start_y + y * (square_dim + 2),
                        square_dim,
                        square_dim,
                    )
                    pygame.draw.rect(screen, color, square_rect)

            # Render and then quit outside
            pygame.display.flip()

            # If we skip, then we basically just render the canvas and then quit outside
            if self.skip:
                timestep = self.fire_env.get_timestep()
                pygame.image.save(screen, IMG_DIRECTORY + str(timestep) + ".png")
                running = False
        pygame.quit()

    def generate_gif(self):
        """
        Save run as a GIF.
        """
        files = [str(i) for i in range(1, self.fire_env.get_timestep() + 1)]
        images = [imageio.imread(IMG_DIRECTORY + f + ".png") for f in files]
        imageio.mimsave("training.gif", images, loop=0)
        shutil.rmtree(IMG_DIRECTORY)


PyroRLEnv = WildfireEvacuationEnv
