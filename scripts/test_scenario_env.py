import sys
import os
from pathlib import Path
import numpy as np

# Add the repository root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

try:
    from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv
    from pyrorl.pyrorl.map_helpers.create_map_info import generate_map_info
except ImportError:
    # Fallback for different environments if necessary
    from pyrorl.envs.pyrorl import PyroRLEnv
    from pyrorl.map_helpers.create_map_info import generate_map_info

def main():
    # Define environment dimensions
    num_rows, num_cols = 20, 20
    num_populated_areas = 5

    print(f"Initializing map with {num_rows}x{num_cols} grid and {num_populated_areas} populated areas...")

    # Generate valid map inputs using existing utilities
    populated_areas, paths, paths_to_pops = generate_map_info(
        num_rows, 
        num_cols, 
        num_populated_areas=num_populated_areas, 
        save_map=False
    )

    print("Initializing PyroRLEnv with 'saudi' calibration and 'high_wind' scenario...")

    # Correct initialization with all required positional arguments
    env = PyroRLEnv(
        num_rows=num_rows,
        num_cols=num_cols,
        populated_areas=populated_areas,
        paths=paths,
        paths_to_pops=paths_to_pops,
        calibration="saudi",
        scenario="high_wind"
    )

    # Verify that reset works
    obs, info = env.reset()

    # Basic sanity checks on the state
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    
    print("SCENARIO ENV OK")

if __name__ == "__main__":
    main()
