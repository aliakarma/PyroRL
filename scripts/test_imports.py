import sys
import os
from pathlib import Path

# Add the repository root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

print(f"DEBUG: sys.path includes {repo_root}")

try:
    # Attempting to import using the nested structure as requested
    from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv
    print("SUCCESS: Imported PyroRLEnv from pyrorl.pyrorl.envs.pyrorl")
    
    # Try importing from the package root to verify __init__.py exports
    from pyrorl.pyrorl.envs import PyroRLEnv as PyroRLEnvInit
    print("SUCCESS: Imported PyroRLEnv from pyrorl.pyrorl.envs")
    
    # Initialize environment
    from pyrorl.pyrorl.map_helpers.create_map_info import generate_map_info
    
    num_rows, num_cols = 10, 10
    populated_areas, paths, paths_to_pops = generate_map_info(
        num_rows, num_cols, num_populated_areas=1, save_map=False
    )
    
    env = PyroRLEnv(
        num_rows=num_rows,
        num_cols=num_cols,
        populated_areas=populated_areas,
        paths=paths,
        paths_to_pops=paths_to_pops,
        calibration="saudi",
        scenario="high_wind"
    )
    env.reset()
    
    print("IMPORT + ENV OK")
    
except Exception as e:
    print(f"FAILURE: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
