import sys
import numpy as np
from pathlib import Path
repo_root = Path('.').resolve()
sys.path.insert(0, str(repo_root))

from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv
scenarios = [
    'high_wind', 'low_fuel', 'oasis_cluster', 'multi_ignition', 'narrow_corridor',
    'extreme_wind', 'fuel_depletion', 'random_terrain', 'delayed_ignition', 'dense_population'
]

kwargs = {
    'num_rows': 10, 'num_cols': 10,
    'populated_areas': np.array([[5, 5]]),
    'paths': np.array([[[4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]], dtype=object),
    'paths_to_pops': {0: [[5, 5]]},
    'calibration': 'saudi'
}

for scenario in scenarios:
    print(f'Testing {scenario}...')
    env = PyroRLEnv(scenario=scenario, **kwargs)
    env.reset()
    env.step(0)
    env.close()
    print(f'{scenario} OK')
