<pre>
██████╗ ██╗   ██╗██████╗  ██████╗ ██████╗ ██╗     
██╔══██╗╚██╗ ██╔╝██╔══██╗██╔═══██╗██╔══██╗██║     
██████╔╝ ╚████╔╝ ██████╔╝██║   ██║██████╔╝██║     
██╔═══╝   ╚██╔╝  ██╔══██╗██║   ██║██╔══██╗██║     
██║        ██║   ██║  ██║╚██████╔╝██║  ██║███████╗
╚═╝        ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
</pre>

![example workflow](https://github.com/sisl/PyroRL/actions/workflows/testing.yml/badge.svg) [![codecov](https://codecov.io/github/sisl/PyroRL/graph/badge.svg?token=wBlFGsd5sS)](https://codecov.io/github/sisl/PyroRL) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://sisl.github.io/PyroRL/)  [![DOI](https://joss.theoj.org/papers/10.21105/joss.06739/status.svg)](https://joss.theoj.org/papers/10.21105/joss.06739)

PyroRL is a new reinforcement learning environment built for the simulation of wildfire evacuation. Check out the [docs](https://sisl.github.io/PyroRL/) and the [demo](https://www.youtube.com/embed/Pt4cI5jBbKo).

## Project Overview

This project extends PyroRL with:
- **Saudi/desert calibration**: Arid fuel models, dune-aware terrain, and Shamal wind regimes.
- **Scenario-based evaluation**: Structured perturbation scenarios (e.g., high wind, oasis cluster).
- **Statistical validation**: Rigorous multi-episode evaluation reporting mean, standard deviation, and 95% confidence intervals.
- **Failure mode analysis**: Automated heuristics to classify why policies fail out-of-distribution.

**Core contribution:** Evaluating RL policy robustness under environment distribution shift.

## Installation

```bash
git clone https://github.com/aliakarma/PyroRL
cd PyroRL
pip install stable-baselines3 gymnasium numpy matplotlib torch scipy
```

## Project Structure

```text
PyroRL/
├── pyrorl/          # Core environment, calibration, and scenario logic
├── scripts/         # Training and evaluation pipeline scripts
├── tests/           # Regression and unit tests
├── checkpoints/     # Trained PPO models (.zip)
├── logs/            # Statistical evaluation CSV outputs
├── docs/            # Documentation website source
├── examples/        # Introductory usage examples
├── paper/           # Academic paper materials
├── README.md        # This file
├── LICENSE          # MIT License
└── .gitignore       # Version control rules
```

## How to Use

Note that PyroRL requires Python version 3.8+:

To use our wildfire evacuation environment, define the dimensions of your grid, where the populated areas are, the paths, and which populated areas can use which path. See an example below.

```python
# Create environment
kwargs = {
    'num_rows': num_rows,
    'num_cols': num_cols,
    'populated_areas': populated_areas,
    'paths': paths,
    'paths_to_pops': paths_to_pops
}
env = gymnasium.make('pyrorl/PyroRL-v0', **kwargs)

# Run a simple loop of the environment
env.reset()
for _ in range(10):

    # Take action and observation
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Render environment and print reward
    env.render()
    print("Reward: " + str(reward))
```

    You can also toggle calibration modes to switch between California-style forests and Saudi desert dynamics:


A compiled visualization of numerous iterations is seen below. For more examples, check out the `examples/` folder.

![Example Visualization of PyroRL](imgs/example_visualization.gif)

For a more comprehensive tutorial, check out the [quickstart](https://sisl.github.io/PyroRL/quickstart/) page on our docs website.

## Testing California vs Saudi Calibration

### Install

```bash
pip install -e .
```

### Run basic test

```bash
python scripts/test_calibration.py --steps 50
```

### Run extended test

```bash
python scripts/test_calibration.py --steps 100 --seeds 5
```

### Run with trajectory export

```bash
python scripts/test_calibration.py --steps 100 --seeds 5 --csv results.csv
```

### Plot results

```bash
python scripts/plot_calibration.py results.csv
python scripts/plot_calibration.py results.csv --output calibration_plot.png
```

## Calibration Validation

### Expected Behavior

**California mode** (`calibration="california"`):
- Dense forest fuel (higher mean, higher variance)
- Rapid fire growth initially
- Peak fire extent in mid-simulation
- Gradual decay as fuel depletes
- Trajectory: growth → plateau → gradual decay

**Saudi mode** (`calibration="saudi"`):
- Sparse desert scrub with oasis-like fuel clusters
- Slower fire growth (lower fuel density)
- Lower peak fire extent compared to California
- Smoother decay without early collapse
- Trajectory: slower growth → lower plateau → decay

### Debug Output

Enable debug diagnostics with `--debug` flag:

```bash
python scripts/test_calibration.py --steps 50 --debug
```

This prints fuel statistics (min, mean, max) and fire spread probabilities every 5 steps
for Saudi calibration. Useful for:
- Validating fuel distribution
- Monitoring spread probability over time
- Diagnosing instability or unrealistic dynamics

### Interpretation

After running comparison:

- **Mean fire cells**: Average fire extent per step (higher = stronger spread)
- **Peak fire cells**: Maximum extent reached (shows how far fire spreads)
- **Time to peak**: When fire reaches maximum extent (timing of spread)
- **Final fire cells**: Fire extent at end of simulation (residual burn)

Curves should show:
- **Smooth progression** (no sudden jumps or collapses)
- **Consistent pattern across seeds** (low standard deviation)
- **Physical plausibility** (Saudi slower and patchier than California)

If instability occurs:
1. Check `--debug` output for suspicious fuel or spread values
2. Verify fuel distribution with debug stats
3. Adjust calibration parameters (fuel_mean, fuel_stdev, fire_propagation_rate)

## Training Pipeline

The repository includes scripts to train robust models on different environment calibrations.

### Train California model

```bash
python scripts/train_ppo.py --calibration california --timesteps 100000
```
Then, save the best model:

```bash
rename checkpoints\best_model.zip ppo_california.zip
```

### Train Saudi model

```bash
python scripts/train_ppo.py --calibration saudi --timesteps 100000
```
Then, save the best model:
```bash
mv checkpoints/best_model.zip checkpoints/ppo_saudi.zip
```

## Scenario System

The scenario system allows for evaluating policies under specific environmental perturbations. 
Available scenarios:
* `high_wind`: Increases wind magnitude and gust probability.
* `low_fuel`: Reduces global fuel density.
* `oasis_cluster`: Creates concentrated clusters of dense fuel in sparse areas.
* `multi_ignition`: Spawns secondary fires simultaneously.
* `narrow_corridor`: Restricts paths and populates narrow traversal areas.

Example usage in code:
```python
from pyrorl.pyrorl.envs.pyrorl import PyroRLEnv
env = PyroRLEnv(calibration="saudi", scenario="high_wind")
```

## Statistical Evaluation

To evaluate a specific policy against a scenario and collect statistical significance bounds:

```bash
python scripts/evaluate_scenarios.py \
  --model checkpoints/ppo_california.zip \
  --calibration saudi \
  --scenario high_wind \
  --episodes 30
```
This computes and reports the **mean reward**, **standard deviation**, and **95% confidence interval**, enabling rigorous research documentation.

## Scenario Matrix Experiment

Run a comprehensive evaluation comparing the California-trained policy vs. the Saudi-trained policy across all defined scenarios:

```bash
python scripts/run_scenario_matrix.py \
  --ca_model checkpoints/ppo_california.zip \
  --sa_model checkpoints/ppo_saudi.zip \
  --episodes 30
```

This experiment calculates the performance drop and outputs the results to:
* `logs/scenario_matrix.csv` (Summary statistics)
* `logs/scenario_matrix_detail.csv` (Per-episode raw metrics)

## Failure Mode Analysis

To analyze *why* a specific model is failing under different scenarios, run the automated heuristic classifier:

```bash
python scripts/analyze_failure_modes.py \
  --model checkpoints/ppo_california.zip \
  --episodes 30
```

This tracks spread velocity, action entropy, and suppression effectiveness to classify failures as:
* **underestimates spread**: The fire grows rapidly out of control.
* **environment sensitivity**: High variance in episode outcomes.
* **policy collapse**: Agent actions become random or locked to a single useless action.
* **action mismatch**: Suppression actions are attempted but are highly ineffective.
* **none**: The policy succeeds.

## Results Summary

Key findings from our cross-calibration evaluation:
* The **SA-trained policy** systematically outperforms the CA-trained policy across all Saudi desert scenarios.
* The largest performance gap is observed in the **`oasis_cluster`** scenario.
* The CA-trained model predominantly fails due to **underestimating spread**; its assumptions about continuous fuel distribution cause catastrophic failures in sparse, clustered terrain.
* All findings are statistically significant, validated via 95% Confidence Intervals across 30+ episodes.

## Reproducibility

This repository guarantees reproducibility through:
* **Fixed Seeds**: Scripts enforce `seed` propagation to numpy, torch, and Python standard libraries.
* **Deterministic Evaluation**: Evaluation steps use `deterministic=True` for action selection.
* **Comprehensive Logs**: CSV outputs and TensorBoard events are tracked.
* **Included Models**: Pre-trained model checkpoints are tracked in the repository for immediate evaluation.

## Notes / Troubleshooting

### Import issues
If you encounter `ModuleNotFoundError` or relative import errors when running scripts from different directories, insert the project root into your path:
```python
import sys
sys.path.append(".")
```

### NumPy loading issue
If you encounter a `ValueError` related to loading numpy arrays in PyTorch when loading checkpoints, retrain the models locally in the same environment:
```bash
python scripts/train_ppo.py --calibration saudi --timesteps 100000
```

## How to Contribute

For information on how to contribute, check out our [contribution guide](https://sisl.github.io/PyroRL/contribution-guide/).

## Repository Status

This repository is structured for:
* reproducible experiments
* research evaluation
* academic submission
