# PyroRL

![example workflow](https://github.com/sisl/PyroRL/actions/workflows/testing.yml/badge.svg) [![codecov](https://codecov.io/github/sisl/PyroRL/graph/badge.svg?token=wBlFGsd5sS)](https://codecov.io/github/sisl/PyroRL) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://sisl.github.io/PyroRL/)

PyroRL is a new reinforcement learning environment built for the simulation of wildfire evacuation. Check out the [docs](https://sisl.github.io/PyroRL/) and the [demo](https://www.loom.com/share/39ddd19c790a49c0a1ea7e13cd4d1005?sid=679b631a-74b7-41e3-bd88-3e7d14c0adc2).

## How to Use

First, install our package:

```bash
pip install pyrorl
```

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

A compiled visualization of numerous iterations is seen below. For more examples, check out the `examples/` folder in the online repository.

![Example Visualization of PyroRL](https://github.com/sisl/PyroRL/raw/master/imgs/example_visualization.gif)

## Training (Extended Runs)

PyroRL supports stable, long-running PPO training suitable for research experiments.

To run a long training session (e.g., 300k timesteps) with the California calibration:

```bash
python scripts/train_ppo.py --calibration california --timesteps 300000
```

### Explanation of Outputs

- **Logs**: Training logs are automatically saved to `logs/ppo/`. This includes a `monitor.csv` with episode stats and a `<calibration>_training_curve.csv` containing step, reward, episode length, and a moving average of the reward. You can monitor the training progress via these files or TensorBoard logs if active.
- **Checkpoints**: Model checkpoints are saved in the `checkpoints/` directory.

### Best Model vs Final Model

During training, an evaluation callback runs periodically.
- **Best Model** (`checkpoints/<calibration>_best.zip`): Saved whenever the model achieves a new highest average reward during evaluation episodes. This is usually the model you want to use for downstream tasks.
- **Final Model** (e.g., `checkpoints/ppo_california_300k.zip`): The model checkpoint saved at the very end of the training process, regardless of whether it performed best.

### Reproducibility

To ensure reproducible training runs, the training script explicitly sets random seeds across `numpy`, `torch`, the `random` module, and the environment. You can control the seed via the `--seed` argument (defaults to 42):

```bash
python scripts/train_ppo.py --calibration california --timesteps 300000 --seed 42
```

### Suppression Penalty (Optional)

To prevent policies from learning "over-suppression" (spamming suppression actions across the board), you can optionally pass a penalty term that is subtracted from the reward for every suppression action taken:
```bash
python scripts/train_ppo.py --calibration california --timesteps 300000 --suppression_penalty 0.02
```

## Training Protocol

When running long PPO training sessions (e.g., 300k–500k timesteps), PyroRL follows a specific protocol to ensure the highest quality model is retained:

1. **Full-Length Training**: Training runs always execute to the full specified timesteps, without stopping early. This ensures the model explores the environment fully.
2. **Best Model Selection via Evaluation**: During training, an evaluation callback actively evaluates the model. The model achieving the highest average reward is saved as `checkpoints/ppo_<calibration>_best.zip`.
3. **Policy Degradation Awareness**: PPO models may experience policy degradation late in training, causing the performance to drop significantly. Instead of stopping the script early, the `DegradationWarningCallback` simply logs a warning to the console, allowing you to observe the drop without prematurely truncating the run.
4. **Final Evaluation Protocol**: For final testing, deployment, and benchmark comparison, **always** load and evaluate `best_model.zip`. The final model checkpoint saved at the end of training may represent a degraded policy and should not be used to conclude model performance.

## Scenario Definitions

These scenarios are specifically designed to simulate **distribution shift**, exposing the trained RL policy to conditions it did not experience during its standard training phase. They are strictly used for verification experiments to test the robustness and generalizability of the trained models.


```bash
python scripts/run_scenario_matrix.py --ca_model checkpoints/ppo_california_best.zip --sa_model checkpoints/ppo_saudi_best.zip --episodes 50
```

### Suppression Efficiency Metric

The evaluation matrix now tracks **Suppression Efficiency**, defined as the number of burned cells saved per suppression action taken. The script automatically runs a "no suppression" baseline for each scenario, then calculates the efficiency of the CA and SA models against that baseline. 

The matrix automatically generates two outputs:
- `logs/scenario_matrix.csv`: Full evaluation results, including `suppression_actions`, `burned_cells`, and `efficiency`.
- `logs/failure_modes.csv`: An auto-generated heuristic report that classifies if a policy suffered from "over-spread", "over-suppression", or "delayed response" for each scenario.

* **`high_wind`**: Increases wind magnitude and gust probability. Tests if the policy can prioritize containment in the face of rapid, directional spread.
* **`low_fuel`**: Reduces global fuel density. Tests if the policy maintains performance when natural fire decay is higher than expected.
* **`oasis_cluster`**: Creates concentrated clusters of dense fuel in sparse areas. Tests the policy's ability to respond to intense localized flare-ups.
* **`multi_ignition`**: Sets multiple initial fire ignition points across the grid. Tests the policy's capacity to manage multi-front containment strategies.
* **`narrow_corridor`**: Creates a strip of high fuel surrounded by low fuel. Tests if the policy can adapt to highly constrained, rapid linear fire propagation.
* **`extreme_wind`**: Injects very high wind speeds (40-50 range) with random severe gust spikes. Tests if the policy can manage extremely rapid, unpredictable directional spread.
* **`fuel_depletion`**: Causes fuel to decay much faster over time with moderate initial fuel. Tests the policy's reaction to fires that start strong but die off unpredictably.
* **`random_terrain`**: Adds irregular elevation noise, breaking smooth terrain assumptions. Tests if the policy is robust against unpredictable fire channels and non-uniform spread.
* **`delayed_ignition`**: Prevents the fire from starting for the first 15 steps, before sudden ignition occurs. Tests the policy's readiness and if it wastes early actions.
* **`dense_population`**: Increases the number of populated cells significantly across the grid. These extra populations cannot be evacuated, testing the policy's ability to make rapid trade-offs and rely strictly on fire suppression for defense.

## Visualization

PyroRL provides tools to visually interpret experimental results and policy behavior, making them publication-ready.

### 1. Scenario Comparison Plot
To visualize the performance difference between models across various distribution shift scenarios, use the plotting script:
```bash
python scripts/plot_scenario_results.py
```
**What it shows**: This script reads `logs/scenario_matrix.csv` and outputs a grouped bar chart (`logs/fig_scenario_comparison.png`) showing the mean reward of both CA and SA models for each scenario, including 95% Confidence Interval error bars.

### 2. Fire Spread Visualization (GIFs)
To observe exactly how the fire spreads and how the RL policy responds with suppression actions, you can generate an animated GIF of a single episode:
```bash
python scripts/visualize_episode.py --model checkpoints/ppo_california_best.zip --calibration california --scenario oasis_cluster --out logs/fire_oasis_cluster_ca.gif
```
**What it shows**: A step-by-step grid rendering of the environment where you can visually track fire cells (red), unburned grass/fuel (green), population centers (dark blue), evacuation paths (yellow), and finished evacuations (purple). This makes it easy to compare the behavior of different models (e.g., CA vs. SA) under key scenarios like `oasis_cluster` or `random_terrain`.

### 3. Visualization Examples and Output Types
The visualization script now automatically produces research-quality outputs that explicitly show **temporal evolution** of the policy behavior:

1. **Key Frame Snapshots**: Instead of only generating a GIF, the script automatically dumps high-resolution (200 DPI) static frames at critical moments (`t=1`, `t=20`, `t=50`, and `t=100`). These are saved to the `logs/` directory.
2. **Suppression Overlay**: Active suppression zones are visually highlighted on the grid using a black diagonal hatch pattern (`////`). This enables you to exactly pinpoint the model's defensive actions step-by-step.
3. **Fire Direction Annotations**: A wind vector arrow is dynamically rendered beneath the grid to visualize the primary driving direction of the fire spread.

**Why these matter for analysis:**
By extracting high-resolution snapshots at standard temporal intervals, it becomes much easier to embed the exact progression of a policy failure (or success) directly into a paper or report. You can directly correlate the suppression hatch overlays with the fire line to see if the policy successfully anticipated the fire growth or reacted too late.

### 4. Comparative Visualization
For direct side-by-side behavioral comparison between two policies (e.g. CA vs SA), use the `--compare` mode:
```bash
python scripts/visualize_episode.py \
  --compare \
  --ca_model checkpoints/ppo_california_best.zip \
  --sa_model checkpoints/ppo_saudi_best.zip \
  --scenario extreme_wind
```
**What it shows**:
This explicitly runs two parallel simulation environments identically seeded to guarantee the same initial fire spawn and terrain layout. It outputs:
- **`logs/compare_<scenario>.gif`**: A synchronized, side-by-side animated representation of how both policies deal with the identical crisis over time.
- **`logs/compare_<scenario>_t<step>.png`**: A high-resolution, side-by-side plot rendered at key timesteps (`t=1`, `20`, `50`, `100`). This is explicitly structured as publication-ready visual evidence to show divergence in policy behavior under severe distribution shift.

### 5. Annotation Overlays
You can add the `--annotate` flag to any visualizer command to produce annotated overlays designed for slide decks and papers. This feature draws:
- A dashed red bounding box around the largest continuous fire cluster.
- Distinct star markers highlighting population cells.
- A descriptive text box indicating the policy name, scenario, and current timestep.

### 6. Automated Generation
To fully automate the generation of comparative figures, captions, and snapshots across **every scenario**, use the dedicated generator script:
```bash
python scripts/generate_all_visualizations.py --ca_model checkpoints/ppo_california_best.zip --sa_model checkpoints/ppo_saudi_best.zip
```
**What it does**:
This iterates over all 10 registered scenarios, creating a dedicated subdirectory for each inside `logs/` (e.g., `logs/extreme_wind/`). Inside each folder, it will output the full suite of comparison assets and a `caption.txt` with an auto-generated, academic interpretation of the policy behaviors.

## How to Contribute

For information on how to contribute, check out our [contribution guide](https://sisl.github.io/PyroRL/contribution-guide/).
