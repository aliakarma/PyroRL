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

## Training Protocol

When running long PPO training sessions (e.g., 300k–500k timesteps), PyroRL follows a specific protocol to ensure the highest quality model is retained:

1. **Full-Length Training**: Training runs always execute to the full specified timesteps, without stopping early. This ensures the model explores the environment fully.
2. **Best Model Selection via Evaluation**: During training, an evaluation callback actively evaluates the model. The model achieving the highest average reward is saved as `checkpoints/ppo_<calibration>_best.zip`.
3. **Policy Degradation Awareness**: PPO models may experience policy degradation late in training, causing the performance to drop significantly. Instead of stopping the script early, the `DegradationWarningCallback` simply logs a warning to the console, allowing you to observe the drop without prematurely truncating the run.
4. **Final Evaluation Protocol**: For final testing, deployment, and benchmark comparison, **always** load and evaluate `best_model.zip`. The final model checkpoint saved at the end of training may represent a degraded policy and should not be used to conclude model performance.

## How to Contribute

For information on how to contribute, check out our [contribution guide](https://sisl.github.io/PyroRL/contribution-guide/).
