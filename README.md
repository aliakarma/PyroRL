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

## How to Use

First, install our package. Note that PyroRL requires Python version 3.8:

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

    You can also toggle calibration modes to switch between California-style forests and Saudi desert dynamics:

    ```python
    from pyrorl.envs import PyroRLEnv

    env = PyroRLEnv(
        num_rows=num_rows,
        num_cols=num_cols,
        populated_areas=populated_areas,
        paths=paths,
        paths_to_pops=paths_to_pops,
        calibration="saudi",
    )
    ```

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

## PPO Sanity Check

Before scaling training to cloud infrastructure, verify that reinforcement learning can learn meaningful behavior on PyroRL with PPO.

### Install Dependencies

Ensure you have Stable-Baselines3 and plotting tools:

```bash
pip install stable-baselines3 matplotlib pandas tensorboard
```

### Run Training (California)

Train a PPO agent on California calibration for 50,000 timesteps:

```bash
python scripts/train_ppo.py --calibration california --timesteps 50000
```

### Run Training (Saudi)

Train a PPO agent on Saudi calibration for 50,000 timesteps:

```bash
python scripts/train_ppo.py --calibration saudi --timesteps 50000
```

### Training Options

```bash
# Custom timesteps
python scripts/train_ppo.py --calibration california --timesteps 100000

# Custom seed for reproducibility
python scripts/train_ppo.py --calibration saudi --seed 123

# Custom checkpoint directory
python scripts/train_ppo.py --save_path my_models/
```

### What to Expect

This sanity check verifies that:

1. **Training doesn't crash**: Environment, model, and monitoring work together
2. **Rewards trend upward**: Average reward over episodes should increase (even slightly)
3. **Model learns calibration differences**: California and Saudi should show different learning curves
4. **Convergence is reasonable**: Learning should stabilize by ~50k timesteps

### Output

Training produces:

- **Model checkpoint**: `checkpoints/ppo_{calibration}.zip`
- **Training logs**: `logs/{calibration}_ppo/` (TensorBoard-compatible)
- **Monitor data**: Episode rewards and lengths in `monitor.csv`
- **Reward curve plot**: `logs/{calibration}_ppo/{calibration}_reward_curve.png`

### Interpretation

- **Reward trend**: Look for upward trajectory in reward curve (noisy is okay)
- **Episode length**: May increase or decrease depending on learned policy
- **Evaluation**: 5 deterministic rollouts shown at end; should be consistent with training trend
- **Difference between calibrations**: California typically shows faster learning (denser rewards)

### Hyperparameters

Default hyperparameters are conservative and chosen for stability:

- Learning rate: 3e-4
- n_steps: 2048 (episode length before update)
- batch_size: 64
- gamma: 0.99 (discount factor)

For more aggressive learning, increase learning_rate or reduce batch_size (see Stable-Baselines3 docs).

### Next Steps

After confirming sanity check passes:

1. Scale timesteps for better convergence (500k - 1M)
2. Tune hyperparameters for your specific calibration
3. Deploy to cloud for distributed training

## How to Contribute

For information on how to contribute, check out our [contribution guide](https://sisl.github.io/PyroRL/contribution-guide/).
