A2C (TD / GAE / Monte Carlo)

This project extends A2C to MuJoCo continuous-control tasks. It compares 1-step TD, GAE, and Monte Carlo advantage estimation, and also includes a small grid search over several hyperparameters.

## Setup

1. Create and activate a Python environment:

```bash
python -m venv .venv
# PowerShell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify that a MuJoCo environment can be created:

```bash
python -c "import gymnasium as gym; gym.make('Hopper-v4')"
```

## How to run

Run the method comparison experiments (TD / GAE / MC):

```bash
python run_exp.py
```

Run grid search:

```bash
python grid_search.py
```

`run_exp.py` saves curves and raw returns in `results/`.  
`grid_search.py` saves detailed logs and best settings in `grid_results/`.

## Methods

- **1-step TD (`lambda=0`)**: uses one-step bootstrapping, usually with lower variance and higher bias.
- **GAE (`lambda=0.95`)**: provides a practical bias-variance tradeoff.
- **Monte Carlo (`lambda=1`)**: uses long-horizon returns, usually with lower bias and higher variance.

## Environments

The experiments use:

- `Swimmer-v4`
- `HalfCheetah-v4`
- `Hopper-v4`

## File structure

- `a2c.py` - core A2C implementation
- `run_exp.py` - runs the TD / GAE / Monte Carlo comparison
- `grid_search.py` - runs the hyperparameter grid search
- `grid_results/` - full grid logs per environment and `best_params.json`

## Notes

- This implementation uses a vectorized wrapper to run multiple environments in parallel.
- The current grid search varies `num_envs`, `pi_lr`, `v_lr`, and `lam`. `gamma` is currently fixed at `0.99` in `grid_search.py`.
- `best_params.json` stores the best configuration per environment based on the average return over the last part of training used in `grid_search.py`.

## Results Summary

- `Swimmer-v4`: all three methods improve steadily, and 1-step TD reaches the highest final return in this run.
- `HalfCheetah-v4`: all three methods improve from strongly negative returns, and 1-step TD finishes above GAE and Monte Carlo in this run.
- `Hopper-v4`: GAE reaches the best final performance, and Monte Carlo also finishes above 1-step TD.

Overall, all three methods learn reasonable policies on the selected MuJoCo tasks, but their relative performance depends on the environment.

These results are based on single runs, so some variation is expected due to training randomness.
