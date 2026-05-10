# AD-MPC2: Adaptive-Horizon with TD-MPC2 Extension

Read `experiments.ipynb` for reproducing our experiments.

## Method Summary
This extension adds an adaptive rollout horizon to TD-MPC2. The baseline uses a fixed model rollout horizon `horizon`; the extension rolls to `h_max` when adaptive horizon is enabled, then selects an effective horizon `H_t` from candidate depths bounded by `[h_min, h_max]`.

The selection signal is a cheap model-value inconsistency proxy. For the same latent rollout, the agent computes bootstrapped return estimates at depths `1`, `3`, `5`, and `h_max` when those depths are within the rollout. It measures each candidate depth by its absolute deviation from the mean of the recorded return estimates, selects the candidate with the smallest inconsistency, and logs the dispersion across candidates as model-value inconsistency. In the current implementation, `inconsistency_threshold` and `inconsistency_patience` are config placeholders and are not used for first-threshold-crossing truncation.

The extension also adds an optional behavior regularization term for policy improvement. When enabled, it penalizes deviation between policy mean actions and replay-buffer actions at matching latent states.

## New Config Options
```yaml
adaptive_horizon: false
h_min: 1
h_max: 5
inconsistency_threshold: 1.0
inconsistency_patience: 1
behavior_reg_coef: 0
```

- `adaptive_horizon`: Enables adaptive horizon selection when `true`; fixed-horizon behavior is preserved when `false`.
- `h_min`: Minimum selected rollout horizon.
- `h_max`: Maximum rollout horizon used for adaptive planning/training.
- `inconsistency_threshold`: Reserved for threshold-based horizon selection; not used by the current selector.
- `inconsistency_patience`: Reserved for threshold-based horizon selection; not used by the current selector.
- `behavior_reg_coef`: Coefficient for replay-action behavior regularization. Default `0` keeps baseline policy behavior unchanged.

## Exact Smoke-Test Commands
These are intentionally short and are not benchmark runs.

```bash
cd tdmpc2
python train.py --config-name experiments/adaptive_horizon/cartpole_swingup_sanity steps=20 eval_freq=10 eval_episodes=1 seed=1 enable_wandb=false save_csv=false save_video=false save_agent=false compile=false
python train.py --config-name experiments/adaptive_horizon/finger_spin_sanity steps=20 eval_freq=10 eval_episodes=1 seed=1 enable_wandb=false save_csv=false save_video=false save_agent=false compile=false
python train.py --config-name experiments/adaptive_horizon/cheetah_run_benchmark steps=20 eval_freq=10 eval_episodes=1 seed=1 enable_wandb=false save_csv=false save_video=false save_agent=false compile=false
python train.py --config-name experiments/adaptive_horizon/walker_walk_benchmark steps=20 eval_freq=10 eval_episodes=1 seed=1 enable_wandb=false save_csv=false save_video=false save_agent=false compile=false
```

The same commands can be printed from the repository root:

```bash
bash scripts/phase5_smoke_commands.sh
```

## Exact Short Experiment Commands
These are short diagnostic runs, not final benchmarks.

```bash
cd tdmpc2
python train.py --config-name experiments/adaptive_horizon/cartpole_swingup_sanity steps=100000 eval_freq=10000 eval_episodes=5 seed=1
python train.py --config-name experiments/adaptive_horizon/finger_spin_sanity steps=100000 eval_freq=10000 eval_episodes=5 seed=1
python train.py --config-name experiments/adaptive_horizon/cheetah_run_benchmark steps=1000000 eval_freq=25000 eval_episodes=10 seed=1
python train.py --config-name experiments/adaptive_horizon/walker_walk_benchmark steps=1000000 eval_freq=25000 eval_episodes=10 seed=1
```

## Expected Log Files And Plots
Expected local run outputs follow the existing TD-MPC2 work directory convention:

```text
logs/<task>/<seed>/<exp_name>/
```

If CSV logging is enabled, evaluation returns are written to:

```text
logs/<task>/<seed>/<exp_name>/eval.csv
```

Expected WandB metrics and plots:

- Return curve: `train/episode_reward`, `eval/episode_reward`.
- Selected horizon distribution: histogram or time series of `train/rollout_horizon` and `train/plan_rollout_horizon`.
- Model-value inconsistency curve: `train/model_value_inconsistency`, `train/model_value_inconsistency_depth_1`, `train/model_value_inconsistency_depth_3`, `train/model_value_inconsistency_depth_5`, `train/model_value_inconsistency_depth_h`.
- Horizon collapse checks: `train/horizon_at_h_min`, `train/horizon_at_h_max`.
- Sample efficiency: reward versus `train/step`.
- Wall-clock throughput: `train/elapsed_time`, `train/steps_per_second`.
- Value overestimation proxy: `train/value_overestimation_proxy`.
- Behavior constraint: `train/behavior_reg_loss`.

## Known Limitations
- The inconsistency proxy is heuristic and uses dispersion among bootstrapped returns, not an ensemble uncertainty estimate.
- The default threshold is a starting point and likely needs task-specific tuning.
- Horizon selection currently logs aggregate training-step metrics, not per-environment-step histograms.
- The behavior regularizer uses replay actions as a simple support proxy and is not a learned behavior model.
- Smoke commands are for syntax/runtime sanity only and should not be interpreted as performance results.
