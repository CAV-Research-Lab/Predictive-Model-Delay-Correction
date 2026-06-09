# Simple Delay Training-Curve Demonstration

This run trains SAC on the same delayed point-reaching setup used by `simple_delay_type_demo.py`.
The curves evaluate the current policy at regular training intervals and aggregate evaluation episodes/seeds with IQM/IQR.

## What The Curves Mean

- `reward_convergence.png` plots true base-environment evaluation return, so higher is better.
- `distance_convergence.png` plots true mean goal distance, so lower is better.
- `success_convergence.png` plots final-step success rate.
- The learner still receives the delayed reward emitted by the random-delay wrapper during training.

## Parameters

```json
{
  "environment": "PointReachEnv",
  "trainer": "Stable-Baselines3 SAC",
  "delay_wrapper": "wrappers_rd.RandomDelayWrapper via unseen/action-buffer/action-buffer-plus-delay variants",
  "reward_curve_metric": "IQM true evaluation return with IQR bands",
  "distance_curve_metric": "IQM mean goal distance with IQR bands",
  "experiments": [
    "constant"
  ],
  "variants": [
    "unseen",
    "augmented_action"
  ],
  "seeds": [
    0
  ],
  "steps": 1000,
  "eval_every": 500,
  "n_eval_episodes": 2,
  "constant_delay": [
    10,
    10
  ],
  "stochastic_delay": [
    0,
    "0-20"
  ],
  "horizon": 80,
  "action_scale": 0.08,
  "process_noise": 0.005,
  "goal_noise": 0.08,
  "success_threshold": 0.06,
  "assumption": "This is a fast convergence diagnostic for the delay mechanism, not a replacement for full Fetch training."
}
```

## Outputs

- `results.csv`: per-evaluation-episode rows.
- `summary.csv`: IQM/IQR summaries for reward, distance, and success.
- `reward_convergence.png`: reward convergence over training steps.
- `distance_convergence.png`: distance convergence over training steps.
- `success_convergence.png`: success-rate convergence over training steps.
