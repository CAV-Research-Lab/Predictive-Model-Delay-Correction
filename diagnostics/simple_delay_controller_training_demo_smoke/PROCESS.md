# Simple Delay Controller-Training Demonstration

This run optimises a small proportional reaching controller with cross-entropy search.
It is intended to show convergence effects from delay and state information without requiring long SAC training.

## Parameters

```json
{
  "environment": "PointReachEnv",
  "trainer": "Cross-entropy optimisation of a proportional controller",
  "policy_parameters": "kp and action-history correction scale",
  "delay_wrapper": "wrappers_rd.RandomDelayWrapper via unseen/action-buffer/action-buffer-plus-delay variants",
  "reward_curve_metric": "IQM true evaluation return with IQR bands; higher is better",
  "distance_curve_metric": "IQM mean goal distance with IQR bands; lower is better",
  "constant_delays": [
    10,
    15,
    20
  ],
  "stochastic_observation_uppers": [
    10,
    20
  ],
  "augmented_delay_magnitude_variant": "augmented_action_delay",
  "variants": [
    "unseen",
    "augmented_action",
    "augmented_action_delay"
  ],
  "seeds": [
    0
  ],
  "iterations": 2,
  "population": 4,
  "elites": 2,
  "train_episodes_per_candidate": 1,
  "eval_episodes": 2,
  "horizon": 80,
  "assumption": "This is a fast delay-mechanism diagnostic, not a replacement for neural SAC policy training."
}
```

## Outputs

- `results.csv`: per-evaluation-episode rows.
- `summary.csv`: IQM/IQR summaries for reward, distance, and success.
- `reward_augmented_delay_magnitude.png`: augmented-state reward convergence for `(10,10)`, `(15,15)`, and `(20,20)`.
- `reward_state_information_10_10.png`: unseen vs augmented reward convergence at `(10,10)`.
- `reward_constant_vs_stochastic.png`: constant-vs-stochastic reward convergence by method.
- `distance_*`: matching true-distance versions of the requested reward plots.
