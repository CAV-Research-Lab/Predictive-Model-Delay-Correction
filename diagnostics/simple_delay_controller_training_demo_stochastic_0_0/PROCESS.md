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
  "stochastic_uppers": [
    0
  ],
  "stochastic_mode": "both",
  "matrix": "stochastic_augmented",
  "augmented_delay_magnitude_variant": "augmented_action_delay",
  "variants": [
    "unseen",
    "augmented_action",
    "augmented_action_delay"
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "iterations": 24,
  "population": 18,
  "elites": 5,
  "train_episodes_per_candidate": 2,
  "eval_episodes": 8,
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

## Interpretation Notes

- For constant delays, `+action buffer` and `+action buffer + delay values` are expected to overlap because the sampled delay is fixed and adds no extra runtime information.
- The stochastic comparisons use stochastic action and observation ranges when `--stochastic-mode both` is selected.
- A range such as `(0-20,0-20)` has lower mean delay than fixed `(20,20)`, so this plot tests widening stochastic ranges rather than proving every stochastic setting is harder than every constant setting.
