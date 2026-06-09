# Simple Delay-Type Demonstration

This run uses a minimal 2D point-reaching environment to isolate the delay mechanism.
The same random-delay wrappers used by the Fetch experiments are applied around the environment.

## Setup

- Environment state: `[x, y, goal_x, goal_y]`.
- Action: clipped 2D velocity.
- Reward: negative Euclidean distance to the goal.
- Evaluation metric: true final Euclidean distance from the base environment state.
- Controller: proportional controller over the estimated current state.
- Unseen delay observes only the delayed state.
- Augmented (+action buffer) reconstructs state from delayed state plus recent actions using the nominal delay.
- Augmented (+action buffer + delay values) reconstructs using the sampled delay values exposed by the wrapper.

## Parameters

```json
{
  "environment": "PointReachEnv",
  "delay_wrapper": "wrappers_rd.RandomDelayWrapper via UnseenRandomDelayWrapper/AugmentedRandomDelayWrapper/AugmentedDelayInfoWrapper",
  "state": "[x, y, goal_x, goal_y]",
  "action": "2D clipped velocity",
  "metric": "true final Euclidean goal distance from the base environment state",
  "constant_delays": [
    0,
    5
  ],
  "stochastic_observation_uppers": [
    0,
    5
  ],
  "state_constant_delay": 5,
  "state_stochastic_upper": 5,
  "seeds": [
    0,
    1
  ],
  "episodes_per_seed": 3,
  "horizon": 80,
  "assumption": "This is a controlled delay-mechanism demonstration, not a replacement for Fetch policy training."
}
```

## Outputs

- `delay_length_impact.png`: constant and stochastic delay sweeps.
- `state_information.png`: unseen vs augmented state information at fixed delay settings.
- `results.csv`: per-episode results.
- `summary.csv`: IQM/IQR summary used for the plots.
