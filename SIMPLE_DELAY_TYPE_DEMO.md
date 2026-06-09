# Simple Delay-Type Demonstration

This demonstration is a controlled sanity check for the delay wrappers, not a
replacement for the full FetchPush local-remote experiments.  It uses a small
2D point-reaching environment so the qualitative delay story is easy to inspect:

- unseen delayed state performs worst;
- augmenting the state with recent actions improves control;
- adding sampled delay values helps most under stochastic delays;
- increasing constant delay length increases error;
- increasing stochastic observation-delay range increases error.

## Environment

The script `simple_delay_type_demo.py` defines `PointReachEnv`, a 2D velocity
control task with state `[x, y, goal_x, goal_y]`, negative Euclidean distance
reward, and small process noise.  The environment is wrapped using the same
random-delay wrappers as the Fetch experiments:

- `UnseenRandomDelayWrapper`
- `AugmentedRandomDelayWrapper`
- `AugmentedDelayInfoWrapper`

The controller is deliberately simple: a proportional controller over the
estimated current state.  The unseen controller acts on the delayed observation.
The augmented controllers reconstruct state from delayed observation plus recent
actions; the delay-info variant also uses the sampled delay values.

## Command

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python simple_delay_type_demo.py \
  --output-dir diagnostics/simple_delay_type_demo
```

This defaults to 10 seeds, 20 episodes per seed, IQM/IQR summaries, Times-style
plotting, constant delays `(0,0)` through `(20,20)`, and stochastic observation
upper bounds `(0,0)` through `(0,0-20)`.

## Outputs

- `diagnostics/simple_delay_type_demo/delay_length_impact.png`
- `diagnostics/simple_delay_type_demo/state_information.png`
- `diagnostics/simple_delay_type_demo/results.csv`
- `diagnostics/simple_delay_type_demo/summary.csv`
- `diagnostics/simple_delay_type_demo/PROCESS.md`

The plotted metric is IQM mean goal distance, computed from the true base
environment state rather than from delayed reward.

## Result Snapshot

From `diagnostics/simple_delay_type_demo/summary.csv`:

- Constant `(15,15)`: unseen `1.856`, action-buffer `0.560`, action-buffer plus
  delay values `0.560`.
- Stochastic `(0,0-20)`: unseen `0.388`, action-buffer `0.277`,
  action-buffer plus delay values `0.260`.
- Constant delay sweep, unseen: `0.259 -> 2.465` from delay `0` to `20`.
- Stochastic sweep, unseen: `0.259 -> 0.388` from upper bound `0` to `20`.

Lower values are better.  The demo therefore gives the intended didactic story:
state information reduces delay-induced degradation, while increasing delay
length or stochastic delay range worsens performance.

## Fetch Metric Fix

The Fetch evaluation path now logs explicit goal-distance and success columns for
future runs when the underlying Fetch environment exposes achieved and desired
goals:

- `episode_final_goal_distance`
- `episode_min_goal_distance`
- `episode_success`

For Fetch environments this avoids treating sparse reward averages as physical
tracking distance.
