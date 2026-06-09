# Simple Delay Training-Curve Diagnostics

These diagnostics complement `simple_delay_type_demo.py` by plotting reward
over training time for delayed point-reaching.  They are intended to isolate the
delay mechanism, not to replace the full Fetch experiment matrix.

## Fast Controller Optimisation

The clean mechanism plots were generated with:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python simple_delay_controller_training_demo.py \
  --output-dir diagnostics/simple_delay_controller_training_demo \
  --seeds 0 1 2 \
  --iterations 24 \
  --population 18 \
  --elites 5 \
  --train-episodes 2 \
  --eval-episodes 8 \
  --stochastic-mode both
```

This trains a small proportional controller with cross-entropy search.  The
policy parameters are the proportional gain and action-history correction
scale.  The same random-delay wrappers are used for unseen, action-buffer, and
action-buffer-plus-delay-value observations.

Outputs:

- `diagnostics/simple_delay_controller_training_demo/reward_augmented_delay_magnitude.png`
- `diagnostics/simple_delay_controller_training_demo/reward_augmented_stochastic_delay_magnitude.png`
- `diagnostics/simple_delay_controller_training_demo/reward_state_information_10_10.png`
- `diagnostics/simple_delay_controller_training_demo/reward_state_information_stochastic_0_20.png`
- `diagnostics/simple_delay_controller_training_demo/reward_state_information_stochastic_0_40.png`
- `diagnostics/simple_delay_controller_training_demo/reward_constant_vs_stochastic.png`
- Matching `distance_*` plots.
- `diagnostics/simple_delay_controller_training_demo/results.csv`
- `diagnostics/simple_delay_controller_training_demo/summary.csv`
- `diagnostics/simple_delay_controller_training_demo/PROCESS.md`

The plots use Times-style fonts and IQM curves with IQR bands.

The stochastic state-information comparison at `(0-40,0-40)` was generated
separately from:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python simple_delay_controller_training_demo.py \
  --output-dir diagnostics/simple_delay_controller_training_demo_stochastic_0_40 \
  --matrix stochastic_state_info \
  --stochastic-state-upper 40 \
  --stochastic-uppers 40 \
  --seeds 0 1 2 \
  --iterations 24 \
  --population 18 \
  --elites 5 \
  --train-episodes 2 \
  --eval-episodes 8 \
  --stochastic-mode both
```

The stochastic augmented-state companion plot compares `(0-0,0-0)`,
`(0-10,0-10)`, and `(0-20,0-20)`.  The missing zero-delay curve was generated
with:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python simple_delay_controller_training_demo.py \
  --output-dir diagnostics/simple_delay_controller_training_demo_stochastic_0_0 \
  --matrix stochastic_augmented \
  --stochastic-uppers 0 \
  --seeds 0 1 2 \
  --iterations 24 \
  --population 18 \
  --elites 5 \
  --train-episodes 2 \
  --eval-episodes 8 \
  --stochastic-mode both
```

The final plot is generated from
`diagnostics/simple_delay_controller_training_demo/stochastic_augmented_results.csv`.

Zoomed 20k-step versions were regenerated without rerunning training:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python simple_delay_controller_training_demo.py \
  --plot-only \
  --results-csv diagnostics/simple_delay_controller_training_demo/results.csv \
  --output-dir diagnostics/simple_delay_controller_training_demo_20k \
  --max-plot-steps 20000 \
  --stochastic-mode both
```

Those plots are under `diagnostics/simple_delay_controller_training_demo_20k/`.

## Main Findings

- For augmented state with constant delays, increasing delay from `(10,10)` to
  `(15,15)` and `(20,20)` delays convergence and lowers final return.
- At fixed `(10,10)`, unseen delay converges to a worse return than the
  augmented variants.
- At fixed constant delays, `+action buffer` and `+action buffer + delay
  values` overlap because the delay is fixed, so the sampled delay values add no
  new runtime information.
- Wider stochastic ranges are detrimental: `(0-20,0-20)` is worse than
  `(0-10,0-10)`.  They are not worse than every fixed constant-delay setting in
  this toy setup because stochastic ranges have lower mean delay than their
  upper bound.

## SAC Diagnostic

A short one-seed SAC version was also run:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python simple_delay_training_demo.py \
  --output-dir diagnostics/simple_delay_training_demo_requested \
  --preset requested \
  --seeds 0 \
  --steps 10000 \
  --eval-every 1000 \
  --n-eval-episodes 5 \
  --learning-starts 100 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --torch-threads 1
```

This writes the same requested plot names under
`diagnostics/simple_delay_training_demo_requested/`, but the short one-seed SAC
curves are noisy and should not be used as the main evidence.  In particular,
the stochastic unseen curve can improve sharply in one seed, so SAC needs more
seeds and a longer budget before drawing conclusions.

## Why The Curves Have Discrete Points

These plots aggregate evaluation rollouts at fixed training checkpoints.  They
are not raw per-step or per-episode training reward traces.  The controller
diagnostic evaluates after each optimisation iteration; the SAC diagnostic
evaluates after each `--eval-every` chunk.  The line between points is therefore
an interpolation between checkpoint evaluations.

W&B training plots can look more continuous because they usually show many raw
training log records, often smoothed by the UI.  To reproduce that exactly, the
training loop needs to log per-episode or per-step training rewards and the
plotter should read those raw history rows instead of checkpoint evaluation
summaries.
