# FetchPush Delay Experiments

This runner recreates the augmented-vs-unseen delay comparisons for
`FetchPush-RemotePDNorm-v0`. Delay pairs are written as
`(action delay, observation delay)` in environment steps.
`FetchPush-RemotePDNorm-v0` is the local/remote PD-gain environment registered
on top of `FetchPush-v2`.

## Variants

- `unseen`: delayed observation only.
- `augmented_action`: delayed observation plus action buffer.
- `augmented_action_delay`: delayed observation plus action buffer and sampled delay values.

## Default Matrix

- Constant delay length impact: `(0,0)`, `(5,5)`, `(10,10)`, `(15,15)`, `(20,20)`.
- Stochastic delay length impact: `(0,0-5)`, `(0,0-10)`, `(0,0-15)`, `(0,0-20)`.
- State information comparison at constant `(20,20)`.
- State information comparison at stochastic `(0,0-20)`.
- Action-only vs observation-only delays: `(0,0)`, `(10,0)`, `(20,0)`, `(30,0)`, `(40,0)` and `(0,0)`, `(0,10)`, `(0,20)`, `(0,30)`, `(0,40)`.
- Train-test generalisation:
  - train constant `(10,10)`, evaluate `(0,0)`, `(5,5)`, `(10,10)`, `(15,15)`, `(20,20)`;
  - train stochastic `(0,0-10)`, evaluate `(0,0-5)`, `(0,0-10)`, `(0,0-15)`, `(0,0-20)`.

For one seed, the default CLI expands to 57 unique train jobs and 90 evaluation jobs.

## Preflight Smoke Test

Before launching the long run, check the matrix expansion, wrapper dimensions,
resume behavior, sharded CSV writing, multi-CSV plotting, and coverage checker:

```bash
.venv/bin/python smoke_fetch_delay_pipeline.py
```

This command writes only temporary files under `/tmp` unless `--keep-dir` is used.
It trains a few 5-step CPU policies to test the pipeline mechanics; the results
are not meaningful experiment data.

## Full Run

Preferred single-machine entrypoint:

```bash
.venv/bin/python run_fetch_delay_pipeline.py \
  --seeds 0 1 2 3 4 \
  --steps 80000 \
  --n-eval-episodes 20 \
  --device cuda \
  --wandb \
  --wandb-project fetch-delay \
  --wandb-run-name-prefix "FetchPush-RemotePDNorm-v0 delay 80000 steps"
```

This runs the preflight smoke test, trains/evaluates the matrix, generates the
three plots and numeric plot summary, then runs the strict coverage check.
Use `--skip-smoke` after the preflight has already passed. With `--wandb`,
completed evaluation jobs are logged directly to W&B as they finish, and the
plotter uploads the three aggregate figures plus the IQM/IQR summary table.

The lower-level training/evaluation command is:

```bash
.venv/bin/python fetch_delay_experiments.py \
  --seeds 0 1 2 3 4 \
  --steps 80000 \
  --n-eval-episodes 20 \
  --device cuda
```

The runner writes checkpoints under `fetch_delay_runs/models/` and appends
episode-level evaluation rows to `fetch_delay_runs/evaluations.csv`.
It also writes `fetch_delay_runs/run_manifest.json`, or
`run_manifest_shardX-of-Y.json` for sharded runs, recording the exact train/eval
matrix used.
This checkout has been verified with `.venv` using `torch==2.7.1+cu126` on the
local CUDA 12.6 driver, so `--device cuda` is the expected launch mode here.
CSV reads and writes are file-locked, so sharded workers can safely append to
the same `evaluations.csv` on one filesystem.
`--eval-only` preserves an existing manifest by default, so resume/evaluation
workers do not overwrite the original train/eval matrix description.

Use `--dry-run` first to inspect the exact train/eval matrix without training:

```bash
.venv/bin/python fetch_delay_experiments.py --dry-run
```

To split the train jobs across several workers, use deterministic sharding:

```bash
.venv/bin/python fetch_delay_experiments.py --shard-count 4 --shard-index 0 --device cuda
.venv/bin/python fetch_delay_experiments.py --shard-count 4 --shard-index 1 --device cuda
.venv/bin/python fetch_delay_experiments.py --shard-count 4 --shard-index 2 --device cuda
.venv/bin/python fetch_delay_experiments.py --shard-count 4 --shard-index 3 --device cuda
```

To run local shards sequentially through the orchestrator and then plot/check:

```bash
.venv/bin/python run_fetch_delay_pipeline.py \
  --run-shards-locally 4 \
  --seeds 0 1 2 3 4 \
  --steps 80000 \
  --n-eval-episodes 20 \
  --device cuda
```

The orchestrator can also run one deterministic shard, which is useful for
Slurm arrays:

```bash
.venv/bin/python run_fetch_delay_pipeline.py \
  --shard-count 57 \
  --shard-index 0 \
  --seeds 0 1 2 3 4 \
  --steps 80000 \
  --n-eval-episodes 20 \
  --device cuda \
  --skip-smoke \
  --skip-plot \
  --skip-check \
  --wandb
```

After all shards finish, generate plots, run coverage over all shard manifests,
and upload the aggregate W&B plot run:

```bash
.venv/bin/python run_fetch_delay_pipeline.py \
  --skip-smoke \
  --skip-train \
  --manifest 'fetch_delay_runs/run_manifest_shard*-of-57.json' \
  --seeds 0 1 2 3 4 \
  --steps 80000 \
  --n-eval-episodes 20 \
  --device cuda \
  --wandb
```

To run the same full delay matrix on the original Gymnasium Robotics
`FetchPush-v2` task instead of the local/remote PD environment, override the
environment id and write to a separate output directory:

```bash
.venv/bin/python run_fetch_delay_pipeline.py \
  --env-id FetchPush-v2 \
  --output-dir fetch_delay_runs_FetchPush_v2 \
  --results-csv fetch_delay_runs_FetchPush_v2/evaluations.csv \
  --plots-dir fetch_delay_runs_FetchPush_v2/plots \
  --seeds 0 1 2 3 4 \
  --steps 80000 \
  --n-eval-episodes 20 \
  --device cuda \
  --wandb \
  --wandb-project fetch-delay \
  --wandb-run-name-prefix "FetchPush-v2 delay 80000 steps"
```

## Plots

```bash
.venv/bin/python plot_fetch_delay_experiments.py \
  --results-csv fetch_delay_runs/evaluations.csv \
  --output-dir fetch_delay_runs/plots \
  --prefix FetchPush
```

The plotter also accepts several CSVs or a glob, for example:

```bash
.venv/bin/python plot_fetch_delay_experiments.py \
  --results-csv 'fetch_delay_runs/evaluations_shard*.csv' \
  --output-dir fetch_delay_runs/plots \
  --prefix FetchPush
```

The plotter uses a Times New Roman style serif stack and aggregates results with
IQM centres and IQR bands/error bars. It writes three figures plus the numeric
summary table `FetchPush_plot_summary.csv`:

- `FetchPush_delay_length_impact.png`
- `FetchPush_state_information.png`
- `FetchPush_delay_structure_generalization.png`
- `FetchPush_plot_summary.csv`

## W&B Logging

To log directly into a W&B project, install/login once:

```bash
uv --cache-dir /tmp/uv-cache pip install --python .venv/bin/python wandb
.venv/bin/wandb login
```

The pipeline logs directly. `fetch_delay_experiments.py`
opens one W&B evaluation run per completed train/eval/seed job, and
`plot_fetch_delay_experiments.py` opens one aggregate plot run containing:

- `plots/delay_length_impact`
- `plots/state_information`
- `plots/delay_structure_generalization`
- `plots/iqm_iqr_summary`

Run names are informative, for example
`FetchPush-RemotePDNorm-v0 delay 80000 steps | FetchPush seed0 unseen stochastic_sweep/stochastic_observation_upper train=(0,0-20) eval=(0,0-20)`.

The preferred route is to let the orchestrator pass the direct W&B settings to
training/evaluation and plotting:

```bash
.venv/bin/python run_fetch_delay_pipeline.py \
  --seeds 0 1 2 3 4 \
  --steps 80000 \
  --n-eval-episodes 20 \
  --device cuda \
  --skip-smoke \
  --wandb \
  --wandb-project fetch-delay
```

To upload only aggregate plots from an existing CSV:

```bash
.venv/bin/python plot_fetch_delay_experiments.py \
  --results-csv fetch_delay_runs/evaluations.csv \
  --output-dir fetch_delay_runs/plots \
  --prefix FetchPush \
  --wandb \
  --wandb-project fetch-delay
```

## Slurm

The reusable Slurm wrapper is `slurm/run_fetch_delay_pipeline.sbatch`.

Run the full pipeline in one job:

```bash
sbatch slurm/run_fetch_delay_pipeline.sbatch
```

Run the 57 train-spec shards as a GPU array, limiting concurrency to 8 jobs:

```bash
sbatch --array=0-56%8 slurm/run_fetch_delay_pipeline.sbatch
```

Then submit the finalize job after the array succeeds:

```bash
sbatch --dependency=afterok:<array_job_id> \
  --export=ALL,RUN_MODE=finalize,SHARD_COUNT=57 \
  slurm/run_fetch_delay_pipeline.sbatch
```

Common overrides:

```bash
sbatch --export=ALL,SEEDS="0",WANDB_PROJECT=fetch-delay slurm/run_fetch_delay_pipeline.sbatch
sbatch --export=ALL,ENV_ID=FetchPush-v2 slurm/run_fetch_delay_pipeline.sbatch
sbatch --export=ALL,WANDB_MODE=offline slurm/run_fetch_delay_pipeline.sbatch
```

`ENV_ID` defaults to `FetchPush-RemotePDNorm-v0`. When `ENV_ID=FetchPush-v2`
is used and `OUTPUT_DIR` is not provided, the script writes to
`fetch_delay_runs_FetchPush_v2` so the original-environment results do not mix
with the local/remote results.

For a sharded original-environment run, pass the same `ENV_ID` to both the array
and finalize jobs:

```bash
sbatch --array=0-56%8 --export=ALL,ENV_ID=FetchPush-v2 slurm/run_fetch_delay_pipeline.sbatch
sbatch --dependency=afterok:<array_job_id> \
  --export=ALL,RUN_MODE=finalize,SHARD_COUNT=57,ENV_ID=FetchPush-v2 \
  slurm/run_fetch_delay_pipeline.sbatch
```

## Coverage Check

After training, evaluation, and plotting, validate that all requested rows and
plot files are present. The checker also validates the manifest coverage, plot
summary aggregate rows, finite IQM/IQR values, and referenced model files:

```bash
.venv/bin/python check_fetch_delay_results.py \
  --results-csv fetch_delay_runs/evaluations.csv \
  --manifest fetch_delay_runs/run_manifest.json \
  --plots-dir fetch_delay_runs/plots \
  --prefix FetchPush \
  --seeds 0 1 2 3 4 \
  --n-eval-episodes 20
```

For sharded runs, pass all shard manifests:

```bash
.venv/bin/python check_fetch_delay_results.py \
  --results-csv fetch_delay_runs/evaluations.csv \
  --manifest 'fetch_delay_runs/run_manifest_shard*.json' \
  --plots-dir fetch_delay_runs/plots \
  --prefix FetchPush
```

By default, the checker fails if shard manifests cover only part of the full
matrix or if duplicate shard manifests are supplied.
