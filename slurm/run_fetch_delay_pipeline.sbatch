#!/usr/bin/env bash
#
# Full single-GPU run:
#   sbatch slurm/run_fetch_delay_pipeline.sbatch
#
# Parallel sharded run, then final plots/check/direct W&B plot upload:
#   sbatch --array=0-56%8 slurm/run_fetch_delay_pipeline.sbatch
#   sbatch --dependency=afterok:<array_job_id> --export=ALL,RUN_MODE=finalize,SHARD_COUNT=57 slurm/run_fetch_delay_pipeline.sbatch
#
# Useful overrides:
#   sbatch --export=ALL,SEEDS="0",STEPS=80000,WANDB_PROJECT=fetch-delay slurm/run_fetch_delay_pipeline.sbatch
#   sbatch --export=ALL,ENV_ID=FetchPush-v2 slurm/run_fetch_delay_pipeline.sbatch
#   sbatch --export=ALL,ENV_ID=FetchReach-v2 slurm/run_fetch_delay_pipeline.sbatch
#   sbatch --export=ALL,ENV_ID=FetchPush-RemoteDirect-v0 slurm/run_fetch_delay_pipeline.sbatch
#   sbatch --export=ALL,WANDB_MODE=offline slurm/run_fetch_delay_pipeline.sbatch
#
#SBATCH --job-name=fetch-delay
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

RUN_MODE="${RUN_MODE:-}"
if [[ -z "$RUN_MODE" ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    RUN_MODE="shard"
  else
    RUN_MODE="full"
  fi
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"

VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"
ENV_ID="${ENV_ID:-FetchPush-RemotePDNorm-v0}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "$ENV_ID" == "FetchPush-RemotePDNorm-v0" ]]; then
    OUTPUT_DIR="fetch_delay_runs"
  else
    SAFE_ENV_ID="${ENV_ID//-/_}"
    OUTPUT_DIR="fetch_delay_runs_${SAFE_ENV_ID}"
  fi
fi
RESULTS_CSV="${RESULTS_CSV:-${OUTPUT_DIR}/evaluations.csv}"
PLOTS_DIR="${PLOTS_DIR:-${OUTPUT_DIR}/plots}"
if [[ -z "${PREFIX:-}" ]]; then
  case "$ENV_ID" in
    FetchReach*) PREFIX="FetchReach" ;;
    FetchPush*) PREFIX="FetchPush" ;;
    FetchSlide*) PREFIX="FetchSlide" ;;
    FetchPickAndPlace*) PREFIX="FetchPickAndPlace" ;;
    *) PREFIX="${ENV_ID%-v0}" ;;
  esac
fi

SEEDS="${SEEDS:-0 1 2 3 4}"
STEPS="${STEPS:-80000}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-20}"
DEVICE="${DEVICE:-cuda}"
BUFFER_SIZE="${BUFFER_SIZE:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_STARTS="${LEARNING_STARTS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-fetch-delay}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-${ENV_ID}-${STEPS}steps-${N_EVAL_EPISODES}evaleps-${SEEDS// /_}}"
WANDB_RUN_NAME_PREFIX="${WANDB_RUN_NAME_PREFIX:-${ENV_ID} delay ${STEPS} steps}"
export WANDB_MODE

DEFAULT_SHARD_COUNT="${DEFAULT_SHARD_COUNT:-57}"
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  SHARD_COUNT="${SHARD_COUNT:-$SLURM_ARRAY_TASK_COUNT}"
elif [[ "$RUN_MODE" == "finalize" ]]; then
  SHARD_COUNT="${SHARD_COUNT:-$DEFAULT_SHARD_COUNT}"
else
  SHARD_COUNT="${SHARD_COUNT:-1}"
fi
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

read -r -a SEED_ARGS <<< "$SEEDS"

COMMON_ARGS=(
  --env-id "$ENV_ID"
  --output-dir "$OUTPUT_DIR"
  --results-csv "$RESULTS_CSV"
  --plots-dir "$PLOTS_DIR"
  --prefix "$PREFIX"
  --seeds "${SEED_ARGS[@]}"
  --steps "$STEPS"
  --n-eval-episodes "$N_EVAL_EPISODES"
  --device "$DEVICE"
  --buffer-size "$BUFFER_SIZE"
  --batch-size "$BATCH_SIZE"
  --learning-starts "$LEARNING_STARTS"
  --log-interval "$LOG_INTERVAL"
)

if [[ "${SKIP_SMOKE:-1}" == "1" ]]; then
  COMMON_ARGS+=(--skip-smoke)
fi

if [[ "$USE_WANDB" == "1" ]]; then
  COMMON_ARGS+=(
    --wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-mode "$WANDB_MODE"
    --wandb-group "$WANDB_GROUP"
    --wandb-run-name-prefix "$WANDB_RUN_NAME_PREFIX"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    COMMON_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_PIPELINE_ARGS:-}" ]]; then
  # Intentionally split like normal shell CLI flags, e.g. EXTRA_PIPELINE_ARGS='--metric episode_mean_distance'.
  # shellcheck disable=SC2206
  EXTRA_ARGS=($EXTRA_PIPELINE_ARGS)
fi

case "$RUN_MODE" in
  full)
    exec "$VENV_PYTHON" run_fetch_delay_pipeline.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
    ;;
  shard)
    exec "$VENV_PYTHON" run_fetch_delay_pipeline.py \
      "${COMMON_ARGS[@]}" \
      --shard-count "$SHARD_COUNT" \
      --shard-index "$SHARD_INDEX" \
      --skip-plot \
      --skip-check \
      "${EXTRA_ARGS[@]}"
    ;;
  finalize)
    exec "$VENV_PYTHON" run_fetch_delay_pipeline.py \
      "${COMMON_ARGS[@]}" \
      --skip-train \
      --manifest "${OUTPUT_DIR}/run_manifest_shard*-of-${SHARD_COUNT}.json" \
      "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unknown RUN_MODE: $RUN_MODE" >&2
    exit 2
    ;;
esac
