#!/usr/bin/env bash
#
# PMDC method comparison (ABSP vs SBSP vs A-SAC vs SAC) as a SLURM job array, driving
# delay_correcting_training.py (NOT the run_fetch_delay_pipeline.py augmented/unseen sweep).
# One array task per (method, seed). Logs ALL metrics (track/step_reward, wm/*, train/*) to
# W&B via sync_tensorboard, so plots can be rebuilt from W&B alone.
#
# Defaults: FetchReach-RemotePDNorm-v0 @250-290ms, 8 seeds, 80k steps -> 4 methods x 8 = 32 tasks.
#
# Submit (32 tasks, max 8 concurrent):
#   sbatch --array=0-31%8 slurm/run_pmdc_comparison.sbatch
# Override env / project / entity / seeds:
#   sbatch --array=0-31%8 --export=ALL,WANDB_PROJECT=pmdc-fetchreach,WANDB_ENTITY=cavlab slurm/run_pmdc_comparison.sbatch
#   sbatch --array=0-31%8 --export=ALL,ENV_ID=FetchSlide-RemotePDNorm-v0,OPERATOR=operator_models/FetchSlide-v2_SAC_seed0.zip slurm/run_pmdc_comparison.sbatch
# Fewer seeds (e.g. 5): set SEEDS_PER_METHOD=5 and array 0-19:
#   sbatch --array=0-19%8 --export=ALL,SEEDS_PER_METHOD=5 slurm/run_pmdc_comparison.sbatch
#
#SBATCH --job-name=pmdc-cmp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"
VENV_PYTHON="${VENV_PYTHON:-.venv/bin/python}"

ENV_ID="${ENV_ID:-FetchReach-RemotePDNorm-v0}"
DELAY="${DELAY:-250-290}"
STEPS="${STEPS:-80000}"
SEEDS_PER_METHOD="${SEEDS_PER_METHOD:-8}"
DEVICE="${DEVICE:-cuda}"
OPERATOR="${OPERATOR:-operator_models/FetchReach-v2_SAC_seed0.zip}"

WANDB_PROJECT="${WANDB_PROJECT:-pmdc-fetchreach}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-${ENV_ID}_${DELAY}ms}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# Per-env output base (keep separate from other envs; Pitfall K). ABSP gets its own subtree so
# it does not collide with SBSP -- both have algorithm == PMDC.
SAFE_ENV="${ENV_ID%-RemotePDNorm-v0}"
OUT_BASE="${OUT_BASE:-runs_${SAFE_ENV}_80k}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
METHOD_IDX=$(( TASK / SEEDS_PER_METHOD ))
SEED=$(( TASK % SEEDS_PER_METHOD ))

# Method dispatch: 0=ABSP 1=SBSP 2=A-SAC 3=SAC.
# PMDC runs use the W4 FAIR config (PMDC-side improvements only; SAC kept at the same
# defaults as the A-SAC/SAC baselines -- see PMDC_TUNING_LOG.md). The old SAC-side tweaks
# (gamma 0.95, tent -1) are deliberately NOT used. PMDC_SAVE_WM persists the (policy, WM)
# pair for valid offline / on-distribution diagnostics.
W4_CFG=(PMDC_FIX_PREVOBS=1 PMDC_WM_NORM=1 PMDC_WM_LR=1e-3 PMDC_CLIP_STATE=10 PMDC_WM_DELTA=1 PMDC_WM_WARMUP=10000 PMDC_SAVE_WM=1)
PMDC_ENV=()
case "$METHOD_IDX" in
  0) ALGO=PMDC;  OUTDIR="${OUT_BASE}_absp"; PMDC_ENV=(PMDC_PRED_MODE=absp "${W4_CFG[@]}") ;;
  1) ALGO=PMDC;  OUTDIR="${OUT_BASE}";      PMDC_ENV=(PMDC_PRED_MODE=sbsp "${W4_CFG[@]}") ;;
  2) ALGO=A-SAC; OUTDIR="${OUT_BASE}";      PMDC_ENV=() ;;
  3) ALGO=SAC;   OUTDIR="${OUT_BASE}";      PMDC_ENV=() ;;
  *) echo "bad METHOD_IDX $METHOD_IDX (task $TASK, SEEDS_PER_METHOD $SEEDS_PER_METHOD)" >&2; exit 2 ;;
esac

echo "task=$TASK -> method=$ALGO seed=$SEED env=$ENV_ID out=$OUTDIR delay=$DELAY"

ARGS=(
  --env-id "$ENV_ID" --delay-ranges "$DELAY" --algorithms "$ALGO"
  --steps "$STEPS" --seed "$SEED" --device "$DEVICE"
  --operator-model "$OPERATOR" --output-dir "$OUTDIR" --skip-existing
  --wandb --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" --wandb-group "$WANDB_GROUP"
)
[[ -n "$WANDB_ENTITY" ]] && ARGS+=(--wandb-entity "$WANDB_ENTITY")

# ${PMDC_ENV[@]+...} expands to nothing when the array is empty (A-SAC/SAC) without tripping
# `set -u` on older bash (4.2), and to the quoted VAR=val tokens for ABSP/SBSP.
exec env ${PMDC_ENV[@]+"${PMDC_ENV[@]}"} OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  "$VENV_PYTHON" delay_correcting_training.py "${ARGS[@]}"
