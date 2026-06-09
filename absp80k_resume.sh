#!/usr/bin/env bash
# RESUME driver after the reboot: re-run only the 6 incomplete (method,seed) jobs.
# 16/20 already complete (policy zip on disk) -> --skip-existing skips them as a safety net.
# This script lives in the repo (not /tmp) so a future reboot does not wipe it; the on-disk
# results + --skip-existing make it idempotent (safe to re-run any time).
set -u
cd /home/dook/PhD-thesis/code/time-delay/Predictive-Model-Delay-Correction || exit 1

OP=operator_models/FetchSlide-v2_SAC_seed0.zip
LOGDIR=/tmp/absp80k_logs; mkdir -p "$LOGDIR"
DRV="$LOGDIR/driver.log"
MAXJOBS=4
STEPS=80000
DELAY=250-290
PMDC_CFG="PMDC_FIX_PREVOBS=1 PMDC_SAC_GAMMA=0.95 PMDC_SAC_TENT=-1"

echo "[$(date '+%F %H:%M:%S')] RESUME DRIVER START pid=$$ (6 jobs)" >> "$DRV"

# name|outdir|algo|seed|extra_env  (2 ABSP first = both heavy jobs run concurrently from t=0)
jobs=(
  "absp_s3|runs_FetchSlide_80k_absp|PMDC|3|$PMDC_CFG PMDC_PRED_MODE=absp"
  "absp_s4|runs_FetchSlide_80k_absp|PMDC|4|$PMDC_CFG PMDC_PRED_MODE=absp"
  "sbsp_s4|runs_FetchSlide_80k|PMDC|4|$PMDC_CFG PMDC_PRED_MODE=sbsp"
  "sac_s3|runs_FetchSlide_80k|SAC|3|"
  "asac_s4|runs_FetchSlide_80k|A-SAC|4|"
  "sac_s4|runs_FetchSlide_80k|SAC|4|"
)

running=0
for spec in "${jobs[@]}"; do
  IFS='|' read -r name outdir algo seed envs <<< "$spec"
  (
    env $envs CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 delay_correcting_training.py \
        --env-id FetchSlide-RemotePDNorm-v0 --delay-ranges "$DELAY" \
        --algorithms "$algo" --steps "$STEPS" --seed "$seed" --device cuda \
        --operator-model "$OP" --output-dir "$outdir" --skip-existing \
        > "$LOGDIR/$name.log" 2>&1
    echo "[$(date '+%F %H:%M:%S')] DONE  $name rc=$?" >> "$DRV"
  ) &
  echo "[$(date '+%F %H:%M:%S')] START $name (pid $!) algo=$algo seed=$seed out=$outdir" >> "$DRV"
  running=$((running+1))
  if (( running >= MAXJOBS )); then wait -n; running=$((running-1)); fi
done
wait
echo "[$(date '+%F %H:%M:%S')] RESUME DRIVER ALL DONE" >> "$DRV"
