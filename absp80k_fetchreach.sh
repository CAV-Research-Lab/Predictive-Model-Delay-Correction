#!/usr/bin/env bash
# FetchReach comparison: ABSP vs SBSP vs A-SAC vs SAC on FetchReach-RemotePDNorm-v0 @250-290ms,
# 80k steps, 8 seeds (0-7). Mirrors absp80k_master.sh (FetchSlide). Separate output dirs from
# FetchSlide so nothing collides (Pitfall K). Idempotent via --skip-existing.
set -u
cd /home/dook/PhD-thesis/code/time-delay/Predictive-Model-Delay-Correction || exit 1

OP=operator_models/FetchReach-v2_SAC_seed0.zip
LOGDIR=/tmp/fetchreach80k_logs; mkdir -p "$LOGDIR"
DRV="$LOGDIR/driver.log"
MAXJOBS=5
STEPS=80000
DELAY=250-290
ENVID=FetchReach-RemotePDNorm-v0
PMDC_CFG="PMDC_FIX_PREVOBS=1 PMDC_SAC_GAMMA=0.95 PMDC_SAC_TENT=-1"

echo "[$(date '+%F %H:%M:%S')] FETCHREACH DRIVER START pid=$$ maxjobs=$MAXJOBS seeds=0-7" >> "$DRV"

SEEDS=(0 1 2 3 4 5 6 7)
ABSP=("${SEEDS[@]}"); SBSP=("${SEEDS[@]}"); ASAC=("${SEEDS[@]}"); SAC=("${SEEDS[@]}")
jobs=()
while (( ${#ABSP[@]} + ${#SBSP[@]} + ${#ASAC[@]} + ${#SAC[@]} > 0 )); do
  if (( ${#ABSP[@]} )); then s=${ABSP[0]}; ABSP=("${ABSP[@]:1}"); jobs+=("absp_s$s|runs_FetchReach_80k_absp|PMDC|$s|$PMDC_CFG PMDC_PRED_MODE=absp"); fi
  if (( ${#SBSP[@]} )); then s=${SBSP[0]}; SBSP=("${SBSP[@]:1}"); jobs+=("sbsp_s$s|runs_FetchReach_80k|PMDC|$s|$PMDC_CFG PMDC_PRED_MODE=sbsp"); fi
  if (( ${#ASAC[@]} )); then s=${ASAC[0]}; ASAC=("${ASAC[@]:1}"); jobs+=("asac_s$s|runs_FetchReach_80k|A-SAC|$s|"); fi
  if (( ${#SAC[@]}  )); then s=${SAC[0]};  SAC=("${SAC[@]:1}");   jobs+=("sac_s$s|runs_FetchReach_80k|SAC|$s|"); fi
done

running=0
for spec in "${jobs[@]}"; do
  IFS='|' read -r name outdir algo seed envs <<< "$spec"
  (
    env $envs CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 delay_correcting_training.py \
        --env-id "$ENVID" --delay-ranges "$DELAY" \
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
echo "[$(date '+%F %H:%M:%S')] FETCHREACH DRIVER ALL DONE" >> "$DRV"
