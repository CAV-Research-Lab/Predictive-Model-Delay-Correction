#!/usr/bin/env bash
# Wait for the FetchSlide master run to finish, THEN launch the FetchReach comparison, so the
# two ~26h GPU-bound runs do not contend (GPU is already saturated by one run). Lives in repo.
cd /home/dook/PhD-thesis/code/time-delay/Predictive-Model-Delay-Correction || exit 1
mkdir -p /tmp/fetchreach80k_logs
CHAIN=/tmp/fetchreach80k_logs/chain.log
SLIDE_LOG=/tmp/absp80k_logs/driver.log
echo "[$(date '+%F %H:%M:%S')] FetchReach chain: waiting for FetchSlide 'MASTER DRIVER ALL DONE'..." >> "$CHAIN"
until grep -q 'MASTER DRIVER ALL DONE' "$SLIDE_LOG" 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %H:%M:%S')] FetchSlide complete -> launching FetchReach driver" >> "$CHAIN"
exec bash absp80k_fetchreach.sh
