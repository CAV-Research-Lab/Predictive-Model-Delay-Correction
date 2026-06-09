#!/usr/bin/env bash
# Fires (exits) when a new run completes, when all done, or if the master driver dies.
# Re-armed after each report with the new DONE count as $1. Lives in repo to survive reboot.
BASE=${1:-0}
LOG=/tmp/absp80k_logs/driver.log
while true; do
  done=$(grep -c '\] DONE  ' "$LOG" 2>/dev/null || echo 0)
  grep -q 'MASTER DRIVER ALL DONE' "$LOG" 2>/dev/null && { echo "EVENT=ALLDONE done=$done"; break; }
  [ "${done:-0}" -gt "$BASE" ] && { echo "EVENT=NEWDONE done=$done"; break; }
  pgrep -f absp80k_master.sh >/dev/null 2>&1 || { echo "EVENT=DRIVER_GONE done=$done"; break; }
  sleep 180
done
echo "=== driver.log ==="; cat "$LOG"
