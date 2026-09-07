#!/bin/bash
# edac-ce-watch.sh
# Pre-trip warning gate: monitors corrected-ECC (CE) counts per DIMM channel and alerts
# when a channel's CE count ramps (the known pre-cursor to an uncorrectable sync-flood on
# this build — channel G / slot MM4 is the prime suspect).
#
# Sources per-channel CE from `ras-mc-ctl --summary` (parsed) and burst detection from the
# aggregate EDAC ce_count. On a spike it logs a prominent ALERT to a persistent file so a
# future trip has a prior-warning trail.
#
# Usage: bash edac-ce-watch.sh             (foreground loop; EDAC_CE_INTERVAL)
#        bash edac-ce-watch.sh --once      (single sample + check)
#        bash edac-ce-watch.sh --status    (print current per-channel CE totals)
# Env: EDAC_CE_INTERVAL=30  EDAC_CE_THRESHOLD=200  EDAC_ANOMALY_NOISE=20
#      EDAC_LOG_DIR (default /var/tmp/powertrip — user-writable; set to /buffer/powertrip if run as root)
set -u

INTERVAL="${EDAC_CE_INTERVAL:-30}"
THRESHOLD="${EDAC_CE_THRESHOLD:-200}"
ANOMALY="${EDAC_ANOMALY_NOISE:-20}"
LOGDIR="${EDAC_LOG_DIR:-/var/tmp/powertrip}"
mkdir -p "$LOGDIR" 2>/dev/null || LOGDIR=/tmp/powertrip
ALERT="$LOGDIR/edac-ce-alerts.log"
LAST="$LOGDIR/edac-ce-last"
mkdir -p "$LOGDIR"

log(){ echo "$*"; }
log_alert(){ echo "[$(date -Is)] ALERT: $*" | tee -a "$ALERT"; }

# Per-channel CE totals. Try ras-mc-ctl --summary lines like:
#   Corrected on DIMM Label(s): 'mc#0csrow#0channel#6' location: 0:0:6:-1 errors: 1367
# Falls back to aggregate EDAC ce_count if not available.
get_ce_totals(){
  if command -v ras-mc-ctl >/dev/null 2>&1; then
    ras-mc-ctl --summary 2>/dev/null | grep -a "Corrected on DIMM Label" \
      | sed -E "s/.*channel#([0-9]+)'.*errors: ([0-9]+).*/channel#\1 \2/"
  else
    echo "mc0_ce $(cat /sys/devices/system/edac/mc/mc0/ce_count 2>/dev/null || echo 0)"
  fi
}

sample(){
  local totals; totals=$(get_ce_totals)
  local cur_ce=0 alert_count=0
  if [ -z "$totals" ]; then
    # aggregate fallback
    cur_ce=$(cat /sys/devices/system/edac/mc/mc0/ce_count 2>/dev/null || echo 0)
    log "no per-channel report; aggregate mc0/ce_count=$cur_ce"
  else
    while read -r ch ce; do
      [ -n "$ch" ] || continue
      [ "$ce" -ge "$THRESHOLD" ] || continue
      log_alert "$ch has $ce corrected-ECC errors (>= ${THRESHOLD}) — likely faulty DIMM; channel G = slot MM4."
      alert_count=$((alert_count+1))
    done <<< "$totals"
    cur_ce=0
    while read -r _ch c; do [ -n "$_ch" ] && cur_ce=$((cur_ce + c)); done <<< "$totals"
  fi
  # burst detection
  local prev_ce=0
  [ -f "$LAST" ] && prev_ce=$(cat "$LAST" 2>/dev/null || echo 0)
  local gain=$((cur_ce - prev_ce))
  if [ "$gain" -ge "$ANOMALY" ]; then
    log_alert "Corrected-ECC burst: +$gain CE in ${INTERVAL}s (total $cur_ce) — CE ramping toward potential uncorrectable fault."
    alert_count=$((alert_count+1))
  fi
  echo "$cur_ce" > "$LAST" 2>/dev/null || true
  [ "$alert_count" -gt 0 ]
}

case "${1:-}" in
  --once) sample; echo "exit=$? (0=alert, 1=ok)"; ;;
  --status) echo "Per-channel corrected-ECC:"; get_ce_totals ;;
  *) log "EDAC CE watch armed: interval=${INTERVAL}s threshold=${THRESHOLD} burst_gain>=${ANOMALY}. Alerts: $ALERT"
     while true; do sample >/dev/null 2>&1; sleep "$INTERVAL"; done ;;
esac
