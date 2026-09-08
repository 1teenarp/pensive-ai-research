#!/bin/bash
# edac-ce-watch.sh
# Pre-trip warning gate: monitors corrected-ECC (CE) counts per DIMM channel and alerts
# when a channel's CE count RAMPS (new CEs appearing) — the known pre-cursor to an
# uncorrectable sync-flood on this build.
#
# IMPORTANT (corrected 2026-09-07): `ras-mc-ctl --summary` returns CUMULATIVE lifetime counts,
# which inflate every boot and produce false, perpetual alerts (e.g. "channel#6 has 1367").
# This script therefore watches the WINDOW DELTA of CE events from the rasdaemon DB
# (timestamped), so alerts mean "N new corrected-ECC errors appeared in the last interval",
# not "this DIMM accumulated N ever". See recipe/ecc-per-window.sh.
#
# Sources per-channel CE from the rasdaemon DB (mc_event, middle_layer == channel).
# On a spike it logs a prominent ALERT to a persistent file so a future trip has a
# prior-warning trail.
#
# NOTE on the first run: with no prior cursor file, ALL historical CEs are counted as "new"
# once (baseline). Delete $LAST to reset the baseline; subsequent runs only alert on CEs
# that appear after the last run.
#
# Usage: bash edac-ce-watch.sh             (foreground loop; EDAC_CE_INTERVAL)
#        bash edac-ce-watch.sh --once      (single sample + check)
#        bash edac-ce-watch.sh --status    (print current per-channel CE window totals)
# Env: EDAC_CE_INTERVAL=30  EDAC_CE_THRESHOLD=20  EDAC_CE_DELTA=5
#      EDAC_LOG_DIR (default /var/tmp/powertrip — user-writable; set to /buffer/powertrip if run as root)
set -u

INTERVAL="${EDAC_CE_INTERVAL:-30}"
# Per-window alert threshold: N new CE on one channel in one interval.
THRESHOLD="${EDAC_CE_THRESHOLD:-20}"
# Burst delta considered anomalous (new CE in one interval).
DELTA="${EDAC_CE_DELTA:-5}"
LOGDIR="${EDAC_LOG_DIR:-/var/tmp/powertrip}"
mkdir -p "$LOGDIR" 2>/dev/null || LOGDIR=/tmp/powertrip
ALERT="$LOGDIR/edac-ce-alerts.log"
LAST="$LOGDIR/edac-ce-last"
mkdir -p "$LOGDIR"

DB=$(ls /var/lib/rasdaemon/ras-mc*.db 2>/dev/null | head -1)

log(){ echo "$*"; }
log_alert(){ echo "[$(date -Is)] ALERT: $*" | tee -a "$ALERT"; }

# Current per-channel CE counts from the rasdaemon DB (window of rows since prev snapshot
# by id). Returns channel lines followed by a final line = current MAX(id) cursor.
# Helper: current total per-channel CE (for --status, all history).
get_ce_totals(){
  local tmp; tmp=$(mktemp /tmp/.edac-ttl.XXXXXX) || return 1
  [ -r "$DB" ] || return 1
  cp "$DB" "$tmp" 2>/dev/null || return 1
  chmod 644 "$tmp" 2>/dev/null
  sqlite3 "$tmp" "SELECT 'channel#'||middle_layer, count(*) FROM mc_event WHERE err_type='Corrected' GROUP BY middle_layer;" 2>/dev/null
  rm -f "$tmp"
}
get_ce_delta(){
  local prev_file="$1"
  local tmp; tmp=$(mktemp /tmp/.edac-delta.XXXXXX) || return 1
  [ -r "$DB" ] || return 1
  cp "$DB" "$tmp" 2>/dev/null || return 1
  chmod 644 "$tmp" 2>/dev/null
  # New events since previous snapshot's highest id => count per channel of rows with id > prev_max_id.
  local prev_max_id=0
  [ -r "$prev_file" ] && prev_max_id=$(cat "$prev_file" 2>/dev/null || echo 0)
  sqlite3 "$tmp" "SELECT middle_layer, count(*) FROM mc_event WHERE err_type='Corrected' AND id > ${prev_max_id:-0} GROUP BY middle_layer;" 2>/dev/null
  local max_id; max_id=$(sqlite3 "$tmp" "SELECT COALESCE(MAX(id),0) FROM mc_event;" 2>/dev/null)
  rm -f "$tmp"
  echo "$max_id"
}

# Account: emit total per-channel (window from MAX baseline handled by caller). Placeholder aggregates.
sample(){
  local totals cue alert_count=0
  # capture new-event counts per channel since last sample, then remember new max id
  local tmpfile; tmpfile=$(mktemp /tmp/.edac-snap.XXXXXX)
  totals=$(get_ce_delta "$LAST")
  local maxid; maxid=$(echo "$totals" | tail -1)
  # The channel lines are everything except the last line (which is the max id).
  local ct; ct=$(echo "$totals" | sed '$d')
  echo "$maxid" > "$LAST" 2>/dev/null || true

  while IFS='|' read -r ch ce; do
    [ -n "$ch" ] || continue
    [ "${ce:-0}" -ge "$DELTA" ] || continue
    ce=${ce:-0}
    log_alert "channel#${ch} got $ce new corrected-ECC errors in ${INTERVAL}s (delta>=${DELTA}) — CE ramping toward potential uncorrectable fault; channel G = slot MM4."
    alert_count=$((alert_count+1))
    [ "$ce" -ge "$THRESHOLD" ] && log_alert "channel#${ch} has $ce CE in window (>= ${THRESHOLD}) — accelerated; likely-faulty DIMM; channel G = slot MM4."
  done <<< "$ct"
  rm -f "$tmpfile"
  [ "$alert_count" -gt 0 ]
}

case "${1:-}" in
  --once) sample; echo "exit=$? (0=alert, 1=ok)"; ;;
  --status) echo "Per-channel corrected-ECC (this window, since last snapshot):"; get_ce_delta "$LAST" | sed '$d' ;;
  *) echo "EDAC CE watch armed: interval=${INTERVAL}s threshold=${THRESHOLD} delta>=${DELTA}. Alerts: $ALERT"
     while true; do sample >/dev/null 2>&1; sleep "$INTERVAL"; done ;;
esac
