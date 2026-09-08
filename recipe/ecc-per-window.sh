#!/bin/bash
# ecc-per-window.sh
# Report Corrected/Uncorrectable ECC events per DIMM channel, over a SPECIFIC time window,
# using the rasdaemon database (which timestamps every event).
#
# WHY THIS EXISTS:
#   `ras-mc-ctl --summary` returns CUMULATIVE counts since the rasdaemon DB was created
#   (lifetime). Those big numbers are historic accumulation and are NOT evidence that a
#   given trip was caused by a DIMM. To attribute CEs to a specific boot/trip you must
#   bucket the timestamped events by window. This tool does that.
#
# Usage:
#   ecc-per-window.sh --day 2026-09-07          # per-channel CE for one UTC/local day
#   ecc-per-window.sh --since "2026-09-07 16:00" # per-channel CE since a datetime
#   ecc-per-window.sh --boot -1                 # per-channel CE in journalctl boot -1's window
#   ecc-per-window.sh --all                     # lifetime (== ras-mc-ctl --summary)
#   ecc-per-window.sh --last-hours 6
#   ecc-per-window.sh --help
#
# Output: per-channel CE (and UE if any) counts + a slot (MMn) mapping using the
#         EDAC channel# -> channel letter -> DIMM slot map for this board.
#
# Note: `mc_event` holds one row per corrected event (err_type='Corrected'); uncorrected /
#       non-corrected variants appear in other err_types.
set -u

# --- known board mapping: EDAC channel# -> channel letter -> DIMM slot ---
CH_LETTER=(A B C D E F G H)                                   # channel idx -> letter
SLOT_BY_CH=(MM7 MM5 MM3 MM1 MM8 MM6 MM4 MM2)                 # letter idx -> slot

SINCE=""
until_dt=""
DAY=""
LAST_HOURS=0
BOOT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --since) SINCE="$2"; shift 2;;
    --until) until_dt="$2"; shift 2;;
    --day) DAY="$2"; shift 2;;
    --boot) BOOT="$2"; shift 2;;
    --last-hours) LAST_HOURS="$2"; shift 2;;
    --all) SINCE="1970-01-01"; shift;;
    --help|-h) sed -n '1,30p' "$0"; exit 0;;
    *) echo "unknown arg $1" >&2; exit 1;;
  esac
done

DB="/var/lib/rasdaemon/ras-mc_event.db"
DBGLOB="/var/lib/rasdaemon/ras-mc*.db"
[ -r "$DB" ] || DB=$(ls $DBGLOB 2>/dev/null | head -1)

# resolve boot window. NOTE: rasdaemon DB timestamps mix UTC (+0000, older) and local
# (-0700, newer) offsets, and journalctl uses local time, so exact datetime matching is
# unreliable across TZ. Use DAY granularity: filter the DB by the boot's local start date(s).
if [ -n "$BOOT" ]; then
  bstart=$(journalctl -b "$BOOT" --no-pager -o short-iso 2>/dev/null | head -1 | awk '{print $1}')
  bend=$(journalctl -b "$BOOT" --no-pager -o short-iso 2>/dev/null | tail -1 | awk '{print $1}')
  bstart=${bstart%%T*}; bend=${bend%%T*}
  if [ -n "$bstart" ]; then
    DAY="$bstart"
    if [ "$bstart" != "$bend" ]; then
      echo "NOTE: boot spans $bstart..$bend; --day uses start date $bstart. Use --since/--until for full range." >&2
    fi
  fi
fi

# build SQL WHERE
where="1=1"
if [ -n "$SINCE" ]; then
  where="$where AND timestamp >= '$SINCE'"
fi
if [ -n "$until_dt" ]; then
  where="$where AND timestamp <= '$until_dt'"
fi
if [ -n "$DAY" ]; then
  where="$where AND substr(timestamp,1,10) = '$DAY'"
fi
if [ "$LAST_HOURS" -gt 0 ]; then
  where="$where AND timestamp >= datetime('now','-$LAST_HOURS hours')"
fi

echo "======================================================================"
echo " ECC by DIMM channel — window-based accounting"
echo " host: $(hostname)   now: $(date)"
echo " window: since='${SINCE:-ALL}' until='${until_dt:-now}' day='${DAY:-}' last_hours=$LAST_HOURS"
echo "======================================================================"

if [ -z "$DB" ] || [ ! -r "$DB" ]; then
  echo "ERROR: cannot read rasdaemon DB ($DB). Run as root or copy to readable path." >&2
  echo "  Fallback: ras-mc-ctl --summary is CUMULATIVE (lifetime), not window-based." >&2
  exit 1
fi

cp "$DB" /tmp/.ecc-window-copy.db 2>/dev/null || { echo "cannot copy DB"; exit 1; }
chmod 644 /tmp/.ecc-window-copy.db 2>/dev/null

echo
echo "[Channel CE counts in window]"
sqlite3 /tmp/.ecc-window-copy.db \
  "SELECT middle_layer, count(*) FROM mc_event WHERE err_type='Corrected' AND ($where) GROUP BY middle_layer ORDER BY count(*) DESC;" 2>&1 \
  | while IFS='|' read -r ch cnt; do
      [ -n "$ch" ] || continue
      letter="${CH_LETTER[$ch]:-?}"
      slot="${SLOT_BY_CH[$(( $(printf '%d' "$ch") ))]:-?}"
      printf "  channel#%-2s (channel %s / %s): %s CE\n" "$ch" "$letter" "$slot" "$cnt"
    done

echo
echo "[Non-corrected / uncorrectable events in window]"
sqlite3 /tmp/.ecc-window-copy.db \
  "SELECT err_type, middle_layer, count(*) FROM mc_event WHERE err_type != 'Corrected' AND ($where) GROUP BY err_type, middle_layer ORDER BY count(*) DESC;" 2>&1

echo
echo "[Total CE events in window: $(sqlite3 /tmp/.ecc-window-copy.db "SELECT count(*) FROM mc_event WHERE err_type='Corrected' AND ($where);" 2>/dev/null)]"

rm -f /tmp/.ecc-window-copy.db
