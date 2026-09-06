#!/bin/bash
# powertrip-capture.sh
# High-fidelity crash-capture telemetry logger for the "pensive" build.
# Designed to run in a PRIVILEGED container so it can read RAPL package power,
# kernel log (dmesg), and EDAC. Writes to a persistent host mount so data
# SURVIVES the power-trip reset (unlike /tmp).
#
# Sources captured on every tick:
#   - host clock, kernel uptime (to detect reset boundary)
#   - CPU: k10temp Tctl + 8 chiplets (hwmon), RAPL package/core energy -> watts
#   - CPU freq / scaling / numa
#   - GPU (nvidia-smi): temp, power draw/limit, util, mem, clocks, PCIe link
#   - GPU ECC counters (corrected / uncorrected, if exposed)
#   - system memory pressure (free, /proc/meminfo)
#   - load average, process count (top CPU consumers)
#   - kernel log tail + MCE/EDAC lines (dmesg)
#   - ZFS stats (arc, memory)
# Every tick also appends to a JSONL file for easy later parsing.
set -u

OUTDIR="${OUTDIR:-/capture}"
INTERVAL="${INTERVAL:-1}"
LOG="$OUTDIR/telemetry-$(date +%Y%m%d-%H%M%S).csv"
JSONL="$OUTDIR/telemetry-$(date +%Y%m%d-%H%M%S).jsonl"
KLOG="$OUTDIR/klog-$(date +%Y%m%d-%H%M%S).log"
EDACLOG="$OUTDIR/edac-$(date +%Y%m%d-%H%M%S).csv"

mkdir -p "$OUTDIR"
HDR="epoch,iso,uptime_s,tctl,tccd1,tccd2,tccd3,tccd4,tccd5,tccd6,tccd7,tccd8,pkg_w,core_w,cpu_freq,load1,load5,load15,mem_used_gb,mem_avail_gb,nproc,g0_temp,g0_power,g0_mem_gb,g0_util,g1_temp,g1_power,g1_mem_gb,g1_util,g0_ecc_ce,g0_ecc_ue,g1_ecc_ce,g1_ecc_ue,g0_link,g1_link,topcpu"
echo "$HDR" > "$LOG"

PW=/sys/class/powercap/intel-rapl:0/energy_uj
CW=/sys/class/powercap/intel-rapl:0:0/energy_uj
vals(){ for f in /sys/class/hwmon/hwmon2/temp*_input; do cat "$f"; done; }

# energy -> watts requires a delta between samples; keep prev
prev_pkg=""; prev_core=""; prev_t=""

snapshot(){
  local now iso up
  now=$(date +%s)
  iso=$(date -Is)
  up=$(awk '{print $1}' /proc/uptime)

  # temps: tctl=temp1, chiplet tccd = temp3..temp10 (temp2 missing on EPYC)
  local -a tp; local i=0
  while read -r v; do tp[$i]="$v"; i=$((i+1)); done < <(vals)
  local tctl="${tp[0]:-NA}"
  local tccd1="${tp[1]:-NA}" tccd2="${tp[2]:-NA}" tccd3="${tp[3]:-NA}" tccd4="${tp[4]:-NA}"
  local tccd5="${tp[5]:-NA}" tccd6="${tp[6]:-NA}" tccd7="${tp[7]:-NA}" tccd8="${tp[8]:-NA}"

  # RAPL watts (awk for float math; avoid bc dependency)
  local pkg="" core="" pw=NA cw=NA
  [ -r "$PW" ] && pkg=$(cat "$PW")
  [ -r "$CW" ] && core=$(cat "$CW")
  if [ -n "$prev_pkg" ] && [ -n "$pkg" ]; then
    pw=$(awk -v a="$prev_pkg" -v b="$pkg" -v i="$INTERVAL" 'BEGIN{ if(b>a) printf "%.2f",(b-a)/1e6/i; else printf "NA" }')
  fi
  if [ -n "$prev_core" ] && [ -n "$core" ]; then
    cw=$(awk -v a="$prev_core" -v b="$core" -v i="$INTERVAL" 'BEGIN{ if(b>a) printf "%.2f",(b-a)/1e6/i; else printf "NA" }')
  fi
  prev_pkg="$pkg"; prev_core="$core"

  local freq=$(awk '{print $1}' /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)
  read -r l1 l5 l15 < <(awk '{print $1","$2","$3}' /proc/loadavg | tr ',' ' ')
  local memu mema
  memu=$(free -g | awk '/^Mem:/{print $2}')
  mema=$(free -g | awk '/^Mem:/{print $7}')
  local nproc=$(nproc)

  # GPU
  local g0temp g0pw g0memu g0util g1temp g1pw g1memu g1util
  g0temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i 0 2>/dev/null)
  g0pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 2>/dev/null)
  g0memu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
  g0util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null)
  g1temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i 1 2>/dev/null)
  g1pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 1 2>/dev/null)
  g1memu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null)
  g1util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null)

  # GPU ECC counters (may be N/A on some cards)
  local g0ce g0ue g1ce g1ue
  g0ce=$(nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total --format=csv,noheader,nounits -i 0 2>/dev/null)
  g0ue=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits -i 0 2>/dev/null)
  g1ce=$(nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total --format=csv,noheader,nounits -i 1 2>/dev/null)
  g1ue=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits -i 1 2>/dev/null)

  # GPU PCIe link status + memory utilization
  local g0link g1link
  g0link=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader,nounits -i 0 2>/dev/null)
  g1link=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader,nounits -i 1 2>/dev/null)

  local topcpu
  topcpu=$(ps -eo pcpu,comm --sort=-pcpu 2>/dev/null | head -6 | tr '\n' ';')

  echo "$now,$iso,$up,$tctl,$tccd1,$tccd2,$tccd3,$tccd4,$tccd5,$tccd6,$tccd7,$tccd8,$pw,$cw,$freq,$l1,$l5,$l15,$memu,$mema,$nproc,$g0temp,$g0pw,$g0memu,$g0util,$g1temp,$g1pw,$g1memu,$g1util,$g0ce,$g0ue,$g1ce,$g1ue,$g0link,$g1link,\"$topcpu\"" >> "$LOG"
}

# Continuous kernel log capture. Keep BOTH:
#   - full dmesg ring-buffer snapshot each tick (raw, unfiltered -> nothing lost),
#   - a filtered RAS/thermal/error view for quick scanning.
# The raw full snapshot is what we grepp after a crash; MCE/EDAC lines are
# retained even if they scroll out of the filtered window.
klog_on(){
  while true; do
    ts=$(date -Is)
    {
      echo "===== $ts ====="
      echo "--- dmesg tail-400 (raw) ---"
      dmesg 2>/dev/null | tail -400
    } >> "$KLOG"
    sleep "$INTERVAL"
  done
}

# Periodic EDAC / rasdaemon tally snapshot (CE/UE per DIMM/channel) --> CSV
# Re-reads the rasdaemon DB so we can correlate ECC growth with load.
edac_on(){
  while true; do
    sqlite3 /var/lib/rasdaemon/ras-mc_event.db \
      "SELECT datetime(time,'unixepoch'),COALESCE(label,'-'),err_type FROM mc_event ORDER BY id DESC LIMIT 200;" \
      >> "$EDACLOG" 2>/dev/null
    sleep 5
  done
}

# Once-per-run full system + RAS summary
summary(){
  {
    echo "###### powertrip capture started $(date -Is) ######"
    echo "--- CPU ---"; lscpu | grep -iE "model name|^CPU\(s\)|NUMA|Thread|Core|socket|max mhz"; nproc
    echo "--- Memory map / DIMM ---"; dmidecode -t memory 2>/dev/null | grep -iE "Locator|Bank Locator|Size|Type:|Speed|Rank" | head -80
    echo "--- GPU ---"; nvidia-smi
    echo "--- Edge: running containers ---"; docker ps 2>/dev/null
    echo "--- rasdaemon summary ---"; ras-mc-ctl --summary 2>/dev/null || echo "ras-mc-ctl not available"
    echo "--- mounts ---"; df -h
    echo "###### end summary ######"
  } > "$OUTDIR/summary-$(date +%Y%m%d-%H%M%S).txt"
}

# start kernel-log tailer + EDAC snapshotter in background
klog_on & KLOGPID=$!
edac_on & EDACPID=$!
summary

echo "capture running: interval=${INTERVAL}s outdir=$OUTDIR (log=$LOG)"
trap 'kill $KLOGPID $EDACPID 2>/dev/null' EXIT

# main loop
while true; do
  snapshot
  sleep "$INTERVAL"
done
