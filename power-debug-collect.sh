#!/usr/bin/env bash
# power-debug-collect.sh
# SAFE, non-stressing diagnostic collector for the "sudden power-off / reboot" issue.
# Gathers evidence without inducing load (run at idle; will NOT trip the system).
# Run as normal user; some items (DMI, EDAC) need root/privileged and will note that.
set -u
DT=$(date +%Y%m%d-%H%M%S)
OUT="$HOME/power-debug-$DT.txt"
{
echo "### Collect started $(date)"
echo "===== 1. UPTIME / LOAD ====="
uptime
echo; echo "===== 2. PREVIOUS SYSTEM RESET REASON (KEY) ====="
journalctl -b 0 --no-pager 2>/dev/null | grep -iE "reset reason|thermal limit was tripped|sync flood|machine check" | head
echo; echo "===== 3. RECENT KERNEL/MCE HARDWARE ERRORS (this boot & prior) ====="
journalctl --no-pager 2>/dev/null | grep -iE "Hardware Error|MCE:|mce:|data fabric|thermal limit|kernel panic|EDAC|Corrected error" | tail -40
echo; echo "===== 4. ECC ERROR TALLY BY MEMORY CHANNEL (rasdaemon DB) ====="
DB=/var/lib/rasdaemon/ras-mc_event.db
if [ -r "$DB" ]; then
  echo "-- total MC events:"; sqlite3 "$DB" "SELECT count(*) FROM mc_event;" 2>/dev/null
  echo "-- per-channel/slot tally (top rows = prime suspects):"
  sqlite3 "$DB" "SELECT label, count(*) FROM mc_event GROUP BY label ORDER BY count(*) DESC;" 2>/dev/null | head -20
  echo "-- corrected vs uncorrected:"
  sqlite3 "$DB" "SELECT err_type, count(*) FROM mc_event GROUP BY err_type;" 2>/dev/null
else
  echo "(DB not readable; try: sudo sqlite3 $DB ... or run via privileged container)"
fi
echo; echo "===== 5. CPU TEMPS (k10temp Tctl/Tccd) ====="
sensors 2>/dev/null | grep -iE "Tctl|Tccd|temp1"
echo; echo "===== 6. CPU FREQ / GOVERNOR / BOOST (thermal & power caps) ====="
for c in 0 40 111; do echo "cpu$c cur=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null)kHz max=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq 2>/dev/null) gov=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null)"; done
echo "bios_limit=$(cat /sys/devices/system/cpu/cpu0/cpufreq/bios_limit 2>/dev/null) kHz  boost=$(cat /sys/devices/system/cpu/cpu0/cpufreq/boost 2>/dev/null)"
echo; echo "===== 7. CPU PACKAGE POWER (RAPL) ====="
P=$(( $(cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null) )); sleep 4; Q=$(( $(cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null) )); awk -v a=$P -v b=$Q 'BEGIN{printf "package-0 idle power ~%.1f W\n",(b-a)/4/1e6}'
echo; echo "===== 8. GPU STATE (temps, power, clocks, limits) ====="
nvidia-smi --query-gpu=index,name,power.draw,temperature.gpu,utilization.gpu,clocks.sm --format=csv
nvidia-smi -q -d POWER 2>/dev/null | grep -iE "Power Limit|Power Draw"
echo; echo "===== 9. NVMe/CPU temps (secondary) ====="
sensors 2>/dev/null | grep -iE "Composite|Sensor"
echo; echo "===== 10. DOCKER / LOAD (what was running at trip) ====="
echo "-- containers now:"; docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null
echo "-- LLM/AI containers:"; docker ps -a --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -iE "qwen|glm|vllm|llama|sglang|mini" | head
echo; echo "===== 11. MEMORY TOPOLOGY (DMI; needs root/privileged dmidecode) ====="
echo "Known map (from earlier privileged read): MM1=D,MM2=H,MM3=C,MM4=G,MM5=B,MM6=F,MM7=A,MM8=E  (Micron 128GB DDR4-3200 8R ECC x8 = 1TB)"
echo "(to re-read: docker run --rm --privileged ubuntu:24.04 bash -c 'apt-get update -qq && apt-get install -y -qq dmidecode && dmidecode -t memory | grep -iE \"Locator: MM|Bank Locator\"')"
echo; echo "### Collect finished $(date)"
} > "$OUT" 2>&1
echo "Saved to $OUT"
