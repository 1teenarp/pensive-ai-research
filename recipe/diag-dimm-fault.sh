#!/bin/bash
# diag-dimm-fault.sh
# Identify the likely-faulty DIMM (slot) from this host's Corrected/Uncorrectable ECC counters.
# Output is formatted to attach to a vendor warranty/RMA claim for a RAM replacement.
#
# Sources:
#   - rasdaemon / ras-mc-ctl (memory controller events, per-channel CE/UE)
#   - EDAC sysfs (/sys/devices/system/edac/mc/mc0/...)
#   - dmidecode (DIMM slot <-> channel <-> rank mapping)
#
# Usage: bash diag-dimm-fault.sh           (normal; runs live ras-mc-ctl/sysfs)
#        bash diag-dimm-fault.sh --full     (also tries dmidecode slot map via privileged docker)
#        bash diag-dimm-fault.sh > report.txt
set -u

echo "======================================================================"
echo " DIMM FAULT DIAGNOSTIC — $(hostname), $(date)"
echo "======================================================================"
echo
echo "NOTE: rasdaemon counters are CUMULATIVE lifetime totals (since the DB was created)."
echo "      A high lifetime count on one channel indicates long-run DIMM stress, but is NOT"
echo "      proof that a specific recent trip was caused by that DIMM. To attribute CEs to a"
echo "      boot/trip, cross-reference the per-day/per-boot bucket in [6] (ecc-per-window.sh)."
echo

echo; echo "[1] Memory controller ECC summary (ras-mc-ctl — CUMULATIVE lifetime)"
echo "----------------------------------------------------------------------"
if command -v ras-mc-ctl >/dev/null 2>&1; then
  ras-mc-ctl --summary 2>/dev/null || echo "ras-mc-ctl --summary failed"
else
  echo "ras-mc-ctl not installed"
fi

echo; echo "[2] Live per-channel Corrected/Uncorrectable counters (EDAC sysfs — THIS BOOT only)"
echo "----------------------------------------------------------------------"
# Summarize by csrow/channel. Many EPYC setups report channels 0-7 per csrow.
# NOTE: these sysfs counters reset to 0 at each boot, so they reflect only the CURRENT boot.
totCE=0; totUE=0
declare -A CE UE
for chf in /sys/devices/system/edac/mc/mc0/csrow*/channel*/ce_count; do
  [ -r "$chf" ] || continue
  ch=$(basename "$(dirname "$chf")")
  ce=$(cat "$chf"); totCE=$((totCE+ce)); CE[$ch]=$(( ${CE[$ch]:-0} + ce ))
done
for chf in /sys/devices/system/edac/mc/mc0/csrow*/channel*/ue_count; do
  [ -r "$chf" ] || continue
  ch=$(basename "$(dirname "$chf")")
  ue=$(cat "$chf"); totUE=$((totUE+ue)); UE[$ch]=$(( ${UE[$ch]:-0} + ue ))
done
echo "Total CE (this boot): $totCE   Total UE: $totUE"
echo "Per-channel (highest first):"
if [ "${#CE[@]}" -gt 0 ]; then
  for ch in "${!CE[@]}"; do printf "  %-12s CE=%s  UE=%s\n" "$ch" "${CE[$ch]}" "${UE[$ch]:-0}"; done | sort -t= -k2 -rn
else
  echo "  (no per-channel ce_count found in sysfs; see [6]/[1] for rasdaemon counts)"
fi
[ -r /sys/devices/system/edac/mc/mc0/ce_noinfo_count ] && echo "CE-noinfo: $(cat /sys/devices/system/edac/mc/mc0/ce_noinfo_count)"
[ -r /sys/devices/system/edac/mc/mc0/ue_noinfo_count ] && echo "UE-noinfo: $(cat /sys/devices/system/edac/mc/mc0/ue_noinfo_count)"
echo "(sysfs may read 0 if this boot had no ECC activity or the driver exposes per-rank nodes)"

echo; echo "[3] DIMM slot <-> channel <-> rank map (dmidecode)"
echo "----------------------------------------------------------------------"
SLOT_CH="MM1:D MM2:H MM3:C MM4:G MM5:B MM6:F MM7:A MM8:E"   # from power-trip-diagnosis.md
if command -v dmidecode >/dev/null 2>&1; then
  dmidecode -t memory 2>/dev/null | grep -iE "Locator: MM|Bank Locator:|Size: [0-9]|Speed: [0-9]|Rank:" \
    || echo "(dmidecode needs root; run with sudo/priv docker)"
else
  echo "(dmidecode not installed; showing known slot->channel map)"
fi
echo "Known slot->channel map (MM<slot>=<channel letter>):"
echo "  $SLOT_CH"

echo; echo "[4] Recommended slot(s) to inspect / get replaced"
echo "----------------------------------------------------------------------"
# Map EDAC channel index -> DMI channel letter -> DIMM slot.
# On this board: EDAC channel#N == 'Channel <letter>' (A=0,B=1,...,H=7) and the
# slot map (from power-trip-diagnosis.md / dmidecode) is:
#   A=MM7 B=MM5 C=MM3 D=MM1 E=MM8 F=MM6 G=MM4 H=MM2
# So EDAC channel#6 = Channel G = slot MM4 ; channel#7 = Channel H = slot MM2.
MAP=(A B C D E F G H)           # EDAC channel idx -> channel letter
SLOT_BY_CH=(MM7 MM5 MM3 MM1 MM8 MM6 MM4 MM2)   # channel letter idx -> DIMM slot
echo "EDAC channel -> channel letter -> DIMM slot:"
for i in "${!MAP[@]}"; do printf "  channel#%s = Channel %s = slot %s\n" "$i" "${MAP[$i]}" "${SLOT_BY_CH[$i]}"; done
echo
echo "Cross-reference the CE-heavy channels found in [1]/[6] with the table above."
echo "The channel(s) with the highest corrected-ECC count are the likely faulty DIMM slot(s)."
echo "Action: reseat or replace that DIMM, then run memtest86 targeting it."
echo "For a warranty/RMA: attach [1] (lifetime), [6] (per-boot window) + the reset-reason history in [5]."
echo "IMPORTANT: a large lifetime count in [1] is long-run evidence; only [6] ties CEs to a trip."

echo; echo "[5] Reset-reason history + decode (uncorrectable escalation evidence)"
echo "----------------------------------------------------------------------"
grep -a "Previous system reset" /var/log/syslog /var/log/kern.log 2>/dev/null | tail -6
echo
DECODER="$(dirname "$0")/reset-reason-decoder.sh"
if [ -x "$DECODER" ]; then
  echo "Reset-reason decode of the latest codes (from recipe/reset-reason-decoder.sh):"
  grep -a "Previous system reset" /var/log/syslog /var/log/kern.log 2>/dev/null \
    | grep -oaE "0x[0-9a-f]{8}" | sort -u | tail -4 | while read -r code; do "$DECODER" "$code" 2>/dev/null; done
else
  echo "(reset-reason-decoder.sh not found next to this script)"
fi
echo
echo "Decode quick-guide (from arch/x86/kernel/cpu/amd.c s5_reset_reason_txt):"
echo "  bit27 = data-fabric sync flood (uncorrected error) -> 0x08000a00 = MEMORY fault"
echo "  bit19 = 'software wrote 0x6 to reset control reg 0xCF9' -> 0x00080a00 = SOFTWARE-initiated warm reset"
echo "  bit21 = ACPI power-state transition -> 0x00200a00/0x00200800 = power/firmware reset"
echo
echo "NOTE: only reset reason 0x08000a00 (bit27 = data-fabric sync flood from an uncorrected"
echo "      error) is direct evidence of a memory fault during that reset."

echo; echo "[6] Per-boot / per-day ECC attribution (rasdaemon DB, window-based)"
echo "----------------------------------------------------------------------"
SCRIPT="$(dirname "$0")/ecc-per-window.sh"
if [ -x "$SCRIPT" ]; then
  echo "Lifetime (all channels):"
  "$SCRIPT" --all 2>&1 | grep -A20 "Channel CE counts"
  echo
  echo "Last 3 boots (per-boot CE):"
  for b in -1 -2 -3; do
    echo "--- boot $b ---"
    "$SCRIPT" --boot "$b" 2>&1 | grep -A20 "Channel CE counts"
  done
else
  echo "ecc-per-window.sh not found next to this script; skipping windowed accounting."
fi
echo "======================================================================"
