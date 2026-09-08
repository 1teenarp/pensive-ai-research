#!/bin/bash
# reset-reason-decoder.sh
# Decode AMD "Previous system reset reason" 0xNNNNNNNN codes from the kernel's
# s5_reset_reason_txt table (arch/x86/kernel/cpu/amd.c). This tells you which RESET
# MECHANISM fired — important because only bit 27 (data-fabric sync flood) is direct
# evidence of an uncorrectable memory error.
#
# Usage: reset-reason-decoder.sh 0x00080a00
#        reset-reason-decoder.sh            # dump the full known table
#        reset-reason-decoder.sh --history  # decode the last resets seen in the logs
set -u

# bit number -> reason string (from arch/x86/kernel/cpu/amd.c s5_reset_reason_txt[])
declare -A R=(
  [0]="thermal pin BP_THERMTRIP_L was tripped"                         # Pin
  [1]="power button was pressed for 4 seconds"                          # Pin
  [2]="shutdown pin was tripped"                                        # Pin
  [4]="remote ASF power off command was received"                       # Remote
  [9]="internal CPU thermal limit was tripped"                          # Internal
  [16]="system reset pin BP_SYS_RST_L was tripped"                      # Pin
  [17]="software issued PCI reset"                                      # Software
  [18]="software wrote 0x4 to reset control register 0xCF9"             # Software
  [19]="software wrote 0x6 to reset control register 0xCF9"             # Software
  [20]="software wrote 0xE to reset control register 0xCF9"             # Software
  [21]="ACPI power state transition occurred"                           # ACPI-state
  [22]="keyboard reset pin KB_RST_L was tripped"                        # Pin
  [23]="internal CPU shutdown event occurred"                           # Internal
  [24]="system failed to boot before failed boot timer expired"         # Hardware
  [25]="hardware watchdog timer expired"                                # Hardware
  [26]="remote ASF reset command was received"                          # Remote
  [27]="an uncorrected error caused a data fabric sync flood event"     # Internal
  [29]="FCH and MP1 failed warm reset handshake"                        # Internal
  [30]="a parity error occurred"                                        # Internal
  [31]="a software sync flood event occurred"                           # Internal
)
# type labels for the table dump
declare -A T=(
  [0]="Pin" [1]="Pin" [2]="Pin" [4]="Remote" [9]="Internal" [16]="Pin"
  [17]="Software" [18]="Software" [19]="Software" [20]="Software" [21]="ACPI-state"
  [22]="Pin" [23]="Internal" [24]="Hardware" [25]="Hardware" [26]="Remote"
  [27]="Internal" [29]="Internal" [30]="Internal" [31]="Internal"
)

dump_table(){
  echo "Known reset-reason bits (s5_reset_reason_txt):"
  for i in $(seq 0 31); do
    if [ -n "${R[$i]:-}" ]; then
      printf "  bit %-2s 0x%08x  %-12s %s\n" "$i" $((1<<i)) "${T[$i]}" "${R[$i]}"
    fi
  done
  echo
  echo "Class guide:"
  echo "  bit 27 (0x08000000) = data-fabric sync flood from an UNCORRECTED ERROR  -> memory fault"
  echo "  bit 19 (0x00080000) = software wrote 0x6 to 0xCF9 -> SOFTWARE-initiated warm reset"
  echo "  bit 21 (0x00200000) = ACPI power-state transition -> power/firmware reset"
  echo "  bit  9 (0x00000200) = CPU internal thermal limit -> appears alongside many (often latched)"
}

history(){
  grep -aH "Previous system reset" /var/log/syslog /var/log/kern.log 2>/dev/null \
    | sed -E 's/.*\[(0x[0-9a-f]+)\]: (.*)/\1|\2/' | tail -20 | while IFS='|' read -r code txt; do
      [ -n "${code:-}" ] || continue
      echo "=============== $code ==============="
      decode "$code"
    done
}

decode(){
  local v="$1" i
  v=${v#0x}; v=$(printf '%d' "0x$v" 2>/dev/null) || { echo "invalid code: $1"; return 1; }
  echo "0x$(printf '%08x' "$v"):"
  local any=0
  for i in $(seq 0 31); do
    if [ $(( v & (1<<i) )) -ne 0 ]; then
      if [ -n "${R[$i]:-}" ]; then
        printf "  bit %-2s 0x%08x  %-12s %s\n" "$i" $((1<<i)) "${T[$i]}" "${R[$i]}"
      else
        printf "  bit %-2s 0x%08x  (unknown/unmapped bit)\n" "$i" $((1<<i))
      fi
      any=1
    fi
  done
  [ "$any" -eq 0 ] && echo "  (no reason bits set — no reset reason recorded)"
  echo
  if [ $(( v & (1<<27) )) -ne 0 ]; then echo "  -> memory-fault evidence: YES (bit27 = data-fabric sync flood)"; else echo "  -> memory-fault evidence: no"; fi
  if [ $(( v & (1<<19) )) -ne 0 ]; then echo "  -> software-reset evidence: YES (bit19 = 0xCF9 write)"; else echo "  -> software-reset evidence: no"; fi
  if [ $(( v & (1<<21) )) -ne 0 ]; then echo "  -> ACPI/power evidence: YES (bit21)"; else echo "  -> ACPI/power evidence: no"; fi
}

case "${1:-}" in
  "") dump_table; exit 0 ;;
  --history|-h) history; exit 0 ;;
  0x*) decode "$1"; exit 0 ;;
  *) echo "usage: $0 [0xXXXX | --history | (no arg = table)]"; exit 1 ;;
esac
