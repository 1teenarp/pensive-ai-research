# Power-Trip Live Capture (crash-capture telemetry)

Date: 2026-09-05
Purpose: on-box, high-fidelity **live telemetry capture** that runs *during* a heavy workload and
writes directly to persistent disk, so that when the recurring power-trip (sync-flood / thermal reset)
hits we retain a per-second trace of temps / power / GPU / memory / kernel log / ECC leading up to the
reset — something the previous incidents lacked.

## Why this exists
Instance 2 of the failure catalog (`power-trip-instances.md`) died during a vLLM model load, but at
that time **no persistent temp/power telemetry was running** (TODO B2) — the crash window had no
timeline. The metrics-stack's Prometheus TSDB was not persisting and `/tmp` logs were wiped on reboot.
This folder adds always-on, crash-safe capture.

## Components (in this repo)
| File | Role |
|---|---|
| `powertrip-capture.sh` | The capture loop: samples CPU/GPU/mem/klog/EDAC every `INTERVAL` (default 1 s). |
| `Dockerfile.powertrip-capture` | Builds the `powertrip-capture:local` image (installs bc/dmidecode/jq/sqlite3). |
| `run-powertrip-capture.sh` | Launcher: `start|stop|restart|status`. Runs the capture container privileged so it can read RAPL power + dmesg + EDAC + DMIDIMM, and mounts host NVIDIA libs so `nvidia-smi` works inside. |

## What it captures (per ~1 s tick, CSV)
`telemetry-*.csv` columns:
- host clock, ISO time, kernel uptime (detects the reset boundary)
- CPU: k10temp **Tctl + 8 chiplets**, **RAPL package & core watts**, core freq, load avg (1/5/15), mem used/avail, `nproc`
- GPU x2: temp, power draw, mem used, util %, **PCIe link gen**, **ECC corrected/uncorrected** (where exposed)
- top-CPU processes

`klog-*.log`: raw `dmesg` ring-buffer snapshots every tick (full, unfiltered) — retains MCE/EDAC/thermal/PCIe lines.
`edac-*.csv`: rasdaemon memory-error table snapshot (CE/UE per channel) every 5 s.
`summary-*.txt`: one-per-run hardware baseline (DIMM map via dmidecode, nvidia-smi, rasdaemon summary, mounts).

## Output location (survives reboot)
By default logs to **`/buffer/powertrip/`** (host), mounted as `/capture` in the container.
`/buffer` is an LVM NVMe mount that is NOT wiped on reboot (unlike `/tmp`).

## Usage
```bash
cd /home/praneet/Workspace/pensive-ai-research
docker build -f Dockerfile.powertrip-capture -t powertrip-capture:local .
bash run-powertrip-capture.sh start     # begin sampling NOW
# ... run the heavy workload (e.g. vLLM model load / inference) ...
bash run-powertrip-capture.sh status
# after any reset: ===== inspect /buffer/powertrip/ =====
```

## Reading results after a trip
- **Temps/power vs. reset:** `telemetry-*.csv` — look at the last rows before the file stops.
  Rising package watts + rising Tctl/Tccd before the reset ⇒ thermal; no rise ⇒ memory/fabric.
- **Kernel errors:** grep `klog-*.log` for `mce|ecc|edac|sync flood|uncorrect|reset`.
- **ECC per channel:** `edac-*.csv` — a jump on channel `G`/MM4 vs. baseline confirms the known faulty DIMM.

## Notes / next hardening
- Requires a **privileged** container (RAPL + dmesg read). Can't read RAPL `energy_uj` as the normal user, hence the container.
- GPU ECC counters report `[N/A]` on these RTX PRO 5000 through `ecc.errors.*.volatile.total` (as before); DIMM ECC is tracked via `edac-*.csv` instead.
- Consider routing the model process's own stdout to a persistent path (or rely on `docker logs`) so
  crash-time vLLM logs survive — `/tmp` is volatile.
