# Power-Off / Reboot Diagnosis (build "pensive")

Date: 2026-09-04
Purpose: root-cause and fix the recurring **sudden power-off / hard reboot** on this AMD EPYC + 2× RTX PRO 5000 workstation, plus provide a reusable runbook to quickly isolate the faulty component next time.

---

## Status update — 2026-09-08 (isolation/RMA testing in progress)

Acted on the fix plan below (§3, step 1): **physically removed the suspect DIMM (slot MM4 / channel G)
and its NUMA-node half (4 of 8 DIMMs total)** to isolate it. memtest86 on the remaining ~512 GB came
back **clean**. This is a temporary diagnostic configuration, not the final state:

1. Current: 4 DIMMs populated (~499 GiB), NUMA nodes 2 and 3 memory-less. Re-testing/monitoring for
   trips in this reduced config.
2. Next: identify the specific bad stick among the 4 removed (this run only excluded "at least one of
   these four," not which), so it can be RMA'd individually.
3. Then: reinstall the other 7 known-good sticks (interim state, one socket-half short until the RMA
   replacement arrives) and re-test again.
4. Finally: reinstall the RMA replacement, back to the full 8-DIMM / 1 TiB config.

`SYSTEM-SPEC.md`'s hardware-status banner and §3 track the live RAM figure at each stage — re-verify
with `numactl -H` rather than trusting any hardcoded number, here or elsewhere, until this settles.

Side effect discovered during re-test: with GPU0's local NUMA node (3) now memory-less, NCCL's
host-memory registration probe (`ncclCuMemHostEnable`/`cuMemCreate`) segfaults on launch instead of
degrading gracefully. Fixed in `recipe/serve-qwen38-flash-next-nvfp4.sh` with a NUMA-topology
auto-detect (sets `NCCL_CUMEM_HOST_ENABLE=0` only when a GPU's local node is memory-less) — this is not
a hardware issue, just a heads-up for whichever intermediate DIMM configuration you're testing next.

---

## 1. What happened (the tripping incident)

While running **memory-bandwidth stress + full-CPU (56-thread) benchmark + GPU PCIe H2D benchmarks**, with a RAM-offloaded 397B LLM server (~103W GPU) and the metrics stack also running, the machine **hard-rebooted**.

The **last thing that ran before the trip** (for context):
1. `torch` multi-threaded (56 threads) `sum()` on an 8 GB float32 array → all cores at 100%, sustained memory bandwidth usage.
2. `cudaMemcpy` H2D / D2H PCIe bandwidth benchmarks on the GPU.
3. Already running: Qwen3.5-397B-A17B RAM-offload vLLM server (`--cpu-offload-gb 200`, ~201 GB weights in system RAM) + GPU; plus the `metrics-stack-*` containers.

## 2. Root cause — evidence

**Reset reason register (logged at next boot):**
```
x86/amd: Previous system reset reason [0x08000a00]: internal CPU thermal limit was tripped
x86/amd: Previous system reset reason [0x08000a00]: an uncorrected error caused a data fabric sync flood event
```
→ The reset was caused by an **uncorrectable memory/fabric error (data-fabric sync-flood reset)** together with a **CPU internal thermal limit trip**. This is a *hardware* reset, not a PSU/breaker trip.

**ECC error log (rasdaemon `ras-mc-event.db`): 2741 corrected errors, all `Corrected`,**
heavily concentrated on **one memory channel — a classic marginal/failing DIMM**:

| EDAC ch | DMI channel | Slot | Corrected/UE |
|---|---|---|---|
| 6 | G | **MM4** | **2294** ★ prime suspect |
| 7 | H | MM2 | 315 |
| 5 | F | MM6 | 110 |
| 0–4 | A–E | MM7/MM5/MM3/MM1/MM8 | ~75 (background) |

**DIMMs:** 8 × Micron `144ASQ16G72PSZ-3G2E3` **128 GB DDR4-3200 8-rank ECC** (1 TB total).
Slot↔channel: `MM1=D MM2=H MM3=C MM4=G MM5=B MM6=F MM7=A MM8=E`.

**Idle thermal state (normal):** CPU Tctl ~40 °C, NVMe ~48 °C, GPUs 32–34 °C. So the trip is **load-induced**, not an idle thermal problem. CPU is already capped at **2.0 GHz** (`bios_limit`/`scaling_max_freq=2000000`, governor `schedutil`).

### Conclusion
- **Primary:** an unstable/failing **DIMM in slot MM4 (channel G)** — repeated corrected-ECC, one uncorrectable → data-fabric sync-flood reset. Secondary suspects MM2 (H) and MM6 (F).
- **Secondary:** CPU **thermal limit trip** under sustained all-core + GPU load (cooling/airflow should be checked).
- These likely also explain the original install difficulty, kernel panics, and the many BIOS overrides needed.
- **Critical implication for AI work:** the **RAM+VRAM offload** approach streams constant system-RAM traffic, which is exactly what stresses the failing channel → it will keep tripping until the memory is fixed.

## 3. Fix plan (hardware / BIOS)

1. **Reseat / swap / replace the DIMM in slot MM4 (channel G)** — highest priority. Also try **MM2 (H)** and **MM6 (F)**.
2. **Downclock memory** to a stable speed if needed (3200 → 2933/2666): this is 8-rank 128 GB DIMMs × 8 channels on a single socket, which is aggressive; marginal timing is a plausible cause of the CECC/UE.
3. Run **memtest86** targeting channel G (and H/F) to confirm.
4. **CPU cooling:** verify fans/airflow ramp under load; the CPU already has a 2.0 GHz cap.
5. Re-test under controlled load (see runbook) after the DIMM swap.

## 4. Reusable runbook — quickly isolate the faulty part next time

Run WITHOUT inducing heavy load first; only do the stress step after a DIMM change and while watching temps.

### Step A — Collect evidence (safe, idle)
```bash
bash /home/praneet/Workspace/pensive-ai-research/power-debug-collect.sh
```
Captures: reset reason, MCE/panic lines, per-channel ECC tally, CPU/GPU/NVMe temps, CPU freq/governor/boost, package power, GPU power/temps, running containers, DMI memory map, and memory slot↔channel map.

### Step B — Read the DMI DIMM map (needs a short-lived privileged container)
```bash
docker run --rm --privileged ubuntu:24.04 bash -c \
 'apt-get update -qq && apt-get install -y -qq dmidecode >/dev/null && \
  dmidecode -t memory | grep -iE "Locator: MM|Bank Locator"'
```

### Step C — Query the ECC tally by channel (the #1 fault locator)
```bash
[ -r /var/lib/rasdaemon/ras-mc_event.db ] && \
  sqlite3 /var/lib/rasdaemon/ras-mc_event.db \
    "SELECT label,count(*) FROM mc_event GROUP BY label ORDER BY count(*) DESC;"
```

### Step D — Isolate CPU vs RAM vs GPU (CONTROLLED stress, watch temps & power)
> Run one at a time with a hard monitor that stops the test if Tctl > ~90 °C, or if package/GPU power exceeds a threshold. Keep the other subsystems mostly idle. After each, check the reset-reason (Step A) for a new trip.

1. **CPU-only:** `stress-ng --cpu 112 --cpu-method ackermann 60` (or `sysbench cpu`). Watch Tctl.
2. **Memory-only:** `stress-ng --vm 24 --vm-bytes 16G --vm-method write64 --timeout 120s` (use a valid method: `write64`/`read64`/`galpat-0`/`walk-0d`; **`row` is NOT valid** and silently runs nothing). Short (<1 min) tests may NOT reproduce — the failure needed *sustained* load/heat (the AI offload ran for hours). Run several minutes or match the offload workload to surface errors. Watch ECC counter (Step C) before/after.
3. **GPU-only:** a single `cuda` matmul or `nvidia-smi -pl` controlled run, or a light inference prompt. Watch GPU temp/power.

Interpretation: if ECC errors **jump on channel G (MM4)** during the memory test → RAM is the culprit. If it trips only under CPU/GPU with no new ECC → thermal. If it trips with zero ECC growth and temperature normal → look at power delivery/PSU next.

### Step E — Track re-runs
After each step run `power-debug-collect.sh` and compare the ECC tally and reset-reason lines. A stable run shows no new "uncorrected error / sync flood" reset reason and no growth on channel G.

## 5. Files
- `/home/praneet/Workspace/pensive-ai-research/power-debug-collect.sh` — evidence collector (safe).
- `/home/praneet/Workspace/pensive-ai-research/power-trip-diagnosis.md` — this document.
