# AGENTS.md — Power-Trip Investigation Runbook (build "pensive")

> Purpose: let ANY fresh session (or agent, with no prior conversation history) pick up the
> **data-fabric sync-flood / power-trip investigation** on this machine and continue the deep-dive.
> This is the authoritative entry point. Detailed docs live in the sibling files (see "Reference docs").

## 1. TL;DR — what we're investigating

- This machine ("pensive") **hard-resets under heavy memory/fabric load** (e.g. big LLM weight loads).
- A reset prints an **AMD "Previous system reset reason"** register value. There are **3 distinct
  classes seen** — decode with `recipe/reset-reason-decoder.sh`:
  - `0x08000a00` (bit27) = **uncorrected error → data-fabric sync flood** + bit9 thermal. This is the
    **memory-fault** signature.
  - `0x00080a00` (bit19) = **software wrote 0x6 to reset-control register 0xCF9** + bit9 thermal. A
    **software-initiated warm reset — NOT a memory sync-flood** (e.g. the 2026-09-07 GLM-5.3 trip).
  - `0x00200a00` / `0x00200800` (bit21) = **ACPI power-state transition** (± bit9 thermal). A
    power/firmware reset (benign / not memory).
- **Long-run suspect:** a **marginal/failing DIMM on channel G — slot MM4** (secondary MM2/channel H,
  MM6/channel F). Its **lifetime cumulative** CE count is by far the highest.
- ⚠️ **Do NOT use lifetime `ras-mc-ctl --summary` totals as per-trip evidence.** Those counts are
  **historic accumulation** (96% of channel-6 CEs landed on a single Sep 3–4 burst, not on the trips).
  Attribute each trip using **per-boot window** CE counts (`recipe/ecc-per-window.sh`) + the reset-reason
  bitclass above.
- This is a **hardware fault potential, not a software bug.** Software can only reduce the *load*.
- **EDUCATIONAL doc:** why the reset can't be intercepted → `why-databric-syncflood-not-interceptable.md`.
- **Status (2026-09-08): the suspect DIMM (channel G / MM4) and its NUMA-node half have been physically
  removed for isolation/RMA testing** (memtest86 clean on the remaining ~512 GB). This is temporary —
  see `SYSTEM-SPEC.md`'s hardware-status banner and `power-trip-diagnosis.md` for the reinstall plan
  (RMA the bad stick, put the other 7 back, then the replacement). RAM/NUMA figures below are stale the
  moment this changes; treat `SYSTEM-SPEC.md` as the source of truth and re-verify with `numactl -H`.

## 2. Hardware context (summary)

| Component | Spec |
|---|---|
| CPU | AMD EPYC 7663, 56c/112t, 1 socket, **4 NUMA nodes (NPS4)** |
| GPU | 2× NVIDIA RTX PRO 5000 72 GB Blackwell (`sm_120`), **no NVLink, cross-NUMA** (GPU0=node3, GPU1=node0) |
| RAM | 1 TiB nameplate (8× 128 GB Micron DDR4-3200 8-rank RDIMM); **live now ~499 GiB / 4 DIMMs** — marginal DIMM (channel G / MM4) pulled for isolation testing, see status note above and `SYSTEM-SPEC.md` §3 for current figures |
| Storage | `/trunk/ai` ZFS (17T, ~82% full), `/buffer` NVMe LVM, `/` ext4 |
| PSU | Seasonic Prime 1600W (NOT the constraint) |
| Software | Docker + vLLM (patched) + capture logger; NVIDIA Container Toolkit |

**Topology note:** the two GPUs are on **different root complexes / NUMA nodes** with **no NVLink**, so
GPU↔GPU P2P is impossible → NCCL must use `NCCL_P2P_DISABLE=1` (or `NCCL_P2P_LEVEL`). See "Serving".

## 3. Document map (read these for detail)

| File | Contents |
|---|---|
| `power-trip-diagnosis.md` | Deep root-cause analysis + hardware fix runbook (DIMM map, channels). |
| `power-trip-instances.md` | **Failure-event catalog, Instances 1–5** + isolation test; every trip logged with evidence. |
| `recipe/MODEL-CATALOG.md` | Model catalog + fit verdicts + Recipe A/B. |
| `recipe/serve-qwen38-flash-next-nvfp4.sh` | One-shot server launcher (arms capture + EDAC watch + power cap + serve). |
| `recipe/edac-ce-watch.sh` | Corrected-ECC **pre-trip alert** monitor (**per-interval new-CE delta**, not cumulative). |
| `recipe/diag-dimm-fault.sh` | DIMM fault diagnostic → **warranty/RMA-ready** output (separates cumulative vs per-boot). |
| `recipe/ecc-per-window.sh` | **Window-based** per-channel CE accounting (per day/boot/lifetime) from rasdaemon DB. |
| `recipe/reset-reason-decoder.sh` | Decodes AMD "Previous system reset reason" bitfield (3 reset classes). |
| `powertrip-capture-readme.md` | Crash-capture telemetry logger docs. |
| `README-ramoffload-research.md` | RAM+VRAM offload research + empirical results. |
| `why-databric-syncflood-not-interceptable.md` | Educational: CE vs UE, why a fabric flood can't be intercepted. |
| `RESUME-NOTE.md` | Live "how to bring the server up" + decisions. |
| `SYSTEM-SPEC.md` | Hardware/software spec. |
| `evidence/power-trips/` | Raw captured traces (telemetry CSVs, kernel/serve logs, reset reasons). |

## 4. How to run / arm the monitors (the launch entry point)

**All debug logging (capture + EDAC watch + power cap) is manual — `restart: no`, armed only on demand:**

```bash
cd /home/praneet/Workspace/pensive-ai-research/recipe
./serve-qwen38-flash-next-nvfp4.sh --start    # arms capture, edac-watch, power cap, and serve
./serve-qwen38-flash-next-nvfp4.sh --status   # status of serve + capture
./serve-qwen38-flash-next-nvfp4.sh --stop     # stop serve
./serve-qwen38-flash-next-nvfp4.sh --no-capture --restart   # restart without capture
```

The `--start` command does, in order:
1. `ensure_image` — verifies/rebuilds the patched vLLM image if missing.
2. `arm_capture` — starts `powertrip-capture` (per-second telemetry → `/buffer/powertrip/`).
3. `arm_edac_watch` — starts `edac-ce-watch.sh` (CE pre-trip alerts → `/var/tmp/powertrip/`).
4. `apply_power_cap` — `nvidia-smi -pl 250`.
5. `start_serve` — launches the serve with all working flags.

**Script fixes already applied:** `SKIP_CAP` unbound-variable bug fixed; `--served-model-name
nvidia/Qwen3.8-Flash-Next-NVFP4`; EDAC watch wired in. `powertrip-capture` and `edac-ce-watch` are
**manual (`restart: no`)** — they do NOT auto-start on boot (deliberate).

## 5. The currently-working serve recipe

Built into `recipe/serve-qwen38-flash-next-nvfp4.sh` (patched image `vllm/vllm-openai:qwen38-flash-next-patched`):
- TP2, `--quantization modelopt`, native **262144** context, `--max-num-seqs 2`.
- **CUDA graphs ON** (no `--enforce-eager`) → ~39–55 tok/s (vs ~12 tok/s with eager).
- Env: `NCCL_P2P_DISABLE=1`, `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_QWEN38_PLE_FP8_SCALE=1`,
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Flags: `--disable-custom-all-reduce --max-parallel-loading-workers 1 --host 0.0.0.0
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --trust-remote-code`.
- Docker opts: `--cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined`.

**Software blockers that were fixed (all baked in):** NCCL cross-NUMA/no-NVLink P2P hang → `NCCL_P2P_DISABLE=1`;
FP8-PLE selector bug (vLLM issue #54765) → patched `ple_layer.py` + `VLLM_QWEN38_PLE_FP8_SCALE=1`;
CUSTOM all-reduce CUDA error → `--disable-custom-all-reduce`; `pidfd_getfd` permission → `SYS_PTRACE` caps;
**NCCL host-cuMem segfault on a memory-less GPU-local NUMA node** (`ncclCuMemHostEnable`/`cuMemCreate`,
hit 2026-09-08 after the DIMM pull left GPU0's node memory-less) → the script auto-detects per-launch
and sets `NCCL_CUMEM_HOST_ENABLE=0` only when needed (see `SYSTEM-SPEC.md` §1 topology note); this is
NUMA-topology-driven, not hardcoded, so it self-adjusts as DIMMs are reinstalled.

## 6. How to investigate a NEW trip (given limited/no capture)

The crash-capture is manual and often not running. Investigate from OS post-mortem sources:

```bash
# 1. Did it reboot (vs a clean shutdown)? / which boot?
uptime -p; uptime -s
journalctl --list-boots | tail -8

# 2. THE reset reason (primary evidence)
grep -a "Previous system reset" /var/log/syslog /var/log/kern.log | tail -6
bash recipe/reset-reason-decoder.sh --history   # decode each code (3 classes)
#   0x08000a00 = MEMORY fault (bit27 sync flood). 0x00080a00 = software 0xCF9 warm reset.
#   0x00200a00/0x00200800 = ACPI/power transition (NOT memory).

# 3. ECC / MCE evidence from the boot that tripped (boot -1 = previous boot)
journalctl -k -b -1 | grep -aiE "EDAC MC0:.*CE on|Machine check|CECC|UECC|uncorrect|sync flood|channel#6|syndrome"
#   (note: CEs may not be persisted if the reset was fast; channel#6 = channel G = MM4)

# 4. Per-channel CE tally — USE WINDOW / PER-BOOT, not lifetime
bash recipe/ecc-per-window.sh --boot -1   # fresh CEs in the tripped boot ONLY
bash recipe/ecc-per-window.sh --all       # lifetime (for context; NOT per-trip evidence)
ras-mc-ctl --summary                      # lifetime cumulative (historic; NOT proof per-trip)
bash recipe/diag-dimm-fault.sh            # full warranty-ready output (separates cumulative vs per-boot)

# 5. What stage was the model load at? (if server was running)
grep -a -E "Loading safetensors|Model loading took|PLE weight loading complete|registered" /var/tmp/serve.log
docker logs qwen38-flash-serve 2>&1 | tail -20
```

## 7. EDAC channel → DIMM slot mapping (for warranty)

```
EDAC channel# -> channel letter -> slot:   A=MM7 B=MM5 C=MM3 D=MM1 E=MM8 F=MM6 G=MM4 H=MM2
```
- **Prime suspect: channel#6 = Channel G = slot MM4** (highest CE counts, thousands).
- Secondary: channel#7 = Channel H = slot MM2; channel#5 = F = MM6.

## 8. Key findings / conclusions so far

- Recurring reset = **marginal DIMM (channel G / MM4)** scaled CE→UE. Recorded in Instances 1–5.
- The **GPU power cap (250W) is NOT the stabilizer** — it's defensive; the real lever is the
  burst-reduction flags (serialize load via `--max-parallel-loading-workers 1`, `spawn`, PLE offload)
  + fixing the DIMM.
- **`--load-format dummy` isolation test** did NOT trip → confirms the fault is in the **weight-copy /
  PCIe / data-fabric burst** path, not model allocation.
- **NPS1 / 3rd-GPU:** NPS1 would interleave all memory across the failing channel G (worse); a 3rd GPU
  adds more fabric/memory load on the marginal DIMM. **Fix the memory first.**
- **Current status (as of 2026-09-06):** the `--start` script arms serve + capture + edac-watch; all three
  are `restart: no`. The full load completed **without a trip** in session (serve reached HTTP 200, native
  262k context, CUDA graphs). The recurring DIMM fault (channel G / MM4) is still latent.

## 9. The real fix (gating item) — hardware

1. **memtest86+** on channel G / MM4 (and H/F).
2. **Reseat/swaP MM4** (or replace — **valid RMA case** using `recipe/diag-dimm-fault.sh` output).
3. **Downclock RAM** 3200 → 2933/2666.
4. **GPU slots → Gen 3**; ensure Resizable BAR / Above 4G Decoding; bump vSOC/VDDG; latest BIOS.
   (User noted: could not find RAM-clock setting in Huananzhi H12D-8D BIOS; Gen3/ResizeBAR/decoding were tried.)
5. Re-test with capture armed; change ONE variable at a time.

## 10. Safety note — why the reset is protective

The reset is a **fail-safe**, not a malfunction: an uncorrectable data-integrity error means the machine
can't be trusted; continuing risks silent corruption. There is **no software gate** that can intercept it
(see `why-databric-syncflood-not-interceptable.md`). The only actionable gate is **pre-fault**: watch
corrected-ECC counts (`edac-ce-watch.sh`) and fix the DIMM before a UE escalates.
