# Power-Trip Failure-Event Catalog

Date: 2026-09-05 updated
Purpose: a running catalog of the **recurring thermal-throttle / sync-flood reset** ("power trip")
events on this AMD EPYC + 2× RTX PRO 5000 build ("pensive"). Each entry logs: what workload was
running, the event timeline, the evidence + **how each piece was sourced**, how the debugging went
(which logs helped), root-cause classification, and follow-up pointers. Use this to evaluate and add
more verbose logging/pointers when similar events recur.

> Related files:
> - `power-trip-diagnosis.md` — deep root-cause analysis + hardware fix runbook (the "why/how to fix" for the primary fault).
> - `PROJECT-TODOS.md` — task list (group B = stability/power-trip, currently stashed).
> - `power-debug-collect.sh` — safe, idempotent evidence collector.
> - `README-ramoffload-research.md` — the workloads that tend to trigger trips (RAM+VRAM offload).

---

## Index

| # | Date (local) | Reset reason | Workload at time of trip | Classification | Status |
|---|---|---|---|---|---|
| 1 | 2026-09-04 | `0x08000a00` sync-flood + thermal | RAM-offloaded 397B vLLM serve + CPU/memory H2D stress | Unc. memory error (channel G/MM4) + CPU thermal trip | Diagnosed; hardware fix pending |
| 2 | 2026-09-05 13:18 | `0x08000a00` sync-flood + thermal | Qwen3.8-Flash-Next-NVFP4 TP2 vLLM model load (safetensors, PLE CPU offload) | Same signature (channel G failing DIMM) under sustained I/O | Diagnosed; re-verified; hardware fix pending |
| 3 | 2026-09-05 15:15 | **none** (no reset) | Same load, re-run with live telemetry capture | **No trip**; clean full model load | Capture validated; run succeeded (see note) |
| 4 | 2026-09-05 16:32 | `0x08000a00` sync-flood + thermal | Same load, after PLE-fix; both GPUs in CUDA-graph warmup | **Confirmed channel G (MM4) DIMM fault**; CECC escalation → UC | Full crash-window telemetry captured; software fixes validated |
| 5 | 2026-09-06 17:44 | `0x08000a00` sync-flood + thermal | Same load, +tool-choice flags; both GPUs in main weight load | Same channel G (MM4) DIMM fault; EDAC CE not persisted pre-reset | Crash-window telemetry captured; software path still clean |

Both recorded trips share the identical reset-reason code, strongly indicating the **same root cause**
(a marginal DIMM on channel G / slot MM4) rather than two independent faults.

---

## Instance 1 — 2026-09-04 ("initial" trip during RAM+VRAM offload stress)

### Context
- Primary consumption: `Qwen3.5-397B-A17B` NVFP4 served via vLLM with `--cpu-offload-gb 200`
  (~201 GB of weights streamed from host RAM), plus `torch` 56-thread memory-bandwidth stress
  (8 GB float32 `sum()`), plus `cudaMemcpy` H2D/D2H PCIe benchmarks, plus the metrics stack.

### Event
- Machine hard-rebooted (no graceful shutdown, no kernel panic captured in time).

### Evidence (how sourced)
- **Reset reason** — at next boot, from kernel log:
  `grep "Previous system reset" /var/log/kern.log /var/log/syslog`
  ```
  x86/amd: Previous system reset reason [0x08000a00]: internal CPU thermal limit was tripped
  x86/amd: Previous system reset reason [0x08000a00]: an uncorrected error caused a data fabric sync flood event
  ```
- **ECC tally** — `sqlite3 /var/lib/rasdaemon/ras-mc_event.db "SELECT label,count(*) FROM mc_event GROUP BY label ORDER BY count(*) DESC;"`
  2741 corrected errors, concentrated on **channel G / slot MM4 (2294)**, secondary MM2/H (315), MM6/F (110).

### Debugging steps & which logs helped
- Reset reason register (`0x08000a00`) was the decisive clue → narrows to fabric/uncorrectable-memory
  reset (hardware), not a PSU/breaker trip and not a software crash.
- ECC per-channel tally isolated the single suspect DIMM.
- Idle thermal baseline (CPU ~40 °C, GPUs 32–34 °C) ruled out an *idle* thermal problem and pointed to load-induced.

### Root cause
- Primary: failing/unstable DIMM **slot MM4 (channel G)** — repeated corrected-ECC plus one
  uncorrectable → data-fabric sync-flood reset.
- Secondary: CPU internal thermal-limit trip under sustained all-core + GPU load (airflow/cooling to check).

### Outcome / follow-up
- Detailed hardware fix plan in `power-trip-diagnosis.md` §3 (reseat/swap MM4, downclock 3200→2933/2666, memtest86 channel G).
- Note: short (<1 min) stress tests do **not** reproduce; the fault needs sustained (hours) load/heat.

---

## Instance 2 — 2026-09-05 (~13:15–13:18 local) — Qwen3.8-Flash-Next-NVFP4 TP2 load

### Context
- Workload: serving **`nvidia/Qwen3.8-Flash-Next-NVFP4`** via the dedicated image
  `vllm/vllm-openai:qwen38-flash-next` (vLLM `0.1.dev20073+g8e685d198`), `--tensor-parallel-size 2`,
  `VLLM_PLE_CPU_OFFLOAD=1`, `--max-model-len 262144`.
- Checkpoint is **123.57 GiB** on ZFS; PLE-offload loads the 51B n-gram table into host RAM.
- This was the run where we had **just** diagnosed + fixed the 2-GPU NCCL hang (`NCCL_P2P_DISABLE=1`).

### Event timeline (times from the run log; UTC in log, local = PDT)
| Step | Time (UTC) | State |
|---|---|---|
| vLLM start / model resolve | 20:14:33–20:14:49 | `Qwen4ExpForConditionalGeneration` resolved |
| Engine init | 20:15:11 | V1 engine, TP2 |
| **NCCL init** | 20:15:23–24 | **passed** (world_size=2), all-reduce backends configured |
| PLE worker spawn | 20:15:24–37 | PLE-offload worker initialized, structure discovered |
| **Model weight loading** | 20:15:26 → | "Loading model from scratch", then safetensors shards loading |
| **TRIP** | ~20:15:37+ (≈13:18 local) | Both main + PLE workers mid-shard (9–27% reported); SSH session died |
- **SSH symptom seen by operator:** `Read from remote host pensive: Operation timed out ... Connection to pensive closed. client_loop: send disconnect: Broken`.

### Evidence (how sourced)
1. **Docker run log (survived the reboot)** — full stderr/stdout of the crash is retained by the
   docker daemon on the LVM-backed `/buffer/docker` storage (not on volatile `/tmp`):
   ```
   docker logs qwen38-flash-serve > /tmp/inst2_serve.log
   ```
   Key lines (see file): `Loading safetensors checkpoint shards: 0% ... 9% Completed 1/11`,
   `Loading safetensors checkpoint shards(PLE-offload): 0% ... 27% 3/11`.
   This is the primary proof that the run got **past NCCL** and died during weight load — i.e. the
   software path worked; the trip was a hardware reset mid-I/O.
2. **Reset reason (this crash confirmed at next boot)** — from kernel log:
   `grep -a "Previous system reset" /var/log/kern.log`
   ```
   2026-09-05T13:18:29.033544  x86/amd: Previous system reset reason [0x08000a00]: internal CPU thermal limit was tripped
   2026-09-05T13:18:29.033546  x86/amd: Previous system reset reason [0x08000a00]: an uncorrected error caused a data fabric sync flood event
   ```
   Timestamp 13:18:29 corresponds to the boot log being written at next boot (uptime confirms
   `boot` = 2026-09-05 13:18:10), and docker log timestamps are in UTC (20:15 UTC = 13:15 PDT),
   confirming the crash happened ~13:15–13:18 local.
3. **Boot pairing / timing proof:** `uptime -s` = `2026-09-05 13:18:10`; `TZ=UTC date` cross-check for the log-timezone offset.
4. **NCCL root-cause logs** (captured during prior debugging, `NCCL_DEBUG=INFO` on both ranks):
   - Default run: NCCL selected `P2P/CUMEM` channels → `all_reduce` timed out (watchdog, no error,
     threads at 100%). Logged in the pynccl test (`/tmp/ncclout/r0.log`, `r1.log`).
   - With `NCCL_P2P_DISABLE=1`: `all_reduce OK sum 2000.0` on both ranks.
   - This is the fix that let the TP2 serve reach weight-loading.

### Classification
- Root cause: **same as Instance 1** — uncorrectable memory error on channel G (MM4) → sync-flood reset,
  plus the CPU thermal-limit trip. The *trigger* here is the burst of sustained **disk→RAM (huge .safetensors
  set on ZFS) + PLE host-RAM load**, the exact workload class the diagnosis flags as high-risk for this channel.

### Debugging steps & which logs helped
- The **docker `docker logs <container>`** was the single most useful artifact — it survived the reboot
  and proved the crash happened in weight-load, not distributed init (ruling out a "software hang").
- **Reset-reason register** tied it to the known hardware fault (identical `0x08000a00` to Instance 1).
- Cross-checking **uptime vs. docker-log timezone offset** disambiguated when the crash fired vs. when the box rebooted.
- NCCL debug logs (Instance-2 precondition) isolated the unrelated distributed-comm hang.

### Follow-up / pointers for more verbose capture next time
- **No persistent temp/power capture existed at crash time** (TODO B2). The metrics-stack Prometheus TSDB
  was not persisting; the crash window has no temperature timeline. Add an always-on logger
  (`sensors` + `nvidia-smi` + package/DIMM power to file via cron/systemd) BEFORE the next heavy run.
- **No `pstore`/crash-dump captured** (empty `/sys/fs/pstore`), so no in-crash MCE frame was saved.
  Enable pstore/`mcelog`/`rasdaemon` record persistence so a UE frame is preserved.
- **Log architecture gap:** `/tmp/serve.log` was **volatile** and lost on reboot; the only surviving copy
  was the docker daemon's own log. Best practice: direct heavy-run logs to a persistent path (`/trunk`/`/buffer`)
  or rely on `docker logs` (retained) so crash-time stdout is never lost.
- **ZFS read path:** checkpoint is on ZFS (2.9 TB free) and vLLM logged `Auto-prefetch is disabled ... (ZFS)`;
  `--safetensors-load-strategy=prefetch` and/or staging to `/buffer` (NVMe) reduces load pressure / shortens
  the high-risk window. Consider for future runs.
- **PLE host-RAM offload is high-risk for this channel** (51B n-gram into RAM). Evaluate whether it's
  essential or can be reduced (`--max-num-seqs`/`--max-model-len`), and/or run once memory is fixed.

### Reusable verification once DIMM is addressed
1. Fix memory (TODO B1: reseat/swap MM4, downclock, memtest86 channel G).
2. Relaunch the TP2 serve with the confirmed working flags:
   `NCCL_P2P_DISABLE=1 VLLM_PLE_CPU_OFFLOAD=1` + TP2 recipe flags from `README-ramoffload-research.md`.
3. Watch for the crash window (weight-load) with temp/power logging running; confirm no new `0x08000a00`.

---

## Instance 3 — 2026-09-05 (~15:15 local) — Re-run with LIVE telemetry capture (NO power trip)

### Context
Follow-up to Instance 2. Same Qwen3.8-Flash-Next-NVFP4 TP2 serve, this time with the **crash-capture
telemetry logger armed** (`powertrip-capture` container; see `powertrip-capture-readme.md`). Goal:
re-run the load that previously tripped, with high-fidelity temp/power/EEC sampling, to characterize
the crash window — or confirm it no longer occurs.

### Event
- Model **loaded completely** — `Model loading took 38.27 GiB memory and 825.23 s` (main + PLE layers registered).
- **NO power trip / reset** — machine stayed up (`uptime -s` unchanged, same boot `13:18`). No `0x08000a00`.
- The serve then exited with a **software error** (different from the hardware fault), see "New finding" below.

### Telemetry evidence (sourced from capture, `/buffer/powertrip/telemetry-*.csv`, archived under
`evidence/power-trips/run-20260905/`)
| Metric | Peak during load | Notes |
|---|---|---|
| **CPU package power (RAPL)** | **~278 W** | vs. ~137 W idle baseline — sustained heavy load |
| **CPU Tctl** | ~55.8 °C | highest of run; below thermal trip threshold |
| **GPU0 / GPU1 power** | 70 / 75 W | |
| **GPU0 mem used** | ~58 GB (during load) | |
| Kernel log | 478k lines captured (raw dmesg snapshots) | no MCE/EDAC/uncorrected entries |
| DIMM EDAC table | 0 rows | **no memory errors logged this run** |

- Full artifacts: `evidence/power-trips/run-20260905/{telemetry.csv, kernel-log.txt, summary.txt, serve-log.txt}`.

### How sourced
- Temp/power/EEC all from the live `powertrip-capture` container (columns documented in the capture
  README). Peaks computed with `awk` on the CSV (`$4`=tctl, `$13`=pkg_w, `$23`/`$27`=g0/g1 power).
- Kernel log from the capture `klog-*.log` (raw dmesg snapshots).
- Serve stdout to **`/var/tmp/serve.log`** (persistent, not `/tmp` this time) + `docker logs`.

### New finding (separate from the power-trip) — PLE-offload loader bug
After a clean full weight-load, the engine failed at PLE-offload startup with:
```
RuntimeError: PLE offload worker failed during startup:
ValueError("There is no module or parameter named 'ngram_embedding.weight_scale' in
Qwen3_8FlashNextNGramEmbedding. The available parameters belonging to ngram_embedding
(VocabParallelEmbedding) are: {'ngram_embedding.weight'}")
```
- **Interpretation:** the loaded checkpoint's n-gram embedding exposes only `ngram_embedding.weight`
  (no `.weight_scale`), but this vLLM PLE-offload worker expects `...weight_scale` — a **version /
  loader mismatch** between the checkpoint and the `qwen38-flash-next` image's PLE-offload weight
  discovery. It is NOT the power-trip; NCCL + weight load fully succeeded.
- **Note:** this is the same family of "PLE offload weight-discovery" path that surfaced as the vLLM
  config knob. Next debugging should target PLE offload config (disable PLE offload, upgrade image,
  or use the FP8 checkpoint whose PLE layout the recipe validated) — **not** the memory hardware.

### Outcome / follow-up
- **Value delivered:** capture hardware now works end-to-end; if the power trip recurs it will be
  characterized (temps/watts/EEC per-second). This run produced a clean load with no memory errors, so
  the DIMM did **not** fault this time.
- **Next:** decide whether to (a) continue hardening the capture (auto-start at boot via systemd),
  (b) debug the PLE-offload loader bug to get a serving endpoint, or (c) run a deliberate high-stress
  memory test to try to reproduce the trip with capture armed.

---

## Instance 4 — 2026-09-05 (~23:26–23:32 UTC / 16:32 local) — POWER TRIP during model load, WITH live capture

### Context
The PLE-offload software bugs were root-caused and fixed (see `README-ramoffload-research.md` /
vLLM issue #54765): the FP8 PLE selector was patched (`VLLM_QWEN38_PLE_FP8_SCALE=1`), and
`--disable-custom-all-reduce` + `--cap-add SYS_PTRACE` (unconfined seccomp/apparmor) resolved the
pidfd/CUSTOM-all-reduce errors. With those fixes the run progressed much farther than before:
- `PLE weight loading complete` — `matched 132 checkpoint tensor(s), loaded 5 offload entries, verified 2/2 materialized parameter(s)`.
- Both GPU workers then loaded weights (`~40.9 GB` / `~40.5 GB` VRAM committed) and were in the
  **CUDA-graph capture / KV-init** phase (both GPUs active) when the machine **hard-reset at 16:32:20 local**.

### Telemetry (the payoff — full crash-window trace captured)
Data from the live capture `telemetry-20260905-230551.csv` (archived under
`evidence/power-trips/run-20260905-trip/telemetry-crashwindow.csv`). The **last telemetry row is
23:26:50 UTC**; the reset reason was logged at 23:32:20 UTC — a ~5.5 min gap where the box went into a
degraded/hard-lock state (disk writes stalled) before the hard reset.

| Metric (during load / pre-trip window) | Value |
|---|---|
| CPU **package watts** | **sustained ~260–282 W** (peak 310 W this run) |
| CPU **Tctl** | ~54–57 °C (peak 58.8 °C), tccd1 ~56–57 °C |
| **GPU0 / GPU1 power** | ~70 / 76 W |
| **GPU0 / GPU1 VRAM committed** | ~40.9 / 40.5 GB |
| Room/available RAM | ~808 GB free (PLE table loaded into RAM) |

### Root-cause ECC evidence (bold proof of the memory fault)
Sourced from `journalctl -k -b -1` (the boot the load ran in) — archived as
`evidence/power-trips/run-20260905-trip/ecc-mce-evidence.txt`:
```
Sep 05 15:07:31  EDAC MC0: 1 CE on mc#0csrow#0channel#6 (channel:6 ...)
Sep 05 15:37:00  EDAC MC0: 1 CE on mc#0csrow#0channel#6 (channel:6 ...)
Sep 05 16:11:57  EDAC MC0: 1 CE on mc#0csrow#0channel#6 (channel:6 ...)
...then, at next boot:
Sep 05 16:32:20  x86/amd: Previous system reset reason [0x08000a00]:
                 internal CPU thermal limit was tripped
                 an uncorrected error caused a data fabric sync flood event
```
- **ECC channel 6 = channel G = slot MM4** (per the DIMM↔channel map in `power-trip-diagnosis.md`:
  `MM4=G`). These are **corrected** (CE) errors on the **prime-suspect failing DIMM**, escalating during
  the sustained model load, culminating in an **uncorrectable** error → sync-flood reset.

### Classification / conclusion
- **Same root cause as Instances 1 & 2**: a marginal/failing DIMM on **channel G (slot MM4)**. The
  corrected-ECC (CECC) burst during the high-RAM-traffic model load + the eventual UC → data-fabric
  sync-flood reset (+ CPU thermal trip) is the classic signature.
- **The capture worked.** This is the first trip where we have a per-second temp/power/GPU trace
  leading up to it. It confirms the trip is **load-induced memory/fabric fault** (package watts ~280 W,
  both GPUs active), not thermal alone — though Tctl was climbing.
- The **software work is done**: with the fixes this run got from `load weights → PLE registration →
  CUDA-graph capture` (progress that previously failed). The only thing stopping the model from serving
  was the **hardware reset** mid-warmup.

### Follow-up
- The recurring fix (reseat/swap **MM4**, downclock memory, memtest86 channel G) is **the gating item**.
- Until the DIMM is fixed, heavy model loads with large host-RAM (PLE offload) + both GPUs active will
  keep tripping. Mitigations: **lower the thermal/power ceiling** (avoid both-GPU warmup simultaneously),
  reduce host-RAM offload, and/or auto-start the capture logger so the next trip is captured automatically.

---

## Instance 5 — 2026-09-06 (~00:39–00:44 UTC / 17:44 local) — POWER TRIP during model reload (with tool-choice flags)

### Context
This run re-launched the serve after we added `--host 0.0.0.0 --enable-auto-tool-choice
--tool-call-parser qwen3_xml` (to fix the client's "auto tool choice" error). Software config was
otherwise unchanged (NCCL P2P fix, FP8-PLE selector, disable-custom-all-reduce, pidfd caps). The PLE
offload worker completed and registered (`matched 132 checkpoint tensor(s), verified 2/2 materialized
parameter(s)`); the main GPU worker was at **~64% of 11 safetensors shards** (both GPUs loading,
~40.9 GB each) when the machine **hard-reset at 17:44:01 local (00:44:01 UTC)**.

### Telemetry (crash-window trace captured)
From the live capture `telemetry-20260905-233422.csv` (archive: `evidence/power-trips/run-20260905-trip3/`).
The **last valid telemetry row is 00:39:35 UTC**, the reset reason logged at 00:44:01 UTC — a ~4.5 min
degraded/hard-lock gap before the hard reset (same pattern as Instance 4).

| Metric (during load / pre-trip window) | Value |
|---|---|
| CPU **package watts** | **sustained ~260–276 W** (peak 300 W this run) |
| CPU **Tctl** | ~51–54 °C, tccd1 ~55–57 °C |
| **GPU0 / GPU1 power** | ~67–72 W |
| **GPU0 / GPU1 VRAM committed** | ~40.9 / 40.5 GB |
| Available RAM | ~869 GB free (PLE table in RAM) |

### Reset / evidence
```
Sep 05 17:44:01  x86/amd: Previous system reset reason [0x08000a00]:
                 internal CPU thermal limit was tripped
                 an uncorrected error caused a data fabric sync flood event
Sep 05 16:32:06  mce: HEST corrected error threshold limit: 10   (EDAC/HEST armed in trip boot)
```
- The **`0x08000a00`** reset signature is identical to Instances 1, 2, 4 — the same **channel G / slot MM4
  DIMM** fault pattern, triggered by the sustained both-GPU weight load + host-RAM (PLE) traffic.
- Unlike Instance 4, the per-CEC **channel#6 EDAC lines were NOT persisted** in the trip boot's journal
  (rasdaemon/journald didn't flush them before the hard reset); the HEST threshold and the reset-reason
  register remain the definitive evidence. This is a **capture gap worth noting**: the EDAC CE events
  that precede the UC are sometimes lost because rasdaemon writes to a DB that gets recreated on reboot,
  and journald may not have flushed the MCE lines in time.

### Classification / conclusion
- **Same root cause as Instances 1, 2, 4**: marginal/failing DIMM on **channel G (slot MM4)**. The
  sustained both-GPU model load + host-RAM PLE traffic is the trigger. The trip recurs because the
  memory fault is **hardware** — it can fire on any sustained heavy-load run.
- **Software path remains clean** — the serve got past PLE registration and into the main weight load
  with the tool-choice flags applied; only the hardware reset stopped it.

### Follow-up / hardening notes
- **Gating fix unchanged:** reseat/swap **MM4**, downclock memory (3200→2933/2666), memtest86 channel G.
- **EDAC persistence gap:** the per-CEC EDAC lines are not reliably captured across a sync-flood reset
  (rasdaemon DB recreated each boot; journal may not flush MCE lines). Recommend capturing `ras-mc-ctl
  --summary` AND persisting `mcelog`/EDAC sysfs counters to disk continuously (the crash-capture logger
  could also periodic-read `rasdaemon`), so the CE-before-UC timeline survives.
- **PM/thermal ceiling:** the pre-trip package watts hover ~270 W. Consider a lower GPU power limit
  (`nvidia-smi -pl`) and/or `--enforce-eager` to reduce warmup peak, plus checking CPU cooling.

---

## Summary of recurring pattern
- **Signature:** reset reason `0x08000a00` = "internal CPU thermal limit was tripped" + "an uncorrected
  error caused a data fabric sync flood event".
- **Location:** corrected/uncorrected ECC concentrates on channel G / slot MM4 (prime suspect; MM2/H, MM6/F secondary).
- **Trigger class:** sustained **host-RAM heavy** workloads — RAM+VRAM offload serving, and large
  checkpoint loads that push many GiB of RAM + disk I/O — are what surface the marginal DIMM.
- **Why it keeps recurring:** the memory fault is *hardware*; it will keep interrupting any
  RAM/disk-intensive AI run until the DIMM/cooling is fixed. Short tests don't reproduce it.
- **Status of the AI-serving software path (as of Instance 4):** the Qwen3.8-Flash-Next-NVFP4 TP2 load
  is now *software-clean* (NCCL P2P fix + FP8-PLE selector + disable-custom-all-reduce + pidfd caps),
  reaching full weight load → PLE registration → CUDA-graph capture. The **only** remaining blocker to
  serving is the hardware memory fault (channel G / MM4), which reset the box during warmup.
- **Service got a real serving win (Instance 4 time-frame) before the next trips:** one run did reach
  `Started server process` + `Application startup complete` and returned correct answers; then the box
  tripped on a subsequent reload during the both-GPU weight-load phase.
- **The recurring fault fires on ANY sustained heavy-load run** — it is not deterministic per-run; some
  loads complete, others trip. This is the hallmark of marginal hardware (channel G / MM4). Confirmed by
  Instances 1, 2, 4, and 5 all sharing the identical `0x08000a00` signature.

---

## Isolation test — 2026-09-06 (~01:27 UTC) — `--load-format dummy` (NO power trip)

**Goal:** per the data-fabric diagnosis, isolate whether the sync-flood reset is caused by the
**weight-copy / PCIe / data-fabric burst** (streaming ~80 GB of tensor weights host→GPU across Gen4
lanes + 8-channel RDIMM) vs. the **model structure allocation** itself.

**Test:** relaunch the same serve (TP2, PLE CPU offload, all prior fixes) but with
`--load-format dummy --enforce-eager`. `dummy` allocates the model structure in RAM+VRAM **without
streaming tensor weights across PCIe**; `--enforce-eager` skips the CUDA-graph capture that causes a
large concurrent fabric burst.

**Result:** ✅ **NO power trip.** The box survived the full model-structure initialization, PLE
registration, and dummy-weight allocation (uptime stable > 48 min; reset-reason register shows no new
`0x08000a00`). Telemetry during the run: ~160 W package, ~42 °C Tctl, GPUs ~8–15 W (no weight copy),
GPU mem ~41 GB (structure allocated). No sync-flood reset.

**Conclusion:**
- The **fault is in the weight-copy / PCIe / data-fabric burst path**, NOT the model-allocation or
  memory-init path. Real weight streaming (Instances 2, 4, 5) tripped it; dummy load did not.
- This aligns with the "data fabric sync flood" diagnosis and points to **PCIe signal integrity /
  high Gen4 + 8-channel RDIMM offload burst** as the trigger, i.e. likely mitigated by **GPU PCIe slots
  → Gen 3** and **RAM 3200 → 2933/2666**, plus possibly vSOC/VDDG bump and Resizable BAR.

### Next actions (hardware/BIOS — pending user)
1. **memtest86+** on channel G / slot MM4 (rule out a genuinely failing DIMM; we saw EDAC CE there).
2. **BIOS:** drop GPU PCIe slots to **Gen 3**; RAM **3200 → 2933/2666**; ensure **Resizable BAR /
   Above 4G Decoding**; disable PBO/undervolt; bump **vSOC / VDDG_IOD / VDDG_CCD**; flash latest
   Huananzhi H12D-8D BIOS. Change ONE at a time and re-test the real load with the capture logger armed.
