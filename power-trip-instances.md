# Power-Trip Failure-Event Catalog

Date: 2026-09-07 updated
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

> ⚠️ **2026-09-08 status update:** the prime-suspect DIMM (channel G / MM4) and its NUMA-node half (4 of
> 8 DIMMs) have been **physically removed for isolation/RMA testing**; memtest86 on the remaining
> ~512 GB came back clean, and no new trips have occurred in this reduced config so far. Read every
> "hardware fix pending" status below in that light — see `power-trip-diagnosis.md`'s 2026-09-08 status
> update for the full reinstall plan (identify the specific bad stick → RMA → reinstall the other 7 →
> reinstall the replacement). This is a live, changing state, not a completed fix.

---

## Index

| # | Date (local) | Reset reason | Workload at time of trip | Classification | Status |
|---|---|---|---|---|---|
| 1 | 2026-09-04 | `0x08000a00` sync-flood + thermal | RAM-offloaded 397B vLLM serve + CPU/memory H2D stress | Unc. memory error (channel G/MM4) + CPU thermal trip | Diagnosed; hardware fix pending |
| 2 | 2026-09-05 13:18 | `0x08000a00` sync-flood + thermal | Qwen3.8-Flash-Next-NVFP4 TP2 vLLM model load (safetensors, PLE CPU offload) | Same signature (channel G failing DIMM) under sustained I/O | Diagnosed; re-verified; hardware fix pending |
| 3 | 2026-09-05 15:15 | **none** (no reset) | Same load, re-run with live telemetry capture | **No trip**; clean full model load | Capture validated; run succeeded (see note) |
| 4 | 2026-09-05 16:32 | `0x08000a00` sync-flood + thermal | Same load, after PLE-fix; both GPUs in CUDA-graph warmup | **Confirmed channel G (MM4) DIMM fault**; CECC escalation → UC | Full crash-window telemetry captured; software fixes validated |
| 5 | 2026-09-06 17:44 | `0x08000a00` sync-flood + thermal | Same load, +tool-choice flags; both GPUs in main weight load | Same channel G (MM4) DIMM fault; EDAC CE not persisted pre-reset | Crash-window telemetry captured; software path still clean |
| 6 | 2026-09-06 10:13 | `0x00200a00` thermal + ACPI (**no sync-flood bit**) | Qwen3.8-Flash-Next-NVFP4 TP2 serve, ~15 min after serve start (weight-load window) | Anomalous code: thermal/ACPI event OR unlatched sync-flood bit | No capture coverage |
| 7 | 2026-09-06 16:24 | `0x08000a00` sync-flood + thermal | Same load, ~13 min after serve start (weight-load window) | Same channel G (MM4) DIMM fault; EDAC CE again not persisted | No capture coverage |
| 8 | 2026-09-07 16:54 | `0x00080a00` **software 0xCF9 reset** + thermal (**NOT a sync flood**) | GLM-5.3-Flash (new model) TP2 vLLM load, `--load-format dummy`, fp8-MoE finalize stage | **Distinct from memory fault** — software-initiated warm reset; only 68 CEs on ch6 | New model attempt; different reset class |
| — | 2026-09-08 00:06 & 01:41 local | `0x00080800`/`0x00080a00` software 0xCF9 reset (×2, same signature as #8) | Unknown — box was unattended overnight between Instance 8's reset (19:41) and the next morning's manual DIMM-pull reboot (07:18) | **Newly found 2026-09-08, not yet investigated** — no capture coverage (capture was already quiet before Instance 8 itself), no docker container evidence checked | Undiagnosed; flagged for follow-up, not yet root-caused |
| 9 | 2026-09-08 22:46 local | **No reset** — clean host-wide Linux OOM kill (kernel-logged, `journalctl -k`) | GLM-5.3-Flash TP2 `--dummy` rerun, ~13 min after launch, during engine-core distributed startup / CPU-offload allocation (before reaching Instance 8's fp8-MoE-finalize stage) | **RAM/NUMA capacity, not a fault** — global OOM-killer sacrificed unrelated host processes (`dbus-daemon`, a user `systemd` session) before killing the vLLM worker; EDAC clean, no CE ramp | Root-caused; see full write-up below |

> ⚠️ **2026-09-07 correction (see "Corrected findings" below):** earlier instances (1–7) were classified
> as "channel G / MM4 DIMM fault" primarily on the basis of **lifetime cumulative** `ras-mc-ctl --summary`
> counts. Those counts are a **historic** accumulation (96% landed Sep 3–4) and are **not** per-trip
> evidence. Per-boot attribution (via the rasdaemon DB) shows the trip boots themselves carried far fewer
> fresh CEs (see the corrected-accounting table). Re-read the per-instance classifications with that in mind.

All recorded trips share the identical reset-reason code, strongly indicating the **same root cause**
(a marginal DIMM on channel G / slot MM4) rather than independent faults. Instance 6 is the lone
register-level exception (`0x00200a00`, no sync-flood bit) — see its entry for the ambiguity.

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

## Instance 6 — 2026-09-06 (10:13 local) — POWER TRIP during model load, ANOMALOUS reset code (`0x00200a00`, no sync-flood bit)

### Context
Box booted 09:08:28 local (boot -3; the preceding reset was `0x00200800` — a clean ACPI power cycle).
The crash-capture started 09:10:42 local (`/buffer/powertrip/*-20260906-161042*`, UTC in filename) but
**stopped writing at 09:17** — it was not running for the actual serve launch. The
`qwen38-flash-serve` container started at **09:58:10** (dockerd journal, `sbJoin ... ep=qwen38-flash-serve`).

### Event
Boot -3's journal **ends abruptly at 10:13:18** (last line routine tailscaled noise) — **~15 min after
serve start**, squarely in the weight-load window characteristic of Instances 2/4/5. No shutdown
sequence, no EDAC/MCE lines. The box sat powered off ~5 h 40 min (next boot 15:53:52).

### Evidence (how sourced)
- `journalctl --list-boots` → boot -3: Sep 6 09:08:28 → 10:13:18.
- `journalctl -b -3 | tail` → hard cutoff at 10:13:18, no clean-shutdown trail.
- `/var/log/kern.log` next-boot line (Sep 06 15:54:06): `Previous system reset reason [0x00200a00]:
  internal CPU thermal limit was tripped` + `ACPI power state transition occurred`.
- `/buffer/powertrip` file mtimes → capture last wrote 09:17; **zero telemetry coverage of this trip**.

### Classification / conclusion
- **Register-level anomaly:** `0x00200a00` = thermal limit + ACPI power transition, **without the `0x08`
  sync-flood bit** — the only trip in this catalog lacking the memory signature. Two readings:
  1. a genuine thermal / power-delivery event during the load burst (no UC error occurred), or
  2. the same DIMM sync-flood where the platform failed to latch the `0x08` bit before reset.
- Undecidable post-hoc: no capture coverage, no persisted EDAC. The **timing fingerprint** (hard cutoff
  ~15 min into serve, mid weight-load) matches the Instances 2/4/5 fault class; provisionally grouped
  with the same root cause, flagged for the code difference.

### Follow-up
- Do **not** treat a missing `0x08` bit as "not memory" — the Huananzhi reset-reason register is a
  heuristic APML read; bit-latching across a hard reset is not guaranteed.
- Repeat of the Instance-5 lesson: **capture must be armed at serve start**, not started and stopped.

---

## Instance 7 — 2026-09-06 (16:24 local) — POWER TRIP during model load (classic `0x08000a00`, no capture)

### Context
Box booted 15:53:52 local (boot -2). `qwen38-flash-serve` started at **16:11:16** (dockerd
`sbJoin ... ep=qwen38-flash-serve`). Crash-capture **not running** — the next capture run began
17:31:18, after the post-trip reboot.

### Event
Boot -2's journal **cuts off mid-line at 16:24:34** (routine tailscaled NetInfo entry) — **~13 min after
serve start**, again mid weight-load.

### Reset / evidence
```
Sep 06 16:49:46  x86/amd: Previous system reset reason [0x08000a00]:
                 internal CPU thermal limit was tripped
                 an uncorrected error caused a data fabric sync flood event
```
- Sourced from `/var/log/kern.log` at the next boot (boot -1, 16:49); `journalctl -b -2 | tail` shows
  the abrupt cutoff.
- `journalctl -k -b -2` grep for EDAC/MCE → **empty**: per-CE EDAC lines again not persisted pre-reset
  (same capture gap documented in Instance 5).
- CE ledger carried into later boots (`ras-mc-ctl --summary`): 2749 total CEs — **channel#6 (G/MM4) =
  2301 (1367 + 934 across two chip-select ranks, ~84%)**, channel#7 (H/MM2) = 315, channel#5 (F/MM6) = 110.
  Fault concentration unchanged.

### Classification / conclusion
- **Same root cause as Instances 1, 2, 4, 5**: marginal DIMM on channel G (slot MM4); CE→UC→sync-flood,
  triggered by the both-GPU weight streaming + PLE host-RAM traffic.
- **Intermittency reconfirmed:** after this trip, the same serve recipe came up at 16:49 and served
  **16 h 42 min clean** (full load, HTTP 200, native 262k context), ending in a *clean* shutdown
  Sep 7 09:34 (reset reason `0x00200800` — not a trip). The hardware fault is probabilistic per load,
  not deterministic.

### Follow-up
- Gating fix unchanged: reseat/replace **MM4**, memtest channel G, downclock RAM.
- Corollary for future triage: a long clean run afterwards is **not** evidence the DIMM is fixed —
  trips and 16 h clean serves alternate on the identical recipe.

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
  Instances 1, 2, 4, 5, and 7 all sharing the identical `0x08000a00` signature (Instance 6 cut off at the
  same stage but logged `0x00200a00` without the sync-flood bit — register ambiguity, see its entry).
- **Sep 6 trips (Instances 6–7) had ZERO capture coverage** — capture was stopped (6) or not yet started
  (7) at trip time. Additionally, the capture's `edac-*.csv` output is **0 bytes in every run ever**: the
  EDAC logging inside the crash-capture is broken and must be fixed before it can serve as a
  CE-rate-based pre-trip recorder.

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

---

## Instance 8 — 2026-09-07 (~16:54 local) — GLM-5.3-Flash load, SOFTWARE reset (distinct class)

### Context
Trying a **new model** (`zai-org/GLM-5.3-Flash`) via `vllm/vllm-openai:glm53-flash`, container
`glm53-flash-serve`. Config: TP2, `--load-format dummy`, `--cpu-offload-gb 280`, `--max-model-len 8192`,
`--max-num-seqs 1`, `--enforce-eager`, `--disable-custom-all-reduce`, `--max-parallel-loading-workers 1`.
Boot was 15:30:48 → 16:54:31.

### Event / reset reason
```
x86/amd: Previous system reset reason [0x00080a00]: internal CPU thermal limit was tripped
x86/amd: Previous system reset reason [0x00080a00]: software wrote 0x6 to reset control register 0xCF9
```
**Not** the `0x08000a00` sync-flood bit. Decoded:
- bit19 `0x00080000` = **software wrote 0x6 to reset control register 0xCF9** (software-initiated warm reset)
- bit9  `0x00000200` = CPU internal thermal limit latched bit
- (no bit27 — **no** "uncorrected error / data-fabric sync flood")

### Where it died (docker log tail)
```
... Total CPU offloaded parameters: 148.55
... Using MoEPrepareAndFinalizeNoDPEPModular      <- last line (fp8-MoE finalize)
```
Exited (255) at the tail of load, before serving.

### Per-boot ECC attribution (window-based, NOT cumulative)
Using the rasdaemon DB, boot -1 (this trip) had **only 68 CEs, all on channel#6 (channel G / MM4)**.
No channel-0/5/7 CEs in that boot window. This is much smaller than the lifetime 1367/1002 channel-6
totals, confirming those are historic accumulation, not this trip.

### Classification
- **Reset class differs** from Instances 1/2/4/5/7 (`0x08000a00` sync flood). This reset was triggered by
  **software writing the reset-control register** (0xCF9), not an interceptor-uncorrectable memory error.
- The 68 fresh channel-6 CEs are consistent with ongoing marginal DIMM activity, but **do not by
  themselves** prove the firmware/software reset was a memory sync-flood.
- No thermal runaway (tctl ~48.5°C, GPU 66/69°C, GPU ~115 W) despite the latched "thermal" bit.
- Fair reading: this was likely a software/hardware-initiated warm reset during fp8-MoE finalize of a new
  (Not-yet-fully-supported) model, coinciding with a small amount of channel-6 CE activity. Do **not**
  treat it as the classic memory-fault fingerprint.

### Follow-up
- Re-test GLM-5.3-Flash with capture armed; watch `edac-ce-watch.sh` for a real pre-trip CE ramp.
- Confirm whether the 0xCF9 reset is a vLLM/kernel panic-then-reboot path or a BMC/watchdog path.
- Keep the DIMM on channel G / MM4 as a real (but long-running, not per-trip) suspect pending memtest/RMA.

**2026-09-08 update:** `recipe/serve-glm-53-flash.sh` was found to have a real capture-coverage gap
matching Instances 3/6/7: `powertrip-capture`'s dmesg session went quiet ~27 min into this run, ~2h39m
before the actual death, so the fp8-MoE-finalize crash itself has no kernel log. Fixed this session:
(1) `arm_safety()` now checks klog mtime and warns if capture has gone stale, not just whether the
container is "running"; (2) the launcher also gained the NUMA/NCCL memory-less-node guard from the
Qwen recipe (moot for *this* instance — all 8 DIMMs were populated at the time — but load-bearing for
any rerun today, since the box is now in the reduced-DIMM state with GPU0 on a memory-less node); (3) a
`--dummy`-at-TP2 offload-sizing bug (was requesting the TP1 default of 280 GB/worker instead of 150) was
also fixed. `ipmitool sel list`/`mc watchdog get` still not run (needs interactive sudo) — the
BMC-vs-NMI-watchdog question in the first bullet above is still open. See
`recipe/GLM-53-FLASH-RECIPE.md` §0 for the full writeup.

---

## Instance 9 — 2026-09-08 (~22:46 local) — GLM-5.3-Flash `--dummy` rerun, clean host-wide OOM (NOT a power trip)

### Context
Rerun of the GLM-5.3-Flash TP2 `--dummy` smoke test (see Instance 8, `recipe/GLM-53-FLASH-RECIPE.md` §0),
this time with three fixes in place: the NUMA/NCCL memory-less-node guard (ported from the Qwen
recipe), correct TP2 offload scaling (150 GB/worker, not the TP1 default of 280), and a capture-liveness
check in `arm_safety()`. Also different from Instance 8 in one major way: the box is now in the
**interim isolation-testing RAM config** (~499 GiB, only NUMA nodes 0–1 populated, nodes 2–3
memory-less — Instance 8 ran under the still-full 8-DIMM/1-TiB config). Pre-flight was clean: Qwen
stopped, GPUs free, 362 GiB RAM available, capture and EDAC watch both running.

### Event
- NCCL init passed cleanly under the reduced-NUMA topology — **the new guard worked**:
  `gpu 00000000:01:00.0 -> numa node 3 (0 MB local memory)` detected, `NCCL_CUMEM_HOST_ENABLE=0` forced
  automatically, no segfault.
- Reached the same point as Instance 8 (`FLASHINFER_MLA_SPARSE_SM120` sparse-MLA backend, `DEEPGEMM`
  MoE backend selected) — then died **earlier** than Instance 8, during engine-core distributed startup
  / CPU-offload buffer allocation (`UVAOffloader`), ~13 min after launch. Never reached Instance 8's
  fp8-MoE-finalize stage.
- Container exited: `Status=exited ExitCode=1 OOMKilled=true`. **No host reset** — `uptime -s` unchanged
  throughout.

### Root cause (fully evidenced, no gaps this time)
`journalctl -k --since ... --until ...` for the exact crash window shows a **genuine, severe,
global** Linux OOM event (`cpuset=... global_oom`), not a container-scoped cgroup limit:
```
Sep 08 22:46:40  dcgm-exporter invoked oom-killer ...
Sep 08 22:46:41  Out of memory: Killed process 8301 (dbus-daemon) ...
Sep 08 22:46:41  Out of memory: Killed process 7847 (systemd) ...        [user session systemd]
Sep 08 22:46:41  Out of memory: Killed process 7851 ((sd-pam)) ...
Sep 08 22:46:41  pt_nccl_watchdg invoked oom-killer ...
Sep 08 22:46:41  Out of memory: Killed process 3377512 (VLLM::Worker_TP) total-vm:753559988kB ...
Sep 08 22:46:41  Cannot map memory with base addr 0x796eb2000000 and size of 0x100000 pages
Sep 08 22:46:41  NVRM: failed to copy out ioctl data
```
The kernel killed **unrelated host processes** (dbus, a user systemd session) before finally killing
the vLLM worker — proof this was real, severe memory pressure, not a small overshoot of a single
cgroup limit. The two `VLLM::Worker_TP` processes showed `total_vm` ~718 GB each (mostly virtual
address space — UVA/pinned-memory reservations for the offloaded experts) with hundreds of MB of real
page-table overhead each, and the NVIDIA driver itself started failing memory-mapping ioctls
(`Cannot map memory...`, `NVRM: failed to copy out ioctl data`) — a sign the box was critically low on
free host RAM, not just "a bit over budget."

**Likely mechanism:** GPU0's local NUMA node (3) is memory-less right now, so its ~150 GB CPU-offload
buffer cannot be allocated locally — it must land on a remote node, piling onto nodes 0/1 (which
together hold ~490 GiB) alongside GPU1's own ~150 GB offload (GPU1 sits on node 0). With base host
usage (~137 GiB) and page-table/shared-memory overhead for the huge UVA mappings on top, the nominal
"150×2=300 GiB fits in 362 GiB available" budget did not leave enough real headroom once NUMA locality
skewed the actual placement. This is exactly the risk flagged (as a prediction) in
`recipe/GLM-53-FLASH-RECIPE.md` §2's RAM caveat, now **confirmed empirically**.

### What worked (validating this session's earlier fixes)
- **Capture stayed alive the entire run** — `klog-20260908-151528.log` was still being written *after*
  the crash (23:01, vs. crash at 22:46), unlike Instance 8 where capture went dark ~27 min in. The
  liveness check + this simply being a shorter-lived failure both helped; either way, the gap that hid
  Instance 8's cause did **not** recur.
- **No EDAC/CE ramp** (`edac-ce-watch.sh --status`: empty) — this failure has nothing to do with the
  channel-G/MM4 DIMM.
- **NUMA/NCCL guard worked as designed** — no segfault at `ncclCommInitRank` despite GPU0's memory-less
  local node.

### Classification / conclusion
- **Not a power trip, not a DIMM issue.** A capacity problem: the current reduced-DIMM/reduced-NUMA
  interim config doesn't have enough *effectively-placed* host RAM for this model's CPU-offload
  footprint, even though the raw `free -h` total looked plausible.
- Distinct from, and did not reach, Instance 8's still-unresolved fp8-MoE-finalize `0xCF9` reset — that
  question remains open and untested by this run.

### Follow-up
- Don't retry `--dummy`/`--serve` as-is under the current RAM config. Options: (a) wait for the DIMM
  reinstall (restores nodes 2/3, giving GPU0 local memory and ~896 GiB+ total); (b) explicitly NUMA-bind
  the container's offload to nodes 0/1 and lower `CPU_OFFLOAD_GB` well below 150/worker to leave real
  headroom (e.g. ~100/worker) for page-table/shmem overhead; (c) try TP1 instead of TP2 (single 280 GB
  offload budget, still on the memory-less-node GPU though, so may not help without also NUMA-binding).
- The two newly-noticed, previously-undocumented Sep 8 00:06/01:41 CF9 resets (see Index table) are
  still unexplained — no capture coverage, cause unknown. Worth a dedicated look before assuming they're
  related to either GLM attempt.

---

## Corrected findings (2026-09-07) — cumulative vs per-trip ECC attribution

**Problem:** the earlier analysis (Instances 1–7) leaned on `ras-mc-ctl --summary` / `rasdaemon`
**lifetime cumulative** counts to attribute each trip to "channel G / MM4". Those totals are *not*
per-trip evidence.

**What `ras-mc-ctl --summary` actually reports:** cumulative corrected-ECC events **since the rasdaemon
DB was created** (earliest 2026-08-21). It is repeated verbatim on every boot and in every
`edac-ce-watch` alert, which made it *appear* that each trip produced thousands of new CEs. It does not.

**Per-day breakdown of channel#6 CEs (from rasdaemon DB timestamps):**

| Date | channel#6 CEs |
|---|---|
| 2026-08-23 | 6 |
| 2026-08-27 | 2 |
| **2026-09-03** | **1100** |
| **2026-09-04** | **1186** |
| 2026-09-05 | 5 |
| 2026-09-06 | 2 |
| 2026-09-07 | 68 |

- **~2286 of 2369 channel-6 CEs (96%) occurred on just two days — Sep 3–4** (during a `glm52-sglang`
  run that became a zombie container). That is a single historic burst, not a per-trip signature.
- The individual **trip boots** carried little fresh CE:
  - Sep 6 sync-flood trip (boot -3): **0** fresh channel-6 CEs in that boot.
  - Sep 7 GLM trip (boot -1): **68** fresh channel-6 CEs.

**Implication:** the huge channel-6 totals are **historic residue**, not proof each trip was a memory
sync-flood. The DIMM on channel G / MM4 is a genuine long-run suspect (highest historical CE), but each
trip must be attributed on its **own boot-window CE count + its own reset-reason code**, not on the
lifetime total. Only reset code `0x08000a00` (bit27) is direct evidence of an uncorrectable memory
error; `0x00080a00` (bit19) is a software 0xCF9 reset; `0x00200a00`/`0x00200800` (bit21) are ACPI/power.

**Tooling added for correct accounting:**
- `recipe/ecc-per-window.sh` — per-channel CE for a date window / boot / lifetime (rasdaemon DB).
- `recipe/reset-reason-decoder.sh` — decodes the reset-reason bits (`--history`, or a 0xNN code).
- `recipe/edac-ce-watch.sh` — rewritten to alert on **per-interval new-CE deltas**, not cumulative totals.
- `recipe/diag-dimm-fault.sh` — now separates cumulative `[1]` from per-boot `[6]`, and decodes reset
  reasons in `[5]`.
