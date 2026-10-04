# PROJECT TODOs

Tracked tasks for this build. Primary groups: **A. Model serving / RAM+VRAM offload**, and **B. Stability / power-trip (currently stashed)**.
Related live detail lives in `README-ramoffload-research.md` (offload research/empirical) and `power-trip-diagnosis.md` (power fault).

---

## A. Model serving — RAM+VRAM offload

### A1. Prove offload on a supported model — ✅ DONE (2026-09-04)
**Result:** Qwen3.5-397B-A17B-NVFP4 served end-to-end on 1 GPU via
`vllm serve --tensor-parallel-size 1 --cpu-offload-gb 200 --quantization modelopt_fp4` (vLLM 0.27.1).
`/v1/models` up on `:8081`; coherent text generated. **~1 token/s**, bottleneck = RAM→GPU PCIe H2D ~10 GB/s (see README §11).
- Working flags: `--cpu-offload-gb 200 --gpu-memory-utilization 0.80 --max-model-len 2048 --max-num-seqs 1 --enforce-eager` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- (OOM'd at `--cpu-offload-gb 170` because resident weights + KV left no headroom.)

### A2. Diagnose the 2-GPU NCCL hang — ✅ ROOT-CAUSED + FIXED (2026-09-05)
**Root cause:** the two GPUs sit on **different NUMA nodes** (GPU0→NUMA3, GPU1→NUMA0) with only a
`SYS` (PCIe + SMP/QPI) connection and **no NVLink**. NCCL's topology detection selects **`P2P/CUMEM`
direct channels**, which hang on this cross-NUMA/no-NVLink link — `init_process_group` succeeds but
`all_reduce` never completes (watchdog timeout, threads 100%).
- **Test:** minimal `torch.distributed.init_process_group` + `all_reduce` on 2 GPUs with `NCCL_DEBUG=INFO`.
  Confirmed: default → `Channel 00/0 : 0->1 via P2P/CUMEM` then `all_reduce` hangs.
- **Fix (verified):** `NCCL_P2P_DISABLE=1` → NCCL falls back to the **Socket** path over the host
  interconnect; `all_reduce OK` on both ranks. (Migration path: GPUs on same NUMA node / NVLink/SLI.)
- Result: used TP2 successfully on `Qwen3.8-Flash-Next-NVFP4` (got past distributed init to weight-load).
  See `power-trip-instances.md` Instance 2 (the run then died on the memory power-trip, not NCCL).
- Confirmed also: SymmMemCommunicator unavailable (sm_120/Blackwell); vLLM falls back to CUSTOM/PYNCCL all-reduce.
- **Deeper mechanism found 2026-09-24** (KNOWLEDGE §1, `evidence/p2p-probe-2026-09-24/`): the hang is
  a *symptom*, not the fault. Peer writes on this box are **silently discarded** — `canAccessPeer`
  returns True, `cudaMemcpyPeer` returns success, and the data arrives as all zeros. NCCL hangs only
  because the flag it spins on is one of those discarded writes; it is the **loud** failure mode.
  That makes `NCCL_P2P_DISABLE=1` a **correctness** guard, not a perf workaround — never lift it to
  test throughput. The migration path noted above is now tracked as A4.5.

### A3. Keep debugging GLM-5.2 NVFP4 — PENDING
Get past the SGLang NVFP4 fused-MoE loader shape mismatch (`3072 vs 6144` in `_load_w13`) and/or find a working vLLM sparse-MLA backend on Blackwell.
- Flags to try: `--fp4-gemm-runner-backend` variants, disable fused MoE load, `--moe-dense-tp-size`, `--ep-size`/`--tp-size` combos, alternate `--quantization` (`nvidia_fp4`/`modelopt`), `--weight-loader-disable-mmap`.
- Note: vLLM has no sparse+MLA attention backend for sm_120 here (`compute capability not supported`); vendor-proven path is SGLang `dev-glm52-nvfp4`.
- Bleeding-edge checkpoint; deprioritize if it blocks.

### A4. Decode-throughput optimization — Qwen3.8-Flash-Next-FP8 (Recipe C) — OPENED 2026-09-16
Live baseline (server as configured 2026-09-16): ~20.8 tok/s @ seqs=1, MTP accept 55.9 % (~3.2 tok/round),
bottleneck = TP2 host-bounce collectives (~104 blocking all-reduces/round; 99 % util @ ~100 W spin —
KNOWLEDGE §1). Est. gains below are estimates (P10) — measure each. Rules: one variable per attempt (P2),
fork the launcher (`serve-qwen38-flash-next-fp8-<variant>.sh`) instead of mutating the proven one,
`--check`/`--dummy` first (P7), RUN-LOG entry per attempt (P9).

- **A4.1 · Concurrency at shorter ctx — do first (env-only).** `MAX_MODEL_LEN=65536 MAX_NUM_SEQS=4`
  on a restart (needs a window; current server stays as-is until then). Expected 2–4× *aggregate*
  via P-J amortization; KV fits (4×65k ≪ 580k pool). Sweep concurrency 1/2/4 with representative
  agent prompts; report per-client latency alongside aggregate. **Gate first:** confirm tailnet
  clients tolerate 65k max ctx (target-workload question — ask user).
- **A4.2 · NCCL socket-path tuning — second attempt.** `NCCL_NSOCKS_PERTHREAD` /
  `NCCL_SOCKET_NTHREADS` raises + pin each worker to its GPU-local NUMA node; est. 10–30 % (inferred,
  untested). Re-verify topology after RMA DIMMs land (node 3 memory interacts).
- **A4.3 · Newer-vLLM probe.** Does a current build get CUSTOM all-reduce or symm-mem working on
  sm_120 (both fail in today's build — §1)? Retest `SPEC_TOKENS=5` too (§3: ceiling was image-specific).
  `--check` → `--dummy` first; retire the whole path fast if the selector already says no (R-015 rule).
- **A4.4 · `iommu=pt` host change — needs USER decision + reboot window** (§1). Demoted
  2026-09-24, then **partially reinstated the same day** by the data-integrity probe (KNOWLEDGE §1 +
  `evidence/p2p-probe-2026-09-24/`). The demotion assumed P2P was *structurally* impossible, but that
  was **inferred from topology and never tested** — `iommu=pt` has still never been tried on this
  host. What the probe did establish: peer writes are **silently discarded** (all zeros, no error at
  any layer), AMD-Vi is in `Translated` mode, and the **root cause was not reached** (ACS / AtomicOp
  bits need root). So this is **untested lever (a)**, not ruled out.
  - **Cheap pre-step, no reboot:** `sudo lspci -vvv -s 00:01.1 | grep -E 'ACSCap|ACSCtl|AtomicOp'`
    (and `c0:03.1`). An `ACSCtl: … RequestRedirect+` / `UpstreamForwarding+` would explain the dropped
    writes and is what justifies booking the reboot window.
  - Then: GRUB flag → reboot → `journalctl -k | grep 'Default domain type'` (expect `Passthrough`) →
    re-run `evidence/p2p-probe-2026-09-24/` and **gate on payload, not bandwidth**.
  - **Ceiling if it works:** fallback measures **8.0 GB/s** busbw; a working P2P path is bounded by
    the Gen3 x16 link at ~13–14 GB/s → **~1.7× on comm bandwidth**, and end-to-end decode gain is a
    *fraction* of that since comm is not the whole round. So the old 1.5–2.5× estimate was optimistic
    as a decode figure but not absurd as a comm one — treat it as ~1.7× comm, unquantified decode.
    ⚠️ Do **not** cite the probe's 13.89 GB/s peer-copy figure as evidence of a working path: that
    measurement was dropped writes moving zero bytes (KNOWLEDGE §1).

- **A4.5 · Re-slot both GPUs onto one root complex — needs USER decision + physical access.**
  `[new 2026-09-24]` Untested lever (b). Moving both GPUs under one root complex would remove the
  Infinity-Fabric crossing entirely. **⚠️ CORRECTED 2026-09-25 — this is not a free-slot move.** The
  first draft of this item claimed `0000:c0` "exposes five x16 root ports (`c1`–`c5`)"; it conflated
  *PCIe root ports* with *available expansion slots*. All five are populated by onboard devices:
  `c1`/`c2` = Intel I226-V NICs, `c3`/`c4` = NVMe, `c5` = GPU1. Root complex `0000:00` likewise
  exposes only `00:01.1` (GPU0). **So there may be no pair of physical x16 slots sharing a root
  complex on this board at all** — that is a board-layout question, not a configuration one.
  **Do first (free, no downtime):** read the H12D-8D manual / block diagram and establish whether any
  two x16 slots share a root complex. If not, this lever is dead and lever (a) + A4.6 are all that
  remain. If yes: move → `nvidia-smi topo -m` (expect `PHB`/`NODE` instead of `SYS`) → re-run the
  probe, gating on payload.

- **A4.6 · Lift the PCIe Gen3 cap — BIOS, needs USER decision + reboot window.** `[measured
  2026-09-24]` Both GPU root ports advertise `max_link_speed` **8.0 GT/s** while the cards advertise
  32.0 GT/s and EPYC Milan is Gen4-capable — platform-level (BIOS setting or slot/riser wiring),
  not GPU silicon (KNOWLEDGE §1). This throttles the host-bounced collective path **and** H2D offload
  — both things every TP2 decode round depends on — so it **pays off even if P2P is never revived**,
  which makes it the best expected-value of the three host changes. Look for a "PCIe Link Speed" /
  "Gen Speed" option on the GPU slots; measure `nccl-tests` all_reduce_perf + an H2D benchmark
  before/after. Resizable BAR is already fully on (BAR1 = 64 GB both cards) — not a factor.
- **Ruled out — do not re-attempt:** TP2→PP2 (vLLM PLE guard, KNOWLEDGE §2; and batch-1 bubble
  kills it anyway, P-I). CPU side is not a bottleneck (~3 cores of 56).

### A6. Serve GLM-5.3-Flash (zai FP8) via KTransformers CPU-experts — ✅ STAGE 2 SERVING @ 5.81 tok/s, ctx 501025, long-prompt path proven (R-031, 2026-10-01)
**Why:** R-020's 2.4–2.5 tok/s is structurally comm-bound (TP2 host-bounce collectives + per-token
H2D expert stream). KTransformers moves the 305 GB of routed experts into host RAM where they
**execute on the CPU** (kt-kernel, AVX2 variant — this EPYC has no AVX-512/AMX); GPU0 does
attention/dense/shared; TP1 → zero all-reduces. P10 estimate: 3–12 tok/s.
- **Done:** image `pensive/glm53-kt:latest` (kt-kernel + sglang-kt 0.7.0.post4 from official PyPI
  wheels, `recipe/Dockerfile.glm53-kt`); launcher `recipe/serve-glm-53-flash-kt.sh`
  (`--check` → `--smoke` → `--serve`, port 8093); recipe `GLM-53-FLASH-KT-RECIPE.md`;
  `--check` PASSED (R-022).
  - **AVX2 FP8 expert path: RESOLVED** — all 42 layers create `AVX2_FP8_MOE_TP 0..3` pools
    (R-024/025/026); the tutorial's "AVX-512" line was conservative.
  - **sm_120 DSA/KDA attention: RESOLVED at init** — sglang-kt logs `GPU profile=blackwell_fp8`,
    `NSA dispatcher=trtllm … on SM86/SM89/SM120`, `arch=compute_120a`.
  - **Image toolchain: RESOLVED** — the `-base` image lacked `cc` (R-024), `python3-dev`/`Python.h`
    (R-025) and `nvcc` (R-026); each died at CUDA-graph capture JIT. Now `devel` base +
    `build-essential` + `python3-dev`.
- **Done (R-027):** `--smoke` passed — READY ~27 min, correctness probes pass (arithmetic + factual),
  **5.81 tok/s steady** (2.4× vLLM baseline).
- **Done (R-028):** concurrency sweep (N=1/2/4) — per-request 5.81→3.23→1.76 tok/s, aggregate
  5.83→6.48→7.05 tok/s. Sublinear: shared DRAM expert-stream bound; concurrency not the lever.
  Container left at `MAX_RUNNING_REQUESTS=4`.
- **Done (R-029):** Stage 2 `--serve` at CTX=501025 (tutorial-validated full context; ONE variable
  vs R-028). Fixed user's client `400 Bad Request` (was the 8192 ctx cap on max_tokens). READY
  ~21 min; probes pass; `max_model_len` 501025 confirmed.
- **Done (R-030→R-031):** A user's "very long prompt" crashed the R-029 server (exit 137, host fine)
  with `RuntimeError: KT shared-memory NUMA setup failed`. Root cause was NOT NUMA — the
  layerwise-prefill CPU-expert path lazily dlopens `libnuma.so.1`, which the image lacked.
  Fix: `+libnuma1` in `Dockerfile.glm53-kt` (image now `3cf104f7ddcb`) + a 5 s in-image
  `ctypes.CDLL("libnuma.so.1")` gate in `--check`. Regression-tested the exact crash path:
  3442-token prompt → HTTP 200, clean prefill+decode (prefill ~10.6 tok/s first-use). A second 400
  class (client `reasoning_effort:"max"`) documented in KNOWLEDGE §5 — client must use low/medium/high.
- **Open:** GPU-expert residency for short prompts (raise `KT_GPU_PREFILL_THRESHOLD`), threadpool
  sweep (`KT_THREADPOOL` 4→8/16, `KT_CPUINFER` 56→48), MOE_INT8 A/B.
- **Fallbacks:** `KT_METHOD=MOE_INT8` (AMD-BLIS source build + `convert_cpu_weights.py
  --quant-method moe_int8`; P12-clean) → `KT_METHOD=LLAMAFILE` + unsloth GGUF (**third-party quant —
  explicit user OK required, P12**).
- **P12 exception on record:** user-directed 2026-09-30 (third-party runtime, official wheels, our
  image, vendor weights). Provenance in recipe §0 + KNOWLEDGE.md §2.

### A5. Serve GLM-5.3-Flash-NVFP4 on the ported NoPE image — ✅ SERVED 2026-09-24 (RUN-LOG R-020)
Blocker R-014/R-015 lifted in software: `pensive/glm53-flash:nope-sm120-617d0cc` (vendor-fork base +
Apache-2.0 `glm53_sparse_mla` plugin, RUN-LOG R-016, recipe §0). `--check` green. Launcher defaults
set to the shipped code's limits: `KV_CACHE_DTYPE=bfloat16` (never `auto` — R-017), `BLOCK_SIZE=256`
(R-018), `ENFORCE_EAGER=1` (CG `NEVER`), plugin gate on. Sequence executed: `--dummy` **PASSED**
(R-019, READY 1537 s) → `--serve` **PASSED** (R-020, READY 2637 s; correctness probe: coherent
output, no silent garbage) → measured steady **~2.4–2.5 tok/s, comm-bound** (TP2 host-bounce +
eager + no MTP + 32 GB/wkr offload).
- **Open / next:** stage-3 increments — **MTP first** (`SPEC_CONFIG` `num_speculative_tokens=2`,
  expect ~2.5–3× on a comm-bound box per P-J), then a `VLLM_TORCH_PROFILER_DIR` round to split one
  decode round (NCCL-wait vs kernel vs H2D-gather); fp8-KV and CUDA-graph support only after the
  plugin's README-vs-code claims are tested, one variable each.

---

## B. Stability / power-trip — STASHED (hardware debug)

Root cause of the sudden power-off: **uncorrectable memory error → data-fabric sync-flood reset + CPU internal thermal-limit trip** (reset reason `0x08000a00`). ECC errors concentrated on **channel G / slot MM4** (2,294 of 2,741), with MM2 (H) and MM6 (F) secondary. See `power-trip-diagnosis.md` + `power-debug-collect.sh`.

### B1. Fix the failing memory (slot MM4) — TODO (needs physical/BIOS access)
- Reseat / swap / replace the DIMM in **slot MM4 (channel G)**; also try **MM2 (H)** and **MM6 (F)**.
- **Downclock memory** (3200 → 2933/2666) — 8× Micron 128 GB DDR4-3200 8-rank ECC on a single socket is aggressive and is a plausible cause of the CECC/UE.
- Run **memtest86** on channel G to confirm.
- Verify: no new corrected-ECC on channel G, no `uncorrected error / sync flood` reset reason.

### B2. Capture temperature telemetry next time — ✅ LIVE CAPTURE IN PLACE (2026-09-05)
Now have an on-box, crash-safe telemetry logger: `powertrip-capture.sh` + `powertrip-capture:local`
image (launcher `run-powertrip-capture.sh`). Runs privileged, writes every ~1 s to persistent
`/buffer/powertrip/` (survives reboot): CPU Tctl+8 chiplets, RAPL package/core watts, freq, load, mem,
both GPUs (temp/power/mem/util/PCIe link/ECC), raw `dmesg` snapshots, EDAC table, DIMM map.
See `powertrip-capture-readme.md`. Plus `recipe/edac-ce-watch.sh` for a corrected-ECC pre-trip alert.
- **Manual by design:** capture + edac-ce-watch are `restart: no` and armed on-demand by
  `recipe/serve-qwen38-flash-next-nvfp4.sh --start` for a model-load/debug session. NOT auto-start on boot.

### B3. Investigate CPU thermal trip — TODO
- `thermald` won't run on this AMD EPYC (unsupported) → no OS thermal governor; the CPU relies on its internal SMU limit.
- Determine whether the thermal trip is a real overtemp under sustained all-core+GPU, a cooling shortfall, or a side-effect of the sync-flood reset; decide on CPU power/thermal cap or cooling improvements.

### B4. Investigate the software `0xCF9` reset class — TODO
Distinct failure signature first seen with GLM-5.3-Flash (see `power-trip-instances.md` Instance 8):
reset reason `0x00080a00`/`0x00080800` = **software wrote 0x6 to reset control register 0xCF9** +
thermal-limit bit — explicitly **not** the `0x08000a00` sync-flood/channel-G-DIMM signature that
Instances 1/2/4/5/7 share. First occurrence died at the fp8-MoE-finalize stage of a GLM-5.3-Flash
`--dummy` TP2 load; two more of the same signature turned up later, undocumented until found in this
session (2026-09-08 00:06 and 01:41 local), cause unknown, no capture coverage for any of the three.
- **Root cause open — two live hypotheses, neither confirmed:**
  1. Kernel panic → auto-reboot path (`nmi_watchdog` is enabled on this host — `cat
     /proc/sys/kernel/nmi_watchdog` = 1 — an NMI-watchdog-triggered panic during a CUDA-kernel hang
     would plausibly write `0xCF9` on its own reboot path).
  2. BMC/IPMI hardware watchdog. **Not yet checked** — `ipmitool sel list` / `ipmitool mc watchdog get`
     both need interactive sudo, which this session doesn't have non-interactively.
- **If it recurs:** immediately capture (before the next reboot overwrites state) —
  `journalctl -k -b -1` for a panic backtrace, `ipmitool sel list` for a BMC-logged watchdog event,
  and cross-check `powertrip-capture`'s klog file was actually still writing at the time (per the
  Instance 8/3/6/7 capture-gap pattern — verify with the klog-liveness check now built into
  `recipe/serve-glm-53-flash.sh`'s `arm_safety()` before trusting "no capture" as "nothing happened").
- **When to prioritize:** low urgency while GLM-5.3-Flash itself is blocked on RAM capacity (Instance 9);
  revisit once that's resolved and Stage 1 is retried, since that's the natural next chance to reproduce it.

### B3a. Reusable debug artifacts (already created)
- `power-debug-collect.sh` — safe, idle evidence collector (reset reason, per-channel ECC tally, CPU/GPU temps, freq/governor, package power, containers, DMI map).
- `power-trip-diagnosis.md` — findings, DIMM slot↔channel map, and the 5-step runbook (collect → DMI → ECC tally → controlled CPU/RAM/GPU isolation → re-run).
- Controlled tests done so far: CPU 8/32/112 threads all clean (≤60 °C, no ECC); memory `write64` 24×16 GB clean (no ECC) — short tests do NOT reproduce; the fault needed sustained (hours) heavy load.
