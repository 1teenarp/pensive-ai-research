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

### A3. Keep debugging GLM-5.2 NVFP4 — PENDING
Get past the SGLang NVFP4 fused-MoE loader shape mismatch (`3072 vs 6144` in `_load_w13`) and/or find a working vLLM sparse-MLA backend on Blackwell.
- Flags to try: `--fp4-gemm-runner-backend` variants, disable fused MoE load, `--moe-dense-tp-size`, `--ep-size`/`--tp-size` combos, alternate `--quantization` (`nvidia_fp4`/`modelopt`), `--weight-loader-disable-mmap`.
- Note: vLLM has no sparse+MLA attention backend for sm_120 here (`compute capability not supported`); vendor-proven path is SGLang `dev-glm52-nvfp4`.
- Bleeding-edge checkpoint; deprioritize if it blocks.

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
