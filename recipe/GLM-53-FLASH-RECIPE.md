# GLM-5.3-Flash on "pensive" — research + staged recipe

Date: 2026-09-07, updated 2026-09-08. Status: **Stage 0 verified live; Stage 1 (`--dummy`) attempted
twice — once died to an unresolved software reset (Instance 8), once to a clean/diagnosed RAM-capacity
OOM under the current reduced-DIMM config (Instance 9). See §0. Stage 2 (`--serve`, real weights) NOT
yet run — do not attempt until the RAM issue is resolved.**
Model: `/trunk/ai/huggingface/models/zai-org/GLM-5.3-Flash` (305 GB on disk, 62 shards).
Launcher: `recipe/serve-glm-53-flash.sh` (staged: `--check` → `--dummy` → `--serve`).

---

## 0. Where this actually stands (read this before running anything)

- **Stage 0 (`--check`) — done, re-verified 2026-09-08.** `vllm/vllm-openai:glm53-flash` is pulled and
  present locally (28.8 GB). Re-running `bash recipe/serve-glm-53-flash.sh --check` confirms:
  `vllm=0.1.dev20051+g487ecf187 flashinfer=0.6.17 archs=['Glm5NextForCausalLM',
  'Glm5NextForConditionalGeneration', 'Glm5NextMTPModel']`. No pull or fallback-to-`:nightly` needed.
- **Stage 1 (`--dummy`) — already attempted once, 2026-09-07 16:37–19:43 local, TP2.** This is
  **Instance 8** in `power-trip-instances.md` — it wasn't captured in this doc's original "nothing
  launched" framing, which was wrong; correcting it here. What happened, in order:
  1. NCCL init passed (TP2, `NCCL_P2P_DISABLE=1`), sparse-MLA backend selected
     (`FLASHINFER_MLA_SPARSE_SM120`), FP8 MoE backend selected (`DEEPGEMM`), CPU-offload allocated
     cleanly (**148.55 GB/worker** — so the arch, runtime, and offload allocator all check out).
  2. It then died at the **fp8-MoE finalize stage** (`Using MoEPrepareAndFinalizeNoDPEPModular` is the
     last log line), container exit code 255, ~3 h after launch.
  3. The box's reset-reason register showed **`0x00080a00`: software wrote 0x6 to reset control
     register 0xCF9** — a **software-initiated warm reset**, explicitly **not** the `0x08000a00`
     sync-flood signature that Instances 1/2/4/5/7 share. Only 68 fresh channel-6 CEs in that boot
     window (small, not a DIMM-fault-scale burst).
  4. **Capture coverage gap:** `powertrip-capture`'s dmesg session went quiet ~27 min into the ~3 h run
     (last kernel line at kernel-uptime ~5027 s) — so the actual death, ~2h39m later, has **no kernel
     log**. We can't tell from existing evidence whether `0xCF9` came from a kernel panic→reboot path
     (plausible: `nmi_watchdog` is enabled on this host, `cat /proc/sys/kernel/nmi_watchdog` = 1 — an
     NMI-watchdog-triggered panic during a CUDA-kernel hang would plausibly manifest exactly as a
     software CF9 warm reset) or a BMC/IPMI watchdog (unchecked — `ipmitool` needs interactive sudo on
     this host, not yet run). **Open question, not resolved by this session.**
  5. **Important environment difference for any rerun:** this attempt ran while **all 8 DIMMs were still
     populated** (confirmed via the capture summary's DIMM map at attempt time) — i.e. before the
     DIMM-isolation testing pull. The box is now in the **reduced 4-DIMM state** (2 NUMA nodes
     memory-less, see §2 below); a rerun today exercises a materially different memory/NUMA topology
     than Instance 8 did, so Instance 8's "got past NCCL cleanly" result is not guaranteed to reproduce
     as-is without the fix in the next bullet.
- **Fixed this session (2026-09-08), in `recipe/serve-glm-53-flash.sh`:**
  1. **NUMA/NCCL memory-less-node guard ported from `serve-qwen38-flash-next-nvfp4.sh`.** GPU0's local
     NUMA node (3) is *currently* memory-less (DIMMs pulled for isolation testing). Without this guard,
     `ncclCommInitRank`'s `cuMemCreate` host-memory probe **segfaults** instead of failing gracefully —
     exactly the bug already found and fixed on the Qwen recipe. The GLM script didn't have this fix
     (it predates the DIMM pull); it does now, auto-detected per-launch, no flag needed.
  2. **`--dummy`'s CPU_OFFLOAD_GB scaling bug.** The script ran `--dummy` at TP2 but left
     `--cpu-offload-gb` at the TP1 default (280/worker = up to 560 GB requested across 2 workers),
     contradicting the recipe's own documented TP2 guidance (150/worker). Now auto-scales: TP2→150,
     TP1→280, unless explicitly overridden.
  3. **Capture-liveness check.** `arm_safety()` now checks the current klog file's mtime and warns if
     it's gone stale (>60 s), instead of only checking that the `powertrip-capture` container is
     running — this is the exact gap that left Instance 8's death window uncaptured, and the same
     pattern Instances 3/6/7 hit on the Qwen side.
- **Stage 1 rerun, 2026-09-08 22:46 local — this is Instance 9 in `power-trip-instances.md`.** Result:
  the NUMA/NCCL guard worked (no segfault, `NCCL_CUMEM_HOST_ENABLE=0` auto-applied), it reached the same
  sparse-MLA/MoE-backend point as Instance 8 — then died **earlier** than Instance 8 (during engine-core
  distributed startup / CPU-offload allocation, ~13 min in, never reaching fp8-MoE-finalize), to a
  **clean, kernel-logged, host-wide Linux OOM** (`OOMKilled=true`, no host reset, no EDAC ramp, capture
  stayed alive the whole time — confirming this session's fixes hold up). Root cause: the box's live RAM
  is in the reduced 4-DIMM/2-memory-less-node config (§2), and GPU0's local node being memory-less means
  its ~150 GB offload buffer has to land on the already-loaded nodes 0/1 — the nominal "300 GiB fits in
  362 GiB available" budget didn't leave real headroom once NUMA placement was accounted for. Full
  forensics in Instance 9.
- **Bottom line:** the model/runtime path is real (arch resolves, sparse-MLA initializes, offload
  allocator itself works) — Stage 0 is fully validated, Stage 1 has now failed **twice**, for two
  **different** reasons (Instance 8: unresolved software reset at fp8-MoE-finalize, under full RAM;
  Instance 9: clean OOM during offload setup, under reduced RAM). The current blocker is RAM capacity,
  not runtime support. **Do not retry as-is** — see §2 and §6 for what needs to change first (more RAM
  via the DIMM reinstall, or NUMA-bound/lower offload budgets) before attempting `--dummy` again, and
  Instance 8's fp8-MoE-finalize question is still completely untested and unresolved regardless.

---

## 1. What the model is (from config.json + model card + vLLM recipe)

| Property | Value |
|---|---|
| Architecture | `Glm5NextForConditionalGeneration` (`glm5_next`), **natively multimodal** (image+video vision tower) |
| Size | **320B total / 18B active** MoE — 45 layers + 1 MTP draft layer |
| Experts | **288 routed / 8 per token** + 1 shared, `first_k_dense_replace=3` (42 sparse MoE layers) |
| Attention | **Hybrid**: KDA linear-attn (34 layers, 64 heads) + **NoPE sparse-MLA / DSA** (11 layers, kv_lora_rank 512, index_topk 2048, kpool compress) |
| Extras | mHC hyper-connections (`hc_mult=4`, sinkhorn), MTP head |
| Quant | **native FP8** e4m3, block 128×128, dynamic act (1509 exempt (BF16) modules: embed, lm_head, norms, linear-attn proj, indexer, router gate) |
| Context | native **1,048,576** |
| Runtime req | **vLLM ≥ 0.29.0** + **FlashInfer ≥ 0.6.17** (sparse-MLA). Official target: TP4 on GB200. |
| Parsers | `--tool-call-parser glm47 --reasoning-parser glm45` (from vLLM recipe) |

## 2. Fit math on this box (2×72 GB = ~144 GB VRAM, RAM currently reduced — see caveat)

> **Live RAM caveat (2026-09-08):** the numbers below (and the original §2 estimate) assumed the
> nameplate **1 TiB / 8-DIMM** config. The box is currently in the **interim isolation-testing
> config: ~499 GiB / 4 DIMMs, only NUMA nodes 0–1 populated** (nodes 2–3 memory-less) — see
> `SYSTEM-SPEC.md`'s hardware-status banner. Live check just now: `free -h` → **362 GiB available**
> (137 GiB used, 128 GiB buff/cache). A TP1 load wanting `--cpu-offload-gb 280` or a TP2 load wanting
> 150×2=300 GiB both consume nearly all of that headroom, with much less margin than the original
> "~800 GiB avail after stopping Qwen" estimate assumed. **Re-check `free -h` and `numactl -H` right
> before any real (`--serve`) attempt** — don't trust this number if DIMMs have been reinstalled since.
> Also: GPU0's local NUMA node (3) is one of the memory-less ones, which is why the NCCL guard in §0 is
> now load-bearing, not optional.

- Weights ~306 GiB → **do NOT fit VRAM; CPU weight-offload is mandatory** (same path as
  A1/Qwen3.5-397B in `README-ramoffload-research.md`).
- Expert weight bytes per token (the offload stream): 42 sparse layers × 8 experts × 3×4096×2048 FP8
  ≈ **~8.5 GB/token**. Measured RAM→GPU H2D ≈ **~10 GB/s** (pinned, per GPU):
  | Config | Est. decode |
  |---|---|
  | TP1 + `--cpu-offload-gb ~280` | **~1 tok/s** (same regime as 397B-A17B's measured 1.0 tok/s) |
  | TP2 + per-worker `--cpu-offload-gb ~150` | **~2–2.5 tok/s** (2 PCIe lanes, experts split) |
  | + MTP spec decode (`num_speculative_tokens 2–5`) | **~4–8 tok/s realistic ceiling** |
- KV is **tiny** thanks to hybrid attention: KDA layers hold fixed-size conv/recurrent state; only 11
  DSA layers keep compressed MLA KV (~7 KB/token). 64–128k context is cheap; **weights, not KV, are
  the constraint**. FP8 KV is allowed on Blackwell per the vLLM recipe.
- RAM budget (after stopping the Qwen serve → ~800 GB avail): resident offloaded experts ~260–300 GB
  + load-time page-cache ≈ up to ~600 GB peak. Fits, but see §4.

## 3. Blockers found during research (status as of 2026-09-08)

1. **Runtime support — RESOLVED, verified live.** `vllm/vllm-openai:glm53-flash` is pulled locally and
   `--check` passes (`vllm=0.1.dev20051+g487ecf187 flashinfer=0.6.17`, all three `Glm5Next*` archs
   registered). This also answers the old GLM-5.2 "no Blackwell sparse-MLA backend" blocker: the
   GLM-5.3-Flash path supports NVIDIA **Hopper-and-newer** with FlashInfer ≥0.6.17 (Blackwell sm_120
   qualifies) — and Instance 8 independently confirmed `FLASHINFER_MLA_SPARSE_SM120` actually
   initializes on this hardware, not just that the image claims support.
2. **VRAM conflict** — the Qwen3.8 serve must be stopped before GLM loads (it owns all VRAM + ~200 GB
   host RAM). `require_gpus_free()` in the launcher enforces this automatically (refuses to run while
   `qwen38-flash-serve` is up).
3. Prior GLM-5.2 TP2 NCCL hangs — **root-caused/fixed**: `NCCL_P2P_DISABLE=1` is proven on this host
   (both the Qwen TP2 serve and Instance 8's GLM TP2 attempt got past NCCL init with it set). All
   stages set it. **New, separate NCCL risk as of today**: the memory-less-NUMA-node segfault — see §0
   and §2 — is now guarded by `check_numa_topology()` in the launcher.
4. **Unresolved: the fp8-MoE-finalize software reset (Instance 8, §0).** This is now the actual
   next blocker to clear, not runtime availability. Needs a rerun with capture verified alive
   throughout (now checked automatically by `arm_safety()`), and ideally an `ipmitool sel list` /
   `ipmitool mc watchdog get` check (needs interactive sudo — not run this session) to rule in/out a
   BMC watchdog vs. an NMI-watchdog kernel panic as the `0xCF9` trigger.
5. The BF16/noPE quirks: KV-cache layout envs (`VLLM_KV_CACHE_LAYOUT=HND`,
   `VLLM_SSM_CONV_STATE_LAYOUT=DS`) are only required for PD-disaggregation; single-instance can omit,
   but they're harmless to set (already set by the launcher).

## 4. Power-trip risk assessment (channel G / MM4)

> Reminder (see `power-trip-instances.md` "Corrected findings"): `ras-mc-ctl --summary` totals below
> are **lifetime cumulative since 2026-08-21**, not per-run evidence — 96% of channel-6's historic total
> landed in a single Sep 3–4 burst. Use `recipe/ecc-per-window.sh` or `edac-ce-watch.sh`'s delta output
> for "is this actively ramping right now," not the raw summary number.

Baseline as of 2026-09-08 (current boot, `ras-mc-ctl --summary`, still cumulative-since-2026-08-21):
**channel#6 = 1367 + 1002 (rows 0/1)**, channel#7 = 310, channel#5 = 46+64, channel#3 = 21, channel#0 = 1
— note channel G / MM4 is **currently physically removed** for isolation testing (see §2 caveat), so
these numbers should stop growing until it's reinstalled; if they do grow, something's wrong with the
removal.

- **Weight load** = 306 GB sustained host-RAM traffic — bigger than every past trip-trigger
  (Instances 2/4/5 tripped during ~124 GB loads). **High trip risk event.**
- **Serving with experts in RAM** = *continuous* ~8.5–17 GB/s random host-RAM reads for every
  generated token. Unlike Qwen (bursty load, then quiet serve), this model **never stops stressing
  DDR** while generating. Persistent CE accumulation → eventual UE is the realistic failure mode,
  not a one-shot load trip.
- Non-negotiable gates before ANY load attempt:
  1. `powertrip-capture` armed + `edac-ce-watch.sh` running (abort on CE escalation).
  2. GPU power cap 250 W.
  3. Serialized load: `--max-parallel-loading-workers 1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
  4. Qwen serve stopped, caches dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`).
  5. One variable per attempt; log CE deltas before/after each stage.
- Mitigation choice — load source: `/trunk/ai` ZFS (~175 MB/s HDD) is the **safest** (slow, flat,
  predictable fabric load); staging to `/buffer` (519 GB free, fits the 305 GB) speeds cold start but
  raises sustained read rate ~3–5 GB/s. **Recommend ZFS-direct for the first real load** (slower is
  safer here); stage to NVMe only after the DIMM is fixed/replaced.
- Honesty note: this model is the *worst-case workload* for a marginal DIMM. If channel G MM4 is due
  for RMA, **fix the DIMM first** — every attempt below is at elevated hard-reset risk.

## 5. Staged recipe (one variable at a time)

### Stage 0 — runtime pull + offline check (no GPU, no load) — zero fabric risk — ✅ DONE
```bash
docker pull vllm/vllm-openai:glm53-flash          # already present locally, verified 2026-09-08
bash recipe/serve-glm-53-flash.sh --check         # PASSED: vllm=0.1.dev20051+g487ecf187 flashinfer=0.6.17
```

### Stage 1 — dummy-weight TP2 smoke (no disk read, no real weights) — low risk — ⚠️ ATTEMPTED, DIED
Proves: arch registers, TP2 doesn't NCCL-hang (with `NCCL_P2P_DISABLE=1`), sparse-MLA backend
initializes on sm_120, offload allocator fits. Same isolation trick as the `--load-format dummy`
test in Instance notes. Requires Qwen serve **stopped**.
```bash
bash recipe/serve-glm-53-flash.sh --dummy         # load-format dummy, TP2, eager
```
**Already run once** (2026-09-07 = Instance 8, §0): NCCL/sparse-MLA/offload all proved out, but it
died at fp8-MoE finalize to an unresolved software `0xCF9` reset ~3h in, with a capture-coverage gap
over the actual death window. **Rerun this stage before touching Stage 2** — the launcher now has the
NUMA/NCCL guard, the TP2 offload-scaling fix, and the capture-liveness check (see §0), none of which
were in place for the first attempt. Watch `tail -f /var/tmp/serve-glm53.log` and keep an eye on
`docker logs -f powertrip-capture` / klog mtime for the full run, not just at launch — that's exactly
what went dark last time.

### Stage 2 — real load, TP1, conservative — first high-risk event
```bash
bash recipe/serve-glm-53-flash.sh --serve         # TP1, --cpu-offload-gb 280, ctx 8192, seqs 1, eager
```
Expect: load ~30–60 min from ZFS; serve HTTP 200; decode **~1 tok/s**. If it trips → back to §4
gates, nothing learned is lost (capture + CE deltas tell us the stage it died at).

### Stage 3 — increments (only after Stage 2 stable, one per session)
1. **TP2** (`TP=2`, per-worker `--cpu-offload-gb 150`) → ~2× decode.
2. **MTP spec decode** `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` → the
   single biggest lever for a bandwidth-bound offload (amortizes the 8.5 GB/token across several
   accepted tokens). Raise to 5 once stable.
3. **KV fp8** `--kv-cache-dtype fp8` + raise `--max-model-len` to 65536 (KV is cheap here).
4. CUDA graphs (drop `--enforce-eager`) last — biggest warmup burst.

### Fallbacks if vLLM 0.29 sparse-MLA still fails on sm_120
- **KTransformers** (model card links a GLM-5.3-Flash tutorial; CPU-expert/GPU-attention split — best
  match for 1 TiB RAM; untested on Blackwell here).
- **llama.cpp GGUF** (Unsloth guide; recent llama.cpp + `--n-cpu-moe`-style expert-on-CPU split;
  `llamacpp:*` images exist but would need a rebuild).
- **RedHatAI/GLM-5.3-Flash-NVFP4** (Blackwell-native NVFP4 variant, ~half the bytes/token → ~2× the
  tok/s) — ~155 GB download; the best perf path once the DIMM is fixed.

## 6. Verdict (updated 2026-09-08, after the Instance 9 rerun)

Servable here **in principle** — runtime confirmed live (`--check` passes; both Instance 8 and 9 proved
NCCL/sparse-MLA/`DEEPGEMM` actually initialize on this hardware) — but expect **~1–2 tok/s, ~4–8 tok/s
best case with MTP**: a "does it work" tier, not a daily driver. Stage 1 has now failed twice, for two
different reasons, and neither is the channel-G DIMM:
1. **RAM capacity under the current reduced-DIMM config (Instance 9, confirmed root cause).** GPU0's
   local NUMA node is memory-less right now, so its offload buffer piles onto the two populated nodes
   alongside GPU1's — the nominal offload budget doesn't leave real headroom once NUMA placement and
   UVA-mapping/page-table overhead are accounted for, and it triggered a genuine host-wide OOM (kernel
   killed unrelated processes before the vLLM worker). **Fix before retrying:** either wait for the DIMM
   reinstall (restores nodes 2/3), or NUMA-bind the container's offload to nodes 0/1 and cut
   `CPU_OFFLOAD_GB` well below 150/worker (e.g. ~100) to leave headroom, or try TP1.
2. **The fp8-MoE-finalize software reset (Instance 8) — still completely unresolved.** Instance 9 never
   reached this stage (it died earlier, during offload setup), so this question is untouched by
   today's result. Root cause still undetermined (capture gap over that specific death window, from a
   different run under full RAM). Needs its own rerun, once RAM capacity is no longer the limiting
   factor, with capture verified alive for the entire run.

The channel-G DIMM remains a long-run suspect for *sustained-serving* risk (§4) once past both items
above, but it is not what's blocking progress on either Stage-1 attempt so far. Recommended order:
(a) resolve the RAM/NUMA capacity issue (DIMM reinstall is the clean fix; NUMA-binding + a lower offload
budget is the workaround if there's no time to wait); (b) rerun `--dummy` and see whether it now reaches
and survives fp8-MoE-finalize; (c) if the Instance-8 reset reproduces, treat it as its own investigation
(NMI-watchdog vs. BMC watchdog, per §3.4) before ever attempting Stage 2's real weight load.
