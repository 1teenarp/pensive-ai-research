# RUN-LOG — attempt ledger

**One row per attempt.** Every stage run against a model — `--check`, `--dummy`, `--serve`, and each
stage-3 tuning increment — whether it passed, failed, or took the box down. Newest first.

Why this file exists: attempts that fail *cleanly* (a container exits, the host stays up) used to
leave no durable trace. Raw logs in `/var/tmp/serve-*.log` are overwritten by the next launch, and
`power-trip-instances.md` only covers events that reset or crashed the **host**. This is the index of
*all* attempts, so a later session — or the user, weeks on — can reconstruct what was tried, in what
order, and why each thing was changed.

**Division of labour:**

| File | Holds |
|---|---|
| **`RUN-LOG.md`** (this) | Every attempt, compact, chronological. The index. |
| `KNOWLEDGE.md` | What the attempts *taught* — durable facts per system / runtime / model family, and the generalized patterns. Promote findings there (AGENTS.md §9.2). |
| `power-trip-instances.md` | Deep forensics for attempts that **reset or crashed the host**. Linked from the row. |
| `recipe/<MODEL>-RECIPE.md` §0 | The *current* rolled-up status per model (prose, rewritten in place). |
| `PROJECT-TODOS.md` | Task-level open/closed work. |

## Concurrency — more than one agent works this repo

Sessions overlap. An entry describing a run **you** just did may have been written by someone else
minutes ago, and a `TODO` you didn't write may be sitting in a half-finished entry.

- **Every entry names its author and when it was written.** If you find an entry with no attribution,
  add one rather than assuming it's yours.
- **An entry about your own run, written by someone else, is not a paradox** — check the timestamp
  and the attribution line, take the evidence, and move on. (This cost a session real time on R-014.)
- **Never launch a container without checking what's already running.** One GPU pair, several
  possible sessions: `docker ps` before `--dummy` or `--serve`, always.
- **A correction outranks the entry it corrects**, whoever wrote either one. Correct in place, dated,
  saying what changed and why — don't delete the original claim.

## How to add an entry

```bash
# Snapshot the machine-readable half (host state, exit code, config, failure signature)
bash recipe/run-log.sh --container glm53-nvfp4-serve --stage dummy
```

Then fill the four `TODO` lines — **Outcome**, **Changed vs. last attempt**, **Read**, **Lesson**,
**Next** — and prepend the entry below, plus a row in the ledger table. Do this **before changing
anything else** (AGENTS.md P2/P9): the value is in recording what one variable moved and what it did.

Entries are numbered `R-NNN` ascending by time. A row may be all you write for a routine pass; write
a full entry whenever something failed, surprised you, or moved a number.

---

## Ledger

| # | Date (local) | Model · stage | Changed | Outcome | Detail |
|---|---|---|---|---|---|
| **R-020** | 2026-09-24 | GLM-5.3-Flash-NVFP4 · `--serve` | stage `dummy` (R-019) → real 190 GiB weights; flags unchanged | **PASSED — first end-to-end real-weights serve of this model on pensive.** READY after 2637 s (~44 min); coherent output (correctness probe passed); **~2.4–2.5 tok/s** steady, comm-bound (NCCL spin-wait signature: 100 % util / 6 % mem / ~100 W). | [below](#r-020) |
| **R-019** | 2026-09-24 00:58 | GLM-5.3-Flash-NVFP4 · `--dummy` | `BLOCK_SIZE` 16 → 256 (R-018's "Next"; launcher default) | **PASSED — READY after 1537 s (~25.6 min).** Full NoPE sparse-MLA + DSA-indexer forward path now works on sm_120 (ported kernel, kv bfloat16, block 256, eager). Deepest stage ever: NCCL → plugin backend select → dummy load → KV/kpool alloc → decode warmup → **server up**. | [below](#r-019) |
| **R-018** | 2026-09-17 00:22 | GLM-5.3-Flash-NVFP4 · `--dummy` | `KV_CACHE_DTYPE` auto → `bfloat16` | **Failed at first decode warmup (1553 s)** — deepest yet: plugin backend selected, KV + kpool caches allocated, died in DeepGEMM `fp8_fp4_paged_mqa_logits`: sm_120 non-fp4 asserts `block_kv == 64`, but hybrid page-alignment inflated block 16→2176, so storage = 2176//kpool(4) = 544. **Next: `BLOCK_SIZE=256` (storage 64).** | [below](#r-018) |
| **R-017** | 2026-09-16 23:10 | GLM-5.3-Flash-NVFP4 · `--dummy` | `IMAGE` → `pensive/glm53-flash:nope-sm120-617d0cc` (ported NoPE plugin) | **Failed at backend selection (75 s)** — `--kv-cache-dtype auto` canonicalized to `fp8_e4m3`, rejected by the plugin's dtype list. Plugin registration/override itself proved good. | [below](#r-017) |
| **R-016** | 2026-09-16 22:55 | GLM-5.3-Flash-NVFP4 · port/build | new derived image, no GPU touched | **`--check` PASSED** — plugin entry points discovered, `FLASHINFER_MLA_SPARSE_SM120` slot override verified, inert without env gate. R-014/R-015 attention block lifted in software. | [below](#r-016) |
| **R-015** | 2026-09-16 00:27 | GLM-5.3-Flash-NVFP4 · `--dummy` | `IMAGE` → `vllm/vllm-openai:nightly` (0.29.1rc1, flashinfer 0.6.18) | **Failed, identical signature** — `pe_dim must be 64 for fp8_ds_mla`. Public nightly does not fix it; its compiled kernel still carries the assert and its sm_120 selector is unchanged. | [below](#r-015) |
| **R-014** | 2026-09-15 22:27 | GLM-5.3-Flash-NVFP4 · `--dummy` | first GPU stage for this model | **Failed at KV-cache init** — `pe_dim must be 64 for fp8_ds_mla`. NVFP4 loader + sparse-MLA selection proved good. **Corrected same evening: no flag fixes this — sm_120 has no NoPE sparse-MLA path in this build.** | [below](#r-014) |
| R-013 | 2026-09-14 13:49 | GLM-5.3-Flash-NVFP4 · acquire | — | Download complete, 44/44 files, 204.5 GB / 190.5 GiB, 49m22s | `GLM-53-FLASH-NVFP4-RECIPE.md` §0 |
| R-012 | 2026-09-10 12:54 | Qwen3.8-Flash-Next-FP8 · serve | driver module reloaded | Healthy serve restored after the host-wide driver mismatch was fixed | `MODEL-CATALOG.md` Recipe C |
| R-011 | 2026-09-10 ~10:00 | *(host)* · — | apt upgraded nvidia-driver | **All GPU containers broken host-wide** — module 580.173.02 vs package 580.178.04. Looks model-specific, isn't. | `SYSTEM-SPEC.md` banner |
| R-010 | 2026-09-09 | Qwen3.8-Flash-Next-FP8 · tune | `--kv-offloading-size` | **Rejected** — it's cross-request prefix reuse, not live-context extension. Weight offload is the right lever. | `MODEL-CATALOG.md` Recipe C |
| R-009 | 2026-09-09 | Qwen3.8-Flash-Next-FP8 · tune | `TP=1 PP=2` | **Dead end** — vLLM guards `VLLM_PLE_CPU_OFFLOAD` against `PP>1`. Not fixable without patching. | Recipe C |
| R-008 | 2026-09-09 | Qwen3.8-Flash-Next-FP8 · tune | `num_speculative_tokens` 4 → 5 | **Crash** — `QSA ring capacity 12 must divide the attention block size 1616`. 4 is the tested ceiling. | Recipe C |
| R-007 | 2026-09-09 | Qwen3.8-Flash-Next-FP8 · tune | +MTP spec decode (4) on top of graphs | **~20–24 tok/s** steady (from 11.5). ~65–70 % acceptance. Best config to date. | Recipe C |
| R-006 | 2026-09-09 | Qwen3.8-Flash-Next-FP8 · tune | CUDA graphs on (from eager) | ~8 → ~11.5 tok/s. Only 1.4× — itself evidence the bottleneck is TP comm, not launch overhead. | Recipe C |
| R-005 | 2026-09-08 22:46 | GLM-5.3-Flash (FP8) · `--dummy` rerun | NUMA guard + TP2 offload fix + capture-liveness check | **Host-wide OOM** at ~13 min, during offload allocation. Capacity, not fault. Capture stayed alive — the session's fixes held. | Instance 9 |
| R-004 | 2026-09-07 16:54 | GLM-5.3-Flash (FP8) · `--dummy` | first GPU stage for this model | **Software `0xCF9` warm reset** ~3 h in, at fp8-MoE finalize. Not a sync flood. Capture went blind 27 min in — death window unrecorded. **Still unresolved.** | Instance 8 |
| R-003 | 2026-09-07 | GLM-5.3-Flash (FP8) · `--check` | — | **Passed** — `vllm=0.1.dev20051`, `flashinfer=0.6.17`, all three `Glm5Next*` archs registered | `GLM-53-FLASH-RECIPE.md` §0 |
| R-002 | 2026-09-06 | Qwen3.8-Flash-Next-NVFP4 · serve | PLE patch + burst-reduction flags + CUDA graphs | **Full success, no trip** — native 262k ctx, **39–55 tok/s**. Became Recipe A. | Recipe A / `RESUME-NOTE.md` |
| R-001b | 2026-09-06 01:27 | Qwen3.8-Flash-Next-NVFP4 · isolation | `--load-format dummy` | **No trip** — proved the fault was in the weight-copy/fabric burst path, not model allocation | `power-trip-instances.md` |
| R-001a | 2026-09-05 → 09-06 | Qwen3.8-Flash-Next-NVFP4 · serve ×5 | one flag/fix per attempt | **5 host resets** during weight load (Instances 2, 4, 5, 6, 7) — the DIMM fault | Instances 2–7 |
| R-000c | 2026-09-05 | Qwen3.8-Flash-Next-NVFP4 · `--dummy` | `NCCL_P2P_DISABLE=1` | **NCCL hang root-caused and fixed** — cross-NUMA, no NVLink, P2P/CUMEM channels never complete `all_reduce` | `PROJECT-TODOS.md` A2 |
| R-000b | 2026-09-04 | GLM-5.2-NVFP4 · serve ×7 | provider / TP / quant / eager combos | **All 7 failed** — no Blackwell sparse-MLA backend (vLLM); fused-MoE loader shape bug (SGLang) after loading 656 GiB to RAM | `README-ramoffload-research.md` §9 |
| R-000a | 2026-09-04 | Qwen3.5-397B-A17B-NVFP4 · serve | TP1 + `--cpu-offload-gb 200` | **Served, ~1.0 tok/s** — matched the H2D bandwidth prediction exactly. Offload path proven. | `PROJECT-TODOS.md` A1, ramoffload §11 |

> R-000a … R-013 were **back-filled 2026-09-15** from the documents named in the Detail column —
> they are summaries of records that already existed, not newly recovered runs. R-014 onward are
> written at the time of the attempt.

---

## Entries

### R-020
**2026-09-24 — nvidia/GLM-5.3-Flash-NVFP4 · stage `--serve` (real weights)**
*(opencode session; entry written from in-session evidence — vLLM `/metrics`, `nvidia-smi`, docker
engine-log lines, and the in-conversation coherence probe. Per user directive the full container log
was not re-mined: it carries earlier failed attempts.)*

| | |
|---|---|
| **Outcome** | **PASSED — first end-to-end real-weights serve of this model on pensive.** READY after 2637 s (~44 min) ZFS load; HTTP 200 on :8092; correctness probe returned correct reasoning (worked out "capital of France = Paris" and was counting word counts — coherent, no silent garbage). Steady-state serving verified during the user's long generation. No host reset, no OOM. |
| **Duration** | 2637 s to READY (~44 min: 190 GiB ZFS load + engine init); still serving at entry time |
| **Image** | `pensive/glm53-flash:nope-sm120-617d0cc` |
| **Host** | driver 580.178.04 matched; RAM 751 GB total (~251 GB used / ~499 GB avail during serve); NUMA 125/251/125/247 GB (all populated); booted 2026-09-23 23:16:19 (uninterrupted through the run); no `iommu=pt` |
| **GPUs** | ~67 GB used per card; ~95–102 W draw; **power cap did not apply** (300 W limit, no interactive sudo — known non-issue, less fabric margin) |
| **Capture** | armed, klog actively writing |
| **Changed vs. last attempt** | ONE step: stage `dummy` (R-019) → `serve` (real 190 GiB weights). All flags identical to R-019: `BLOCK_SIZE=256`, `KV_CACHE_DTYPE=bfloat16`, `ENFORCE_EAGER=1`, `SPARSE_MLA=1`, TP2, 32 GB/wkr offload, ctx 8192, seqs 1, MTP off. |

**Config** — identical to R-019 minus `--load-format dummy`:
```
serve /model --tensor-parallel-size 2 --quantization modelopt --served-model-name nvidia/GLM-5.3-Flash-NVFP4
  --cpu-offload-gb 32 --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.90
  --kv-cache-dtype bfloat16 --block-size 256 --disable-custom-all-reduce --max-parallel-loading-workers 1
  --host 0.0.0.0 --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser deepseek_r1
  --enforce-eager
```

**Read:** deepest stage yet, and the first real-weights pass: 190 GiB ModelOpt-NVFP4 weights loaded
from ZFS with no trip/OOM/reset (the load event class that reset this host 5× in the DIMM-fault era
— R-001a). Ported NoPE sparse-MLA + DSA-indexer path executes on real weights; **output is
coherent** — the "silent garbage" failure mode (MoE input-scale fault 2) did not materialize,
consistent with the pre-run audit (36,297 serialized `input_scale` tensors + the base image's
ModelOpt mapping branch). This retires the last "does it actually work on real weights" question;
what remains open is performance characterization (below) and the Stage-3 levers.

**Measured steady state** (single long generation in flight; docker-log windows 15:52–15:56 +
`/metrics` + `nvidia-smi`):
- **~2.4–2.5 tok/s** sustained; mean inter-token latency ~412 ms (2139 s ÷ 5193 tokens)
- GPUs: 100 % util / 6 % memory-bandwidth util / ~95–102 W of 300 W / 64–66 °C →
  **NCCL spin-wait signature = comm-bound**, not compute/memory/thermal-bound
- KV cache 4.5–5.2 % → not KV-bound; SM 2572–2602 MHz (no throttling); MTP off; eager (no CUDA
  graphs — ported kernel declares CG `NEVER`)
- Bottleneck rank: (1) TP2 all-reduce over cross-NUMA host bounce (no NVLink,
  `NCCL_P2P_DISABLE=1`, no `iommu=pt`) — structural floor; (2) eager decode (per-token kernel
  launch overhead); (3) no MTP spec decode; (4) 32 GB/wkr offload → per-token H2D expert gather
  (~4.3 GB/tok stream)

**Lesson:** on the same interconnect this box does 20–24 tok/s on Qwen3.8-Flash-Next-FP8 (graphs +
MTP + 8 GB/wkr) and 2.4 tok/s here — the gap is config levers (CUDA graphs, MTP, offload depth),
not a hardware regression. And a pre-run input-scale audit correctly predicted the silent-garbage
fault would not fire on a fully-serialized checkpoint — audit-then-serve beats probe-after-crash
for the silent-garbage class.

**Next:** Stage 3, one variable — enable MTP:
`SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":2}'` (arch has 1 nextn layer; the FP8
sibling crashed at 4, start at 2; expect ~2.5–3× on a comm-bound box per P-J). After that, a
`VLLM_TORCH_PROFILER_DIR` round to split one decode round into NCCL-wait vs kernel vs H2D-gather
(settles levers 1–4 without flag roulette).

---

### R-019
**2026-09-24 00:58 local — nvidia/GLM-5.3-Flash-NVFP4 · stage `--dummy`**
*(opencode session; launched per user's "let's push"; snapshot via `run-log.sh` at 00:58 while server still up.)*

| | |
|---|---|
| **Outcome** | **Passed — server READY after 1537 s (~25.6 min).** HTTP 200 on :8092; `/v1/models` 200; dummy `/v1/completions` round-trips (gibberish output — expected with `--load-format dummy`). Container still running at snapshot time. |
| **Duration** | still running (exit n/a, OOMKilled=false); READY 1537 s from launch |
| **Image** | `pensive/glm53-flash:nope-sm120-617d0cc` |
| **Host** | driver 580.178.04 matched; RAM 751 GB total, 617 GB avail; NUMA nodes 125/251/125/247 GB; booted 2026-09-23 23:16:19; repo 735d6fa |
| **GPUs** | 0: 67026 MiB, 300 W; 1: 66302 MiB, 300 W — **power cap did not apply** (no interactive sudo; known non-issue, less margin) |
| **Capture** | armed, klog actively writing throughout |
| **Changed vs. last attempt** | ONE variable: `BLOCK_SIZE` 16 → 256 (launcher default; exactly R-018's "Next"). `KV_CACHE_DTYPE=bfloat16`, ported image, `ENFORCE_EAGER=1`, `SPARSE_MLA=1` all unchanged from R-018. |

**Config**
```
serve /model --tensor-parallel-size 2 --quantization modelopt --served-model-name nvidia/GLM-5.3-Flash-NVFP4
  --cpu-offload-gb 32 --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.90
  --kv-cache-dtype bfloat16 --block-size 256 --disable-custom-all-reduce --max-parallel-loading-workers 1
  --host 0.0.0.0 --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser deepseek_r1
  --load-format dummy --enforce-eager
```

**Signature**
```
(no errors) — benign warnings only: max_parallel_loading_workers ignored (parallel.py:959),
SymmMemCommunicator sm_120 unsupported (expected). Engine READY; HTTP 200; completion generated.
```

**Read:** deepest stage of any GLM-5.3 attempt on this box, in order: NCCL init (TP2, PYNCCL) →
**plugin backend selected** (`glm53_sparse_mla` overriding the `FLASHINFER_MLA_SPARSE_SM120` slot) →
ModelOpt NVFP4 loader (`FLASHINFER_CUTLASS` MoE backend) → dummy weights loaded (~57 GiB/rank) →
**KV + kpool caches allocated at block 256** → eager decode warmup → **server up**. This retires
R-018's DeepGEMM `fp8_fp4_paged_mqa_logits` storage-block assert (block 16 → 2176 via the hybrid
page-alignment rule → storage 544 ≠ 64): at block 256 (storage 64) the entire NoPE sparse-MLA +
DSA-indexer forward path executes on sm_120. What this does NOT yet prove: real-weight load and
output **correctness** (Stage 2 + probe) and decode throughput.

**Lesson:** on hybrid-attention models the KV block size is not just a memory-efficiency knob — it
feeds `storage_block = block // index_kpool`, which the indexer's logits kernel asserts on, and
vLLM's hybrid "attention page ≥ mamba page" alignment rule **silently inflates the default**
(16 → 2176 here). Trace the chain block → storage → kernel assert before trusting any default.

**Next:** Stage 2 — `--serve` real weights (190 GiB ZFS load; §3 gates; note power cap not applied
without interactive sudo, `drop_caches` needs root). Then a **correctness probe** (arithmetic +
factual + repeated-token check) before trusting throughput.

---

### R-016
**2026-09-16 23:10 local — nvidia/GLM-5.3-Flash-NVFP4 · port/build (no GPU touched)**
*(opencode session; ran alongside the live `qwen38-flash-serve` — CPU-only work throughout.)*

| | |
|---|---|
| **Outcome** | **Derived image built + `--check` passed (CPU-only).** No GPU stage attempted (GPUs held by the Qwen serve). |
| **Changed vs. last attempt** | The serving *image*: stock `:glm53-flash` (no NoPE path, R-014) → `pensive/glm53-flash:nope-sm120-617d0cc` with the ported NoPE sparse-MLA plugin. Everything else unchanged. |
| **Source** | `Libertai/glm53-flash-vllm-gb10` @ `617d0cc` (Apache-2.0), clean clone at `~/Workspace/vendor/`; Dockerfile `recipe/Dockerfile.glm53-flash-nope-sm120`; kernel built `GLM53_ARCHS=120a` **inside** the base image. |
| **Host** | unchanged from R-015; powertrip-capture + metrics stack up; serve untouched (re-verified by `docker ps` during the session). |

**Read (what `--check` proves without a GPU):** `Glm5Next*` archs + FlashInfer 0.6.17 + modelopt
loader intact on the derived image; both plugin entry points discoverable via
`vllm.general_plugins` (so vLLM's `load_general_plugins()` will arm them in every worker); the
backend override resolves `FLASHINFER_MLA_SPARSE_SM120` → `glm53_sparse_mla.backend` **only** with
`VLLM_GLM53_CUDA_SPARSE_MLA=1`, and is inert without it; the AOT `_C*.so` loads with no driver
present. What it does NOT prove: anything about GPU execution — kernel numerics, eager decode
speed, offload interaction.

**Corrections made during this port (P9/P10):**
1. LibertAI's README quickstart still says `VLLM_GLM53_MOE_INPUT_SCALE=1.0`; their *own pinned
   commit* retracts it (1.0 = 632× the calibrated median → e4m3 block-scale underflow →
   intermittent repetition). We leave the env unset: our nvidia checkpoint is fully serialised
   (36,297 inline `input_scale` tensors, sampled nonzero) and the base image's `routed_experts.py`
   has the ModelOpt-NVFP4 mapping branch that loads them. The "0.0 / `torch.empty`" story is a
   *weight-only-checkpoint* property, not universal.
2. Their README's capability table (fp8 KV, `UNIFORM_BATCH` graphs) contradicts the shipped
   `backend.py` (bf16-only, `NEVER`). Code wins: launcher defaults are `KV_CACHE_DTYPE=bfloat16`
   (settled by R-017), `ENFORCE_EAGER=1`.

**Incidents during the build (both self-inflicted, both recorded as lessons):**
- `rm -rf /k` on a bind-mounted clone path inside a probe container deleted the host-side `kernel/`
  tree (EBUSY protects the mountpoint only). Recovered with `git checkout -- kernel`.
- First image build failed because the in-image *source tree* shadowed the installed package for
  any process with that cwd, and the package's dev-JIT import fallback probes for a GPU. Fixed in
  the Dockerfile (install → delete source → verify from `/`). Also: torch 2.13 `CUDAExtension`
  metadata step imports the parent package — same failure channel.

**Lesson:** a "BLOCKED on hardware/runtime gap" verdict can expire the day someone publishes an
Apache-2.0 plugin that overrides the backend-enum slot — audit vendor-fork *plugins* (entry points,
not images) before retiring a model; P12 forbids their container, not their method.

**Next:** when the GPUs free — `bash recipe/serve-glm-53-flash-nvfp4.sh --dummy`
(bf16 KV, eager, plugin gated on), then `--serve` with the §3 gates, then a **correctness probe**
(arithmetic + factual prompts, repeated-token check) before trusting throughput — this whole saga
began with silent garbage, not crashes.

---

### R-015
**2026-09-16 00:27 local — nvidia/GLM-5.3-Flash-NVFP4 · stage `--dummy`**
*(Written 00:43 local, opencode session; run launched by the same session after user authorized pulling the public nightly.)*

| | |
|---|---|
| **Outcome** | **Failed at KV-cache init — identical signature to R-014.** Clean exit, no host reset, no OOM. |
| **Duration** | 4m16s (exit 1, OOMKilled=false) |
| **Image** | `vllm/vllm-openai:nightly` (vllm 0.29.1rc1.dev187, flashinfer 0.6.18.post1) — pulled fresh for this test |
| **Host** | driver 580.178.04 matched; RAM 751 GB total, 542 GB avail; NUMA 125/251/125/247 GB; booted 2026-09-14 08:12:28 |
| **GPUs** | 0: 26 MiB / 300 W; 1: 2 MiB / 300 W — **power cap did not apply** (no interactive sudo) |
| **Capture** | armed, klog actively writing throughout |
| **Changed vs. last attempt** | ONE variable: `IMAGE` — `glm53-flash` dev build → public `nightly`. Same flags, same config. |

**Config** — identical to R-014 (launcher defaults: TP2, offload 32 GB/wkr, ctx 8192, seqs 1, mem_util 0.90, `--kv-cache-dtype fp8`, dummy weights).

**Signature**
```
(Worker_TP0/TP1) RuntimeError: concat_and_cache_mla,
  /workspace/csrc/libtorch_stable/cache_kernels.cu:937,
  pe_dim must be 64 for fp8_ds_mla
```
(Same assert as R-014, now at line 937 in this build's kernel — assert moved, not removed. Confirmed by `strings` on `_C_stable_libtorch.abi3.so`: both `pe_dim must be 64 for fp8_ds_mla` and `rope_dim must be 64, got` are present in the compiled binary.)

**Read:** reached the same furthest stage as R-014 — NCCL init → backend select (`FLASHINFER_MLA_SPARSE_SM120`, sole candidate, `TRITON_MLA` filtered out) → `FLASHINFER_CUTLASS` NVFP4 MoE backend → dummy weights loaded (57.42 GiB, 70 s) → KV init/CUDA-graph memory profiling → **died in `concat_and_cache_mla`**. This **retires "pull a newer vLLM" as a fix path**: the public nightly's `platforms/cuda.py` sm_120 branch is unchanged (still only `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`), and its compiled kernel still hardcodes `pe_dim == 64`. Notably, the nightly's SM90 branch lacks even the `prefer_fi_sm90` NoPE selector logic the `glm53-flash` dev build carries — i.e. the GLM-5-Next support is a vendor fork, and upstream public vLLM has not merged the sm_120 NoPE path as of 2026-09-16.

**Lesson:** when a runtime gap is "the kernel is built for a different shape," the fix lives in a newer **kernel build**, not in flags or a newer tag of the same branch — check the compiled binary (`strings <.so> | grep <assert>`) before burning an attempt, and check the selector's capability branch before assuming a newer tag changed anything.

**Next:** switch runtime to **SGLang** (NVIDIA's own recipe for this checkpoint is SGLang; local `sglang:dev-glm52-nvfp4` lacks the `Glm5Next` arch — need the newer dev image), or hold for a vendor vLLM image whose sm_120 path handles NoPE DSA. Not a flag problem — do not retry `--kv-cache-dtype` permutations on vLLM sm_120.

---

### R-014
**2026-09-15 22:27 local — nvidia/GLM-5.3-Flash-NVFP4 · stage `--dummy`**
*Run launched by an opencode session; entry written 22:45 by a Claude Code session; corrected 23:10
after the opencode session read the image source. See the concurrency note above.*

| | |
|---|---|
| **Outcome** | **Failed at KV-cache init.** Clean container exit — no host reset, no OOM. |
| **Duration** | 5m46s (exit 1, OOMKilled=false) |
| **Image** | `vllm/vllm-openai:glm53-flash` |
| **Host** | driver 580.178.04 matched; RAM 751 GB total, 543 GB avail; NUMA 125/251/125/247 GB; repo `e91dbfb+dirty` |
| **Capture** | armed and actively writing throughout |
| **Changed vs. last attempt** | First GPU stage for this model — launcher defaults as shipped. |

**Config**
```
serve /model --tensor-parallel-size 2 --quantization modelopt
  --served-model-name nvidia/GLM-5.3-Flash-NVFP4 --cpu-offload-gb 32
  --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.90
  --kv-cache-dtype fp8 --disable-custom-all-reduce --max-parallel-loading-workers 1
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
  --load-format dummy
```

**Signature**
```
RuntimeError: concat_and_cache_mla, csrc/libtorch_stable/cache_kernels.cu:866,
              pe_dim must be 64 for fp8_ds_mla
```

**Read — this got further than any previous GLM-5.3 attempt, and clears the recipe's #1 open
question.** Stages reached, in order:

1. NCCL init clean at TP2 (`PYNCCL` all-reduce for both `tp:0` and `ep:0` groups) — no hang, no
   cuMem segfault, NUMA guard not needed (all nodes populated).
2. **`FLASHINFER_MLA_SPARSE_SM120` sparse-MLA backend selected** on sm_120.
3. **ModelOpt NVFP4 loader works** — `FLASHINFER_CUTLASS` NvFp4 MoE backend selected out of 7
   candidates. This was open question #1 in the recipe; **answered, it comes up clean.**
4. Dummy weights loaded; KV block size set to 64 for the `DEEPSEEK_V32_INDEXER` backend.
5. Died in `do_kv_cache_update` → `concat_and_cache_mla`.

Also of note: `--max-parallel-loading-workers` is **silently ignored** by this vLLM build
(`WARNING [parallel.py:959] ... is currently not supported and will be ignored`) — it is one of our
standing burst-reduction gates, so on a real load that mitigation is not actually in effect here.

**Root cause — a flag conflict, not a model or hardware problem.** `--kv-cache-dtype fp8` makes vLLM
select the `fp8_ds_mla` KV-cache format (logged explicitly: *"Using fp8_ds_mla KV cache format for
FLASHINFER_MLA_SPARSE_SM120 backend"*). That kernel is shaped for DeepSeek-V3.2 and hardcodes
`pe_dim == 64`. GLM-5.3's DSA attention is **NoPE** — `config.json` has `qk_rope_head_dim: 0`,
`qk_nope_head_dim: 256`, `kv_lora_rank: 512`. Zero rope dim can never satisfy the assert.

**Lesson:** a KV-cache dtype is not a free precision knob — it selects a *kernel*, and that kernel
carries shape assumptions from whichever model family it was written for. Check `qk_rope_head_dim`
against the KV format the backend logs before setting `--kv-cache-dtype fp8` on any MLA model. The
model card recommending FP8 KV was for a different serving stack.

**Revised lesson (this is the transferable one):** when a runtime rejects a model, read the
*backend selector*, not just the failing kernel. `vllm/platforms/cuda.py` branches on
`device_capability.major`, and a model family can be first-class on one capability and unsupported on
another **in the same build**. That check is ten minutes and it retires whole days of flag
permutations. → KNOWLEDGE P-H

**Process note:** this entry was written by a Claude Code session *while* an opencode session was
independently running the same stage. The opencode agent found the contradiction by reading the image
source — the right call, and the reason this correction exists. See the concurrency note at the top
of this file.
