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
| **R-031** | 2026-10-01 22:21 | GLM-5.3-Flash (zai FP8) · KT path · `serve` | **image + `libnuma1`** (root-cause fix for R-030); same flags as R-029 (CTX=501025, seqs=4, GPU0) | **PASSED.** libnuma resolves in-container; 3442-token prompt (layerwise CPU prefill — the exact R-030 crash path) → HTTP 200, `finish_reason: stop`, coherent "ACK". Short decode 391 correct. User's OpenWebUI requests 200. | [below](#r-031) |
| **R-030** | 2026-10-01 21:59 | GLM-5.3-Flash (zai FP8) · KT path · `serve` (user workload) | **no config change** — new workload shape: user's "very long prompt" (layerwise CPU prefill, >2048 tok) | **CRASHED (exit 137, host fine) — latent image defect exposed.** `RuntimeError: KT shared-memory NUMA setup failed` → `OSError: libnuma.so.1: cannot open shared object file`: the layerwise-prefill CPU-expert path lazily creates NUMA-aware shared-memory buffers via `ctypes.CDLL("libnuma.so.1")`; the image has no `libnuma1`. 4 h 56 m of short-prompt decode never touched this path. P7 miss: a 5 s `ctypes.CDLL` probe in `--check` would have caught it at intake. | [below](#r-030) |
| **R-029** | 2026-10-01 17:25 | GLM-5.3-Flash (zai FP8) · KT path · **Stage 2 `--serve`** | `--context-length` 8192 → **501025** (tutorial-validated full ctx; ONE variable — `MAX_RUNNING_REQUESTS=4` carried over from R-028) | **PASSED — Stage 2 READY; the client 400 class is fixed.** User's client hit `400 Bad Request` on the 8192-ctx smoke server: sglang validates `prompt+max_tokens ≤ context-length`, and the client's `max_tokens` exceeded 8192. Full-ctx serve: `max_model_len=501025`, previously-failing 9000-token request now 200, correctness probes pass (391; Paris/Seine). ~21 min to READY (warm cache). Also fixed the launcher zombie-reap (added `--init` — see R-029 note) after an `rm -f` left a zombie PID 1. | [below](#r-029) |
| **R-028** | 2026-10-01 12:39 | GLM-5.3-Flash (zai FP8) · KT path · `tune` — concurrency sweep | `MAX_RUNNING_REQUESTS` 1→4 (graphs bs 1/2/4 captured; ONE variable) | **PASSED — concurrency is NOT the lever for this regime.** Sweep 1/2/4 (256-tok gens, warmup discarded): per-request 5.81→3.23→1.76 tok/s (≈1/N split); aggregate decode-phase 5.83→6.48→7.05 tok/s (**+21 % at N=4** — sublinear; bandwidth-bound signature). TTFT 6.6→13.0→24.7 s. Kept `MAX_RUNNING_REQUESTS=4` (strictly ≥ the 1-config: single user identical, multi-user batched instead of queued). | [below](#r-028) |
| **R-027** | 2026-10-01 11:31 | GLM-5.3-Flash (zai FP8) · KT path · `--smoke` (devel base) | base image `-base` → `-devel` (nvcc 13.0.48 — R-026's "Next"; ONE variable) | **PASSED — first READY + coherent output of the KT path.** ~27 min to READY (warm ZFS cache: 62/62 shards ~12 s, expert init ~15 min, graph capture 179.8 s). Correctness probes pass (17×23=391; Paris/Seine). **Steady-state 5.81 tok/s** (client, 256-token gen ×3, warmup discarded; server log 5.76–5.80) — **2.4× the vLLM baseline (2.4–2.5, R-020)**. Container left running per user request. | [below](#r-027) |
| **R-026** | 2026-10-01 ~10:15 | GLM-5.3-Flash (zai FP8) · KT path · `--smoke` rerun #2 | image +`python3-dev` (R-025's "Next"; ONE variable) | **FAILED at CUDA-graph capture (bs=1), one stage deeper than R-024/025** — triton C-helper now compiles (R-025 retired), dies in sgl_kernel's nvcc JIT: `ninja … status 127 … /usr/local/cuda/bin/nvcc: not found` (CUDA toolkit ships only in `-devel`). | [below](#r-026) |
| **R-025** | 2026-10-01 09:31 | GLM-5.3-Flash (zai FP8) · KT path · `--smoke` rerun | image +`build-essential` (R-024's "Next"; ONE variable: image toolchain) | **FAILED at the same stage, one layer deeper** — CUDA-graph capture bs=1, triton JIT now *finds* gcc but the `cuda_utils.c` compile exits 1: `#include <Python.h>` fails, `/usr/include/python3.12` absent (no `python3-dev`). Everything before it (62/62 shards in 12 s warm, all 42 `AVX2_FP8_MOE_TP` inits) passed again. 30m16s, exit 0 (meaningless). | [below](#r-025) |
| **R-024** | 2026-10-01 08:31 | GLM-5.3-Flash (zai FP8) · KT path · **Stage 1 `--smoke`** (first GPU stage) | `--check` (R-023) → real 305 GB load, GPU0, TP1 | **FAILED at CUDA-graph capture (bs=1)** — `RuntimeError: Failed to find C compiler` (triton JIT of KDA `causal_conv1d_update`; base image ships no cc). **Both R-022 smoke unknowns resolved first**: AVX2 FP8 expert kernels + `blackwell_fp8` sm_120 profile are real (init-complete, 42/42 layers). 48m25s, RAM peak 458 GiB, host fine, exit 0 (meaningless — scheduler crashed). | [below](#r-024) |
| **R-023** | 2026-10-01 07:20 | GLM-5.3-Flash (zai FP8) · KT path · re-`--check` (routine) | **nothing** — read-only verification of the R-022 path on a fresh morning | **PASSED — all gates green, ready for `--smoke` (awaiting user go/no-go, P11).** `--check` re-passed on the same boot (2026-09-30 22:33): kt-kernel avx2, sglang-kt 0.7.0.post4, 62/62 shards, 794 GB avail. Both GPUs idle (26/2 MiB), driver 580.178.04 matched. **powertrip-capture was NOT running** (klog 5.9 days stale) → re-armed, klog now writing. edac-ce-watch already up. Docs synced: fork-tutorial provenance + layerwise-prefill/gpu-expert caveat in KNOWLEDGE §2 + recipe §0/§4. | [below](#r-023) |
| **R-022** | 2026-09-30 22:50 | GLM-5.3-Flash (zai FP8) · **new KTransformers path** — image build + `--check` | New runtime path alongside the vLLM one (user-directed pivot 2026-09-30): `pensive/glm53-kt:latest` (kt-kernel + sglang-kt 0.7.0.post4, built from official PyPI wheels) | **Image built + Stage 0 `--check` PASSED (CPU-only; no GPU stage, no serve).** AVX2-only CPU confirmed live → kt-kernel `avx2` variant; sm_120 confirmed via GPU probe (`sgl_kernel` 0.3.21 loads, flashinfer 0.6.3, cc 12.0); sglang-kt carries native `glm5_next*` model files; 62/62 shards. Open for `--smoke`: AVX2 FP8 kernel support + sm_120 DSA attention. P10 est. 3–12 tok/s vs 2.4–2.5 measured (R-020). | [below](#r-022) |
| **R-021** | 2026-09-25 00:51 | Qwen3.8-Flash-Next-FP8 · `tune` (inspection) | **nothing — read-only**, on a serve another session had running | **No change made. CPU parallelism ruled out as a lever, with thread-level evidence.** **24.3 tok/s** sustained at **96.84 % CPU idle**; exactly 2 of 112 cores pegged, and both are the TP workers' *main threads spin-waiting* on collectives (GPU 99–100 % util / 0–12 % mem / 111–116 W). torch already at 56 threads / 112 interop, no `OMP_NUM_THREADS` cap to lift. | [below](#r-021) |
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

### R-031
**2026-10-01 22:21 UTC (container clock) — GLM-5.3-Flash (zai-org FP8) · KTransformers · `serve` — PASSED (root-cause fix + regression)**
*(opencode session; the one variable vs R-030 is the image: `+libnuma1`. All launch flags
identical to R-029: CTX=501025, MAX_RUNNING_REQUESTS=4, GPU0.)*

| | |
|---|---|
| **Outcome** | **PASSED.** libnuma.so.1 resolves in the new image (3cf104f7ddcb); `--check` now includes the R-030 gate (in-image `ctypes.CDLL("libnuma.so.1")` probe) and passes. Server READY in ~9.5 min (fully warm ARC — `drop_caches` needs root, so ZFS page cache survived; 305 GB load in 573 s vs 27 min on cold). **Regression test: a 3442-token prompt (>2048 → layerwise CPU prefill, the exact R-030 crash path) returned HTTP 200, `finish_reason: stop`, coherent "ACK" in 333 s** (prefill ~325 s ≈ 10.6 tok/s + 16-token decode). Short decode correct (17×23=391). User's OpenWebUI requests returning 200. |
| **Duration** | ~9.5 min to READY (warm); regression probe 333 s |
| **Image** | `pensive/glm53-kt:latest` (3cf104f7ddcb) — `+libnuma1` |
| **Host** | driver 580.178.04; same boot; RAM 416 GiB available; no OOM/MCE |
| **GPUs** | GPU0 58.4 GiB; GPU1 2 MiB |
| **Capture** | armed, klog writing; power cap 300 W (no sudo) |
| **Changed vs. last attempt** | Image + `libnuma1` only (ONE variable, P2). |

**Read:** the layerwise-prefill CPU-expert path (first prompt > `--kt-gpu-prefill-token-threshold`
2048) now completes end-to-end — NUMA-aware shared-memory CPU-expert buffers create without
`OSError`. This rules in: the R-030 crash was purely the missing library, not a capacity or
topology problem (confirmed: /dev/shm 376 GB free, NUMA free 75/155/77/105 GB at crash time, no
host OOM/MCE). The long-prompt prefill measured ~10.6 tok/s (3442 tokens in ~325 s, first-use
including lazy compilation — P6: treat as upper bound, not steady state).

**Lesson:** a missing shared library in a lazy-init code path is indistinguishable from a
"NUMA setup failure" unless you read the full traceback — the generic `RuntimeError` message
("NUMA setup failed on at least one TP rank") is misleading; the real cause (`OSError:
libnuma.so.1`) is a few frames above. Always scroll to the root `OSError`/`ImportError`
before classifying. The P7 gate now in `--check` (in-image `ctypes.CDLL` probe) catches this
class in 5 s.

**Next:** the user's OpenWebUI is working. Stage-3 tuning levers remain (one per session, P2):
(1) `--kt-num-gpu-experts` residency for short-prompt workloads, (2) threadpool sweep
(`--kt-threadpool-count` 2/4/8), (3) `--kt-method MOE_INT8` A/B. The concurrency finding
(R-028: sublinear, DRAM-bound) already retired seqs>1 as a throughput lever for this
bandwidth-bound regime.

---

### R-030
**2026-10-01 17:04→21:59 UTC (container clock) — GLM-5.3-Flash (zai-org FP8) · KTransformers · `serve` (R-029 container, user workload) — CRASH**
*(opencode session; no config change vs R-029. Trigger: user's "very long prompt" — the first request
to take the layerwise CPU-prefill path (>2048 tokens vs `--kt-gpu-prefill-token-threshold 2048`)
on this image. Host stayed up the whole time (no reset, no OOM kill in kernel log).)*

| | |
|---|---|
| **Outcome** | **CRASHED (exit 137, OOMKilled=false, host fine) — latent image defect exposed.** ~296 min up: served short/medium prompts at 5.7–5.8 tok/s (incl. the user's first long generation, ~49k KV tokens, 2.4 h decode at 5.74 tok/s). Died at 21:59:49 UTC, 3 s after the user's next (very-long-prompt) request started: `RuntimeError: KT shared-memory NUMA setup failed on at least one TP rank` → root cause in the same traceback: `OSError: libnuma.so.1: cannot open shared object file: No such file or directory`. SIGQUIT → exit 137. |
| **Duration** | 296 min up (started 17:04:15 UTC, R-029) |
| **Image** | `pensive/glm53-kt:latest` (1143ecc59751) — **missing `libnuma1`** (Dockerfile never listed it) |
| **Host** | driver 580.178.04; same boot 2026-09-30 22:33:05; RAM 708 GiB available at crash (no host OOM); no MCE/CE flood; /dev/shm 376 GB, 16 K used (not the cause); NUMA free at crash: 75/155/77/105 GB |
| **GPUs** | GPU0 50.6 GiB; GPU1 2 MiB |
| **Capture** | armed, klog writing; power cap 300 W (no sudo) |
| **Changed vs. last attempt** | No config change — new workload shape: first request over the 2048-token layerwise-prefill threshold on this image. |

**Root cause:** the kt_ep_wrapper's layerwise-prefill path (`apply → _build_full_context →
initialize_cpu_buffers → _create_cpu_buffers`) creates NUMA-aware shared-memory CPU-expert buffers
via `ctypes.CDLL("libnuma.so.1")`. That library is not in the image (base CUDA devel image +
python + ktransformers wheels never pulled it in). The path is **lazy** — it only runs when a
prefill exceeds `--kt-gpu-prefill-token-threshold` (2048), so 4 h 56 m of short-prompt decode
(R-027/R-028 benchmarks + user's earlier chats) never touched it. The user's very-long prompt was
the first trigger; the buffer-creation `OSError` was wrapped into the generic "NUMA setup failed"
RuntimeError and killed the server.

**Evidence:** `docker logs` traceback (`kt_ep_wrapper.py:636 initialize_cpu_buffers → :1121
_create_cpu_buffers → :1094 _commit_cpu_buffer_phase` → `RuntimeError`), the bare `OSError:
libnuma.so.1` two frames above it, and `Dockerfile.glm53-kt` (no `libnuma1` anywhere).

**Read:** the server had been fully healthy; this was a missing-library latent bug, not a capacity
or hardware problem (host OOM/MCE ruled out; /dev/shm fine; NUMA free memory fine). Rules in:
layerwise CPU prefill is *the* first-touch path for long prompts on this image. Rules out:
host fault, /dev/shm sizing, NUMA topology.

**Lesson:** lazy-init code paths hide missing dependencies until the *first* use of that path,
which for a serving box is the worst possible moment (user-facing crash after hours of silence).
Cheap gate (P7): a `--check`-stage probe inside the image —
`python3 -c "import ctypes; ctypes.CDLL('libnuma.so.1')"` — would have caught this in 5 s at intake
instead of 296 min later. Add it to the launcher's `check_runtime()`.

**Next:** R-031 — image + `libnuma1` (ONE variable), rebuild (cached layers make this fast),
relaunch with the R-029 config (CTX=501025, seqs=4), verify libnuma resolves in-container, then
regression-test the crash path itself: a >2048-token prompt must prefill + decode clean.

---

### R-029
**2026-10-01 17:04–17:25 UTC — GLM-5.3-Flash (zai-org FP8) · KTransformers · Stage 2 `--serve` (full context)**
*(opencode session; ONE variable vs R-028: `--context-length` 8192 → 501025 (the launcher's `--serve` default; tutorial-validated config). `MAX_RUNNING_REQUESTS=4` carried over from R-028. Incidental launcher fix: added `--init` to the `docker run` (Docker's own recommendation after a zombie PID-1 blocked a prior `rm -f`; inert to serving behaviour).)*

| | |
|---|---|
| **Outcome** | **PASSED — Stage 2 READY.** Server up 17:25 (~21 min, warm ZFS cache: 62/62 shards ~64 s, expert init ~15 min, graph capture 187.2 s for bs 1/2/4). `max_model_len` now **501025** (was 8192). The user-reported `400 Bad Request` class is **fixed**: a `max_tokens=9000` request (which 400'd on the 8192-ctx server) now returns 200. Correctness probes pass (arithmetic 17×23=391, `finish: stop`; factual Paris/Seine, `finish: stop`). Container running per user request. |
| **Duration** | ~21 min launch → READY (warm cache) |
| **Image** | `pensive/glm53-kt:latest` (1143ecc59751; same as R-027/R-028) |
| **Host** | driver 580.178.04 matched; RAM 751 GB (334 GiB used at READY); NUMA 125/251/125/247 GB; same boot as R-022…R-028 (2026-09-30 22:33:05) |
| **GPUs** | GPU0 only: 50.6 GiB (bs 1/2/4 graphs + KV pool, fp8_e4m3, avail mem 24.6 GB at capture); GPU1 idle (2 MiB) |
| **Capture** | armed, klog writing; power cap 300 W (no sudo) — same caveat as R-027/R-028 |
| **Changed vs. last attempt** | ONE variable: context length 8192 → 501025. |

**The 400 the user hit (root cause):** the R-028 container ran `--context-length 8192` (Stage-1
smoke). sglang's OpenAI layer rejects `prompt_tokens + max_tokens > context-length` with
`400 BadRequest: max_completion_tokens is too large … at most 8192 completion tokens` (and also
`Requested token count exceeds the model's maximum context length` when the *total* exceeds). The
user's client was requesting a `max_tokens` beyond 8192 (typical for a client configured for GLM's
native 1 M context / large max-output). All other common params (model-name variants, temperature,
`n`, `response_format`, `stop`) probed clean — **the context cap was the only 400 trigger**.

**Verified post-restart:** `max_model_len` = 501025; `max_tokens=9000` + 13-token prompt → **HTTP 200**
(coherent reply); arithmetic + factual probes → correct, `finish: stop`.

**Caveats (unchanged from R-027/R-028):** FP8 KV scale defaults to 1.0 ("may lead to less accurate
results" — probes passed); power cap 300 W (no sudo); `drop_caches` needs root (not applied — the
load ran warm off ZFS ARC).

**Read:** Stage 2 of the ladder is done. The endpoint on `:8093` is now the full-context
(501025-ctx, max 4 concurrent, bs 1/2/4 graphs) serving path for GLM-5.3-Flash via KTransformers.
The 400 was a *server-side context cap* hitting a client's large `max_tokens`, not a bug in the
client or the runtime.

**Lesson:** a smoke-stage `--context-length` is a *request contract*, not just a memory setting —
any client that requests `max_tokens` above it gets a hard 400. When promoting a smoke server to a
usable endpoint, raise the context (Stage 2) before pointing other clients at it, or the first
"real" request 400s.

**Next:** the Stage-3 levers in priority order (one per session): (1) GPU-expert residency for
short prompts (`KT_NUM_GPU_EXPERTS` + raise `KT_GPU_PREFILL_THRESHOLD` so prefill stays GPU-side);
(2) threadpool/NUMA sweep (`KT_THREADPOOL` 4→8/16, `KT_CPUINFER` 56→48); (3) A/B MOE_INT8 (AMD BLIS)
vs the current FP8-AVX2 path. Concurrency is retired as a lever (R-028).

**Addendum (same container, 18:44 UTC):** a second 400 from the user's OpenWebUI client
(192.168.1.175) *after* the full-ctx restart — the server cap is no longer the cause; the client's
own model-config context/max-tokens must exceed 501025. Probed and ruled out on this server: model-
name variants, temperature, `n`, `tools` (list + empty), `stream_options`, `reasoning*` fields,
content-as-parts, tool-role history — all 200. The 400 set now is: (a) `prompt+max_tokens > 501025`,
(b) `top_p ∉ (0,1]`, (c) `frequency/presence_penalty ∉ [-2,2]`. **Client-side fix:** set the OWUI
model's Context Length ≤ 501025 (e.g. 262144) and a sane Max Tokens (e.g. 32768); re-fetch/restart
OWUI so it drops the stale 8192 cache. If still 400, the OWUI error body (UI "details" or its
server log) is the next artifact.

---

### R-028
**2026-10-01 12:14–12:39 (container clock) — GLM-5.3-Flash (zai-org FP8) · KTransformers · `tune` (concurrency sweep)**
*(opencode session; ONE variable vs R-027: `MAX_RUNNING_REQUESTS` 1→4, so graph capture now
includes bs 2 and 4. All other flags identical to R-027 (`--smoke`, ctx 8192, GPU0, FP8,
cpuinfer 56, threadpool 4, mem-fraction 0.65).)*

| | |
|---|---|
| **Outcome** | **PASSED — concurrency sweep complete; concurrency is NOT the throughput lever for this regime.** READY ~24.5 min (warm cache). Sweep (256-tok gen, fixed prompt, temp=0, warmup discarded per P6): per-request decode **5.81 → 3.23 → 1.76 tok/s** at N=1/2/4 (≈1/N split); **aggregate decode-phase 5.83 → 6.48 → 7.05 tok/s (+21 % at N=4)**; TTFT 6.6 → 13.0 → 24.7 s. Sublinear aggregate growth + 1/N per-request split = **shared-bandwidth (CPU-DRAM) bottleneck confirmed**, not compute-bound. Container left at `MAX_RUNNING_REQUESTS=4` (strictly ≥ the 1-config: single user gets the bs=1 graph — 5.81 tok/s, 6.6 s TTFT, identical to R-027 — while 2–4 concurrent users batch instead of queue). |
| **Duration** | ~24.5 min launch → READY (warm cache; 62/62 shards 64 s, graph capture 188.3 s for bs 1/2/4) |
| **Image** | `pensive/glm53-kt:latest` (same as R-027) |
| **Host** | driver 580.178.04 matched; RAM 751 GB (steady ~392 GiB); same boot as R-022…R-027 |
| **GPUs** | GPU0 only: 50.2 GiB; bs 1/2/4 graphs all captured (188.3 s); GPU1 idle |
| **Capture** | armed, klog writing; power cap still 300 W (no sudo) — same as R-027 |
| **Changed vs. last attempt** | ONE variable: `MAX_RUNNING_REQUESTS` 1→4 (the Stage-3 concurrency increment). |

**Measurement** (`/tmp/opencode/kt-concurrency.py`; streaming, fixed 200-word prompt, 256 max
tokens, temp=0; first request at each N discarded as warmup):

| N | wall | per-request decode | aggregate (decode phase) | TTFT |
|---|---|---|---|---|
| 1 | 50.5 s | 5.81 tok/s | 5.83 tok/s | 6.6 s |
| 2 | 92.0 s | 3.23 tok/s each | 6.48 tok/s | 13.0 s |
| 4 | 169.9 s | 1.76 tok/s each | 7.05 tok/s | 24.7 s |

**Read:** every decode step streams ~8.5 GB of expert weights from host RAM (42 layers × 8 experts
× 25.2 MB × 1 B FP8) *per token*, and that DRAM stream is shared across concurrent requests — so
per-request rate splits ~1/N. Batching recovers only launch-amortization + partial expert overlap:
+21 % aggregate at N=4. The decode is **bandwidth-bound, not compute-bound**; concurrency cannot
move the ceiling. The remaining levers are structural: (1) short-prompt GPU-expert residency
(`KT_NUM_GPU_EXPERTS`, recall the layerwise-prefill threshold-2048 normalization caveat), (2)
context length (KV is cheap at DSA/MLA fp8), (3) faster DRAM (AVX-512/AMX or DDR5 would raise the
bandwidth ceiling itself — hardware, not config).

**Lesson:** for a CPU-experts MoE on a bandwidth-bound box, measure the *aggregate* decode-phase
rate across a 1→2→4 sweep before believing a "concurrency" improvement — a flat aggregate with a
1/N per-request split is the fingerprint of a shared DRAM stream, and it retires concurrency as a
lever in one test instead of a week of flag roulette.

**Next:** Stage-3 lever #1 — `KT_NUM_GPU_EXPERTS` with a short-prompt config (raise
`KT_GPU_PREFILL_THRESHOLD` or keep prompts ≤ 2048 so prefill stays GPU-side and resident experts
pay off); or context 8192→32768 if the target workload is long-context.

---

### R-027
**2026-10-01 11:04–11:31 UTC — GLM-5.3-Flash (zai-org FP8) · KTransformers · `--smoke` (devel base image)**
*(opencode session; ONE variable vs R-026: base image → `nvidia/cuda:13.0.0-devel-ubuntu24.04`
(nvcc 13.0.48 verified present before launch). All launch flags identical to R-026.)*

| | |
|---|---|
| **Outcome** | **PASSED — first READY + coherent output of the KT path.** Server up 11:31 UTC (~27 min after launch; warm ZFS cache: 62/62 shards ~12 s, 42/42 expert inits, graph capture 179.8 s). Correctness probes passed (arithmetic 17×23=391 with correct reasoning; factual Paris/Seine in one line). **Steady-state 5.81 tok/s** (client-side, 256-token generation ×3, warmup discarded per P6; server-log cross-check 5.76–5.80 tok/s). **2.4× the vLLM baseline (2.4–2.5 tok/s, R-020)**. Container left running per user request ("keep the container running and measure it"). |
| **Duration** | ~27 min launch → READY (warm cache); 179.8 s graph capture; container still running at write time |
| **Image** | `pensive/glm53-kt:latest` (1143ecc59751; `nvidia/cuda:13.0.0-devel-ubuntu24.04` + `build-essential` + `python3-dev`) |
| **Host** | driver 580.178.04 matched; RAM 751 GB (peak ~448 GiB used during decode); NUMA 125/251/125/247 GB; same boot as R-022…R-026 (2026-09-30 22:33:05) |
| **GPUs** | GPU0 only, by design (TP1): 50.1 GiB (attention+dense+shared+embeds+KV+graphs, within mem-fraction 0.65). GPU1 idle (2 MiB) |
| **Capture** | armed, klog writing; **power cap 250 W NOT applied** (no passwordless sudo in session; read-back shows 300 W default — P4/P11: gate not passed, recorded honestly; host ran stable at 300 W) |
| **Changed vs. last attempt** | ONE variable: base image `-base` → `-devel` (nvcc present — exactly R-026's "Next"). |

**Stages reached, in order (all passed):** backend select (`blackwell_fp8`, trtllm NSA dispatcher on
SM120) → 62/62 shards (warm) → all 42 `AVX2_FP8_MOE_TP` expert inits → KV + kpool (fp8_e4m3,
page 64) → triton C-helper compile **OK** (R-025's death retired) → sgl_kernel nvcc JIT **OK**
(R-026's death retired; compile targets `arch=compute_120a, code=sm_120a`) → CUDA-graph capture
bs=1 (179.8 s) → warmup prefill → serving.

**Signature**
```
[2026-10-01 11:27:58] Capture cuda graph begin. … avail mem=24.57 GB
[2026-10-01 11:27:58] Capture cuda graph bs [1]
[2026-10-01 11:30:58] Capture cuda graph end. Time elapsed: 179.78 s. mem usage=0.13 GB
[2026-10-01 11:31:01] INFO:     Application startup complete.
[2026-10-01 11:41:09] Decode batch, #running-req: 1, #token: 128, … gen throughput (token/s): 5.76, cuda graph: True
```

**Caveats**
- Log: `Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0.
  This may lead to less accurate results!` — both correctness probes passed, so non-fatal; watch if
  downstream quality degrades.
- Power cap ran at 300 W (default), not the 250 W defensive gate — no sudo available in session.
- Only bs=1 graph captured (`--max-running-requests 1`); a concurrency increment will add bs=2/4 capture.

**Read:** every prior death on this path was *toolchain*, not architecture. R-022's two open
unknowns (AVX2 FP8 CPU-expert path, sm_120 DSA attention) are now confirmed at forward-pass level:
coherent, correct output at 5.81 tok/s. The three-stage JIT chain (cc → Python.h → nvcc) cost three
305 GB loads; the image is now complete.

**Lesson:** for any sglang-kt / KTransformers image that JIT-compiles at runtime (triton C-helper +
sgl_kernel CUDA JIT), the base must be `-devel` (or explicitly `build-essential` + `python3-dev` +
nvcc). The failures surface **only at graph capture, after the full weight load** — `which cc &&
which nvcc` at build time is the only cheap gate. → KNOWLEDGE §2.

**Next:** Stage-3 increments (one per session; container running, user may be using it):
(1) `MAX_RUNNING_REQS=2` — concurrency (bs=2 graph capture); (2) `MAX_MODEL_LEN` 8192 → 32768
(KV is cheap on DSA/MLA, fp8); (3) layerwise-prefill + `KT_NUM_GPU_EXPERTS` for short prompts
(recall the threshold-2048 normalization caveat).

---

### R-026
**2026-10-01 ~10:15 local — GLM-5.3-Flash (zai-org FP8) · KTransformers · `--smoke` rerun #2 (python3-dev fix)**
*(opencode session; ONE variable vs R-025: image gained `python3-dev` — `/usr/include/python3.12/Python.h` now present, verified `PYTHON_H_OK` + `PY_COMPILE_OK` before launch. All launch flags identical.)*

| | |
|---|---|
| **Outcome** | **FAILED at CUDA-graph capture (bs=1) — one stage deeper than R-024/R-025.** The triton C-helper compile (R-025's death) now **passes**; capture then dies in **sgl_kernel's nvcc JIT**: `ninja exited with status 127 … /usr/local/cuda/bin/nvcc: not found`. Container exited, host fine, no OOM, no reset. |
| **Duration** | 25m02s (exit 0, OOMKilled=false) |
| **Image** | `pensive/glm53-kt:latest` (rebuild: `nvidia/cuda:13.0.0-base` + build-essential + python3-dev) |
| **Host** | driver 580.178.04 matched; RAM 751 GB total; NUMA 125/251/125/247 GB; booted 2026-09-30 22:33:05 (same boot as R-022/023/024/025) |
| **GPUs** | GPU0 only, by design (TP1): ~17 GB during load; GPU1 idle (2 MiB) |
| **Capture** | armed, klog writing (0 s stale at snapshot); power cap not applied (no sudo — known non-issue) |
| **Changed vs. last attempt** | ONE variable: image gained `python3-dev` (exactly R-025's "Next"). |

**Stages reached, in order:** backend select (`blackwell_fp8` profile, trtllm NSA dispatcher) →
62/62 shards (fast — warm ZFS cache) → **all 42 `AVX2_FP8_MOE_TP` expert inits** → KV + kpool
(fp8_e4m3, page 64) → triton C-helper compile **OK** (R-025's failure retired) → **died in
CUDA-graph capture**: sgl_kernel JIT of `sgl_kernel_jit_fp8_blockwise_scaled_mm` (FP8 block-wise
GEMM, GPU-side dense/shared path) invokes `/usr/local/cuda/bin/nvcc` → not present in the `-base`
image (CUDA *toolkit* ships only in `-devel`).

**Signature**
```
Exception: Capture cuda graph failed: ninja exited with status 127
[1/2] /usr/local/cuda/bin/nvcc … -gencode=arch=compute_120a,code=sm_120a …
      -c /root/.cache/tvm-ffi/sgl_kernel_jit_fp8_blockwise_scaled_mm_7f4e79626d14ecda/cuda.cu
/bin/sh: 1: /usr/local/cuda/bin/nvcc: not found
```

**Read:** R-025's fix confirmed working (triton C-helper compiles cleanly). This run retired
"triton needs more than cc" and exposed the *next* JIT stage: sgl_kernel's nvcc path. Notably the
compile line requests `arch=compute_120a, code=sm_120a` — the sm_120 story keeps strengthening
(the GPU-side FP8 GEMM explicitly targets sm_120a, no fallback arch listed). Both make-or-break
questions from R-022 (AVX2 FP8 expert path, sm_120 attention) are now answered at init level;
forward-pass correctness still awaits a READY server. Load+init 25 min (warm cache) vs 48 min cold.

**Lesson:** a "base" CUDA image is missing **both** toolchains the sglang-kt JIT path needs —
`cc` (triton's C helper) **and** `nvcc` (sgl_kernel's CUDA JIT) — and each missing tool only
surfaces at graph capture, after the full weight load. Probe `which cc; which nvcc` at image-build
time for any image that will JIT-compile at runtime.

**Next:** R-027 — base image → `nvidia/cuda:13.0.0-devel-ubuntu24.04` (ONE variable; already on
disk locally), re-run `--smoke` with identical flags.

---

### R-025
**2026-10-01 09:30–10:01 local — GLM-5.3-Flash (zai-org FP8) · KTransformers · `--smoke` rerun (build-essential fix)**
*(opencode session; ONE variable vs R-024: image `pensive/glm53-kt:latest` rebuilt with a separate
`build-essential` layer (R-024's "Next"). All launch flags identical to R-024.)*

| | |
|---|---|
| **Outcome** | **FAILED at CUDA-graph capture (bs=1), one layer deeper than R-024.** Triton's JIT now *finds* a C compiler (gcc 13.3.0) but the compile of its generated `cuda_utils.c` exits 1: the source `#include <Python.h>`s and `/usr/include/python3.12` does not exist — the image has no `python3-dev`. Container exited, host fine, no OOM, no reset. |
| **Duration** | 30m16s (container 2026-10-01T09:30:47Z → sigquit 10:01:03; exit 0, OOMKilled=false) |
| **Image** | `pensive/glm53-kt:latest` (rebuild: `nvidia/cuda:13.0.0-base` + `build-essential`; pip layer cached) |
| **Host** | driver 580.178.04 matched; RAM 751 GB total (peak ~458 GiB during expert init); NUMA 128/258/129/254 GB (all populated); booted 2026-09-30 22:33:05 (same boot as R-022/023/024) |
| **GPUs** | GPU0 only, by design (TP1): 17 GB during load; GPU1 idle (2 MiB) |
| **Capture** | armed, klog writing throughout; power cap not applied (no interactive sudo — known non-issue) |
| **Changed vs. last attempt** | ONE variable: image gained `build-essential` (cc/gcc/make). Exactly R-024's "Next". |

**Stages reached, in order:** backend select (`blackwell_fp8` / trtllm NSA) → 62/62 shards in 12 s
(warm ZFS cache; `drop_caches` skipped — needs root, launcher warned and continued) → all 42
`AVX2_FP8_MOE_TP` expert inits (~38 s/layer) → KV + kpool allocated (fp8_e4m3, page 64) → **died in
CUDA-graph capture bs=1**: triton `driver → CudaUtils → compile_module_from_src(cuda_utils.c)` →
`CalledProcessError: /usr/bin/gcc … returned non-zero exit status 1` (missing `Python.h`; the
compile line's `-I/usr/include/python3.12` points at an absent dir).

**Signature**
```
File ".../triton/backends/nvidia/driver.py", line 63, in __init__
    mod = compile_module_from_src(...)
File ".../triton/runtime/build.py", line 51, in _build
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
subprocess.CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmp2x91lus6/cuda_utils.c',
  '-O3', '-shared', '-fPIC', …, '-I/usr/include/python3.12']' returned non-zero exit status 1.
```

**Read:** R-024's fix retired the "no compiler" failure and advanced the run exactly one JIT stage
further — the triton C-helper now compiles *through* the compiler search and dies at the missing
Python header. Everything upstream (load, 42× AVX2 FP8 expert init, KV/kpool) passed again, so those
remain retired. What this run rules in: the image is missing exactly one package for the triton path
(`python3-dev`), and the `-base` image still lacks the CUDA toolkit (`nvcc`), which the next capture
stage will want — flagged so R-026's failure isn't a surprise. (It wasn't: see R-026.)

**Lesson:** "add the compiler" is not one fix for a JIT-heavy runtime — triton needs a C compiler
*and* the Python dev headers; sgl_kernel/flashinfer additionally need `nvcc`. On any image that
JIT-compiles at runtime, probe `which cc; which nvcc; test -f /usr/include/python3.12/Python.h`
at build time, not after the 40-minute load.

**Next:** image + `python3-dev` (→ R-026). Note: a second, independent failure was found on the same
run's later capture stage — `nvcc` missing (→ recorded in R-026).

---

### R-024
**2026-10-01 08:31–09:20 local — GLM-5.3-Flash (zai-org FP8) · KTransformers · Stage 1 `--smoke` (first GPU stage)**
*(opencode session; user go 08:31 local (P11). Biggest single fabric event on this box to date:
305 GB ZFS → host RAM. Snapshot via `run-log.sh` at 09:22 while the (exited) container still existed.)*

| | |
|---|---|
| **Outcome** | **FAILED at CUDA-graph capture (bs=1)** — `RuntimeError: Failed to find C compiler` (triton JIT, KDA `causal_conv1d_update`). Container exited, host fine, no OOM, no reset. |
| **Duration** | 48m25s (exit 0, OOMKilled=false) |
| **Image** | `pensive/glm53-kt:latest` |
| **Host** | driver 580.178.04 matched; RAM 751 GB total (peak ~458 GiB during expert init); NUMA 128/258/129/254 GB (all populated); booted 2026-09-30 22:33:05 |
| **GPUs** | GPU0 only, by design (TP1): 17 GB peak; GPU1 idle (2 MiB) — see recipe §0 "Why only GPU0" |
| **Capture** | armed (re-armed 07:16 this morning, R-023) and writing throughout; power cap not applied (no interactive sudo — known non-issue) |
| **Changed vs. last attempt** | Stage: `--check` (R-022/R-023, CPU-only) → `--smoke` (real 305 GB load, GPU0, TP1, `KT_METHOD=FP8`). First GPU stage of the KT path. |

**Config** — launcher defaults (`recipe/serve-glm-53-flash-kt.sh --smoke`):
```
sglang.launch_server --model-path /model --kt-weight-path /cpu_weights --tp-size 1
  --context-length 8192 --mem-fraction-static 0.65 --chunked-prefill-size 2048
  --kt-method FP8 --kt-cpuinfer 56 --kt-threadpool-count 4 --kt-num-gpu-experts 0
  --kt-gpu-prefill-token-threshold 2048 --max-running-requests 1
  --cuda-graph-bs 1 2 4 --tool-call-parser glm47 --reasoning-parser glm45
  (GPU0, port 8093→30000, OMP_NUM_THREADS=56)
```

**Stages reached, in order:**
1. Backend select — `Set GLM-5-Next GPU profile=blackwell_fp8, KV cache=fp8_e4m3, KPool
   cache=fp8_e4m3, NSA dispatcher=trtllm; … graph-safe kernels on SM86/SM89/SM120` `[observed]`
2. Weights: 62/62 shards in 8m09s (~7.5 s/shard, warm ZFS cache; drop_caches was skipped — needs
   root, launcher warned and continued)
3. **CPU-expert init: all 42 sparse layers, `AVX2_FP8_MOE_TP 0..3` pools per NUMA node,
   `cpp_load_weights` ~51 s/layer (~35 min)** — RAM 380→458 GiB `[observed]`
4. KV + kpool caches allocated (fp8_e4m3, page 64, DSA)
5. **Died in CUDA-graph capture bs=1** — triton JIT of `causal_conv1d_update` (KDA conv1d,
   `linear_attn_backend=triton`, selected 09:20:13 as `decode=triton, prefill=triton`) →
   `triton.runtime.driver` → `compile_module_from_src` → **no C compiler in the image**

**Signature**
```
[2026-10-01 09:20:14] Scheduler hit an exception: … cuda_graph_runner.py:954 capture_one_batch_size
  File ".../sglang/srt/layers/attention/mamba/causal_conv1d_triton.py", line 1125, in causal_conv1d_update
  File ".../triton/runtime/build.py", line 31, in _build
RuntimeError: Failed to find C compiler. Please specify via CC environment variable
       or set triton.knobs.build.impl.
Exception: Capture cuda graph failed: Failed to find C compiler. …
```

**Read:** both of R-022's open smoke questions are now answered at init level:
(1) **the AVX2 FP8 CPU-expert path works on this Zen 3** — the tutorial's "AVX-512 FP8 CPU expert
kernel" line was conservative; kt-kernel 0.7.0.post4 created and populated `AVX2_FP8_MOE_TP` pools
for every sparse layer. (2) **sm_120 attention works at init** — sglang-kt explicitly selected the
`blackwell_fp8` GLM-5-Next profile with the trtllm NSA dispatcher, which names SM120. What is
still NOT proven: a forward pass (correctness) and decode throughput. RAM peak 458 GiB — the
recipe's ~350 GB estimate held with 150 GiB headroom.

**Lesson:** `nvidia/cuda:*-base` images ship **no C compiler**, and sglang-kt's triton path needs
one at graph-capture time (first JIT compiles a C driver module). Any future sglang-kt/KT image
needs `build-essential` (triton bundles its own `ptxas`/`cuobjdump` — only `cc` was missing).
Second: **exit 0 ≠ success** on this path — the scheduler exception propagated but the process
exited 0, so the container exit code is useless; the HTTP-readiness probe in `--wait` is the real
gate (it fired correctly here).

**Next:** R-025 — rebuild `pensive/glm53-kt` with `build-essential` (ONE variable: image toolchain),
re-run `--smoke` with identical flags. Correctness probe (arithmetic + factual + repeat-token)
remains the gate before any throughput reading.

---

### R-023
**2026-10-01 07:20 local — GLM-5.3-Flash (zai-org FP8) · KTransformers runtime: re-`--check` (routine re-verification) + preflight + capture re-arm**
*(opencode session; no variable moved. User asked for the KTransformers pivot to be re-confirmed and the
recipe re-verified after a fresh morning; ran `--check` again, probed host state, re-armed capture.)*

| | |
|---|---|
| **Outcome** | **PASSED — routine re-check. All standing gates green; ready for `--smoke`, awaiting user go/no-go (P11). No GPU stage attempted.** |
| **Changed vs. last attempt** | **Nothing.** Read-only verification of the R-022 path. |
| **Host** | driver 580.178.04 matched (module + package); RAM 751 GB total / 794 GB avail (free 724); NUMA 128/258/129/254 GB (all populated); booted 2026-09-30 22:33:05 (same boot as R-022) |
| **GPUs** | both idle: 0 → 26 MiB, 1 → 2 MiB; power limit 300 W (no cap applied — known non-issue) |
| **Capture** | **was NOT running** (klog 5.9 days stale, last write 2026-09-25) → re-armed via `run-powertrip-capture.sh start`; new klog `20261001-071605` writing (age=1s). edac-ce-watch already running. |

**`--check` (CPU-only, inside `pensive/glm53-kt:latest`):** kt-kernel 0.7.0.post4 **cpu-variant =
avx2** (EPYC 7663 Zen 3 — no AVX-512, no AMX) · sglang-kt 0.7.0.post4 · `glm5_next*.py` model files
present (base/dsa/moe/norm) · `config.json` parses (`model_type: glm5_next`,
`Glm5NextForConditionalGeneration`) · RAM 794 GB ≥ 350 GB required · model 62/62 shards (305 GB).

**Docs synced this session (no runtime change):** (a) verified the user-provided
`1teenarp/ktransformers` fork's GLM-5.3-Flash tutorial is **content-identical** to kvcache-ai main
(same 109 lines / 3.78 KB, same flags/numbers) — provenance cites kvcache-ai upstream; (b) recorded
the tutorial §4 fact that **Layerwise Prefill normalizes `--kt-num-gpu-experts` to 0**, reshaping
recipe Stage-3 lever 1 (GPU-resident experts only pay off for prompts ≤ the 2048 prefill threshold).
Both recorded in `KNOWLEDGE.md` §2 and the recipe §0/§4.

**Read:** re-confirms R-022's Stage-0 findings on an unchanged boot. Retires nothing new — the two
open `--smoke` questions remain: (1) does the **avx2** kt-kernel variant ship the **FP8** CPU-expert
path on Zen 3, and (2) does sglang-kt's DSA/KDA attention work on **sm_120**. Capture liveness is now
verified-fresh (P4) rather than assumed.

**Lesson:** "capture running" from a prior session does not survive a reboot — the powertrip-capture
container is `restart: no` by design (P8), so it is down after every boot. Re-arm and **check the
klog mtime**, not just the container's prior state, before any heavy load (P4: liveness ≠ working;
here the container simply wasn't running at all).

**Next:** Stage 1 `--smoke` — `bash recipe/serve-glm-53-flash-kt.sh --smoke` (real 305 GB ZFS load →
RAM, ctx 8192, GPU0, `KT_METHOD=FP8`, `OMP_NUM_THREADS=56`). **Needs user go/no-go** (P11: biggest
single fabric event on this box). Gate before trusting throughput: correctness probe (arithmetic +
factual + repeat-token). On AVX2-FP8 failure: fall back per recipe §4 (MOE_INT8 source build is
P12-clean; LLAMAFILE/GGUF is P12-flagged, needs explicit OK).

---

### R-022
**2026-09-30 22:50 local — GLM-5.3-Flash (zai-org FP8) · KTransformers runtime: image build + Stage 0 `--check`**
*(opencode session; user-directed pivot to KTransformers per kvcache-ai's GLM-5.3-Flash tutorial.
CPU-only work + one GPU probe; no model loaded, no serve started, no GPU held.)*

| | |
|---|---|
| **Outcome** | **Image built + Stage 0 `--check` PASSED.** No GPU stage attempted. |
| **Changed vs. last attempt** | Not an increment of an existing recipe — a **new parallel runtime path** for the same model (zai-org GLM-5.3-Flash, 305 GB / 62 shards): vLLM + ported NoPE kernel (R-016–R-020, measured 2.4–2.5 tok/s, comm-bound) → **KTransformers CPU-experts** (experts execute on CPU, attention/dense/shared on GPU, TP1 → zero all-reduces). One variable moved: runtime. |
| **Host** | driver 580.178.04 matched (module+package); RAM 751 GiB total / 794 GiB avail; NUMA 125/251/125/247 GB (all populated); booted 2026-09-30 22:33:05 |
| **Build** | `docker build -f recipe/Dockerfile.glm53-kt -t pensive/glm53-kt:latest` → 16.2 GB: `nvidia/cuda:13.0.0-base-ubuntu24.04` + python venv + `ktransformers[sglang]==0.7.0.post4` (kt-kernel 0.7.0.post4, sglang-kt 0.7.0.post4, transformers 5.6.0.post5, torch 2.9.1+cu128, sgl_kernel 0.3.21, flashinfer 0.6.3). Build-time import check passed: `kt-kernel 0.7.0.post4 variant: avx2`, `sglang 0.0.0.dev0` (sglang-kt fork). |
| **Probe** | GPU container (no model): `torch.cuda` True, cc (12, 0), `sgl_kernel` loads OK (0.3.21), flashinfer 0.6.3. `sgl_kernel`'s payload cache ships `sm90` + `sm100` variants only — **sm_120 load-verified live** (it resolved and imported), but that's load, not use. |

**`--check` results (all inside the image, CPU-only):** kt-kernel cpu-variant = **avx2** (EPYC 7663 is
Zen 3 — **no AVX-512, no AMX**, live-probed via /proc/cpuinfo) · sglang-kt 0.7.0.post4 ·
`sglang/srt/models/glm5_next*.py` present (4 files: base, dsa, moe, norm) · `config.json` parses
(`model_type: glm5_next`, `architectures: [Glm5NextForConditionalGeneration]`) · RAM 794 GB ≥ 350 GB
required · model dir 62/62 shards.

**Two corrections made during this attempt (recorded per P9):**
1. `transformers.AutoConfig.from_pretrained` **cannot** load this checkpoint — plain transformers
   5.6.0.post5 has no `glm5_next` model_type and the config has no `auto_map`. The architecture is
   registered **in sglang-kt itself**, so `--check` asserts on sglang-kt's model files + config.json
   instead. (This cost one failed check before being caught.)
2. Full `from sglang.srt.models.glm5_next import …` pulls a deep chain ending in `sgl_kernel`
   (`eplb → forward_batch_info → moe_runner → deep_gemm → sgl_kernel`), which **cannot import in a
   no-GPU container** (no `libcuda.so.1`; payload variants are sm90/sm100 only). Deferred to
   `--smoke` with a GPU — the load probe above shows `sgl_kernel` *does* import on sm_120.

**Read:** reached Stage 0 (image + runtime + model presence). Retires: image-build risk, CPU-variant
mismatch risk (avx2 auto-selected as expected), model-availability risk. Proves **nothing** about
GPU execution. The two live open questions for Stage 1: (a) does the **avx2** kt-kernel variant
actually ship the **FP8** CPU-expert path (GLM-5.3 tutorial says "AVX-512 FP8 CPU expert kernel";
AVX2-Tutorial lists FP8 among supported methods — docs disagree, runtime will decide at load time);
(b) does sglang-kt's DSA/KDA attention path work on **sm_120** (tutorial claims SM120; the generic
kt-kernel CUDA matrix lists only SM80–90).

**Lesson:** on a new runtime, the two make-or-break questions (CPU ISA variant + GPU arch support)
each resolve in under a minute with a throwaway container probe (`kt_kernel.__cpu_variant__`,
`torch.cuda.get_device_capability`, `import sgl_kernel`) — do those **before** writing a recipe's
risk section, and before committing to a 305 GB load. A "native support" claim in a model-specific
tutorial outranks the generic matrix, but still needs a live load probe to retire it.

**P12 note:** user-directed exception (2026-09-30) — the runtime is third-party
(kvcache-ai/KTransformers, Apache-2.0) but installed from **official PyPI wheels into our own image**
(`recipe/Dockerfile.glm53-kt`); default weights are vendor-published zai-org FP8 (no conversion, no
third-party quant). Fallbacks: `KT_METHOD=MOE_INT8` (AMD-BLIS source build, P12-clean — vendor
weights, KT's own converter) or `KT_METHOD=LLAMAFILE` + unsloth GGUF (**third-party quant — needs
explicit user OK**). Provenance recorded in `recipe/GLM-53-FLASH-KT-RECIPE.md` §0 and
`KNOWLEDGE.md` §2.

**Next:** Stage 1 `--smoke` (real 305 GB load, ctx 8192, GPU0, all standing gates: capture+CE watch,
power-cap readback, exclusivity, drop_caches) — **needs user go/no-go** (P11; biggest fabric event on
this box to date). Correctness probe (arithmetic + factual + repeat-token) is the gate before any
throughput reading — this family's saga began with silent garbage, not crashes. On AVX2-FP8
failure: fall back per §4 of the KT recipe.

---

### R-021
**2026-09-25 — Qwen/Qwen3.8-Flash-Next-FP8 · stage `tune` (resource-usage inspection, no variable changed)**
*(Claude Code session, written 2026-09-25 00:59 local. Read-only: the serve was already running —
started by another session ~00:39 — and a client request was generating throughout. Nothing was
started, stopped, or reconfigured. Evidence: `mpstat`, `mpstat -P ALL`, `ps -eLo psr`, vLLM
`/metrics`, `nvidia-smi`, `docker exec` env read.)*

| | |
|---|---|
| **Outcome** | **No change made — question answered in the negative.** User asked whether *CPU parallelism* could be raised to improve token generation. It cannot: the CPU is not the constraint and no CPU-side cap exists to lift. |
| **Duration** | inspection only; serve still running (exit 0, OOMKilled=false) |
| **Image** | `vllm/vllm-openai:qwen38-flash-next-patched` |
| **Host** | driver 580.178.04 matched; RAM 751 GB total, 659 GB avail; NUMA 125/251/125/247 GB; booted 2026-09-24 19:11:56; repo 6d87191 |
| **Changed vs. last attempt** | **Nothing.** Deliberately read-only (P2 not applicable — no variable moved). |

**Config** (as found, unmodified)
```
serve /model --tensor-parallel-size 2 --pipeline-parallel-size 1 --max-model-len 262144
--max-num-seqs 1 --gpu-memory-utilization 0.95 --disable-custom-all-reduce
--max-parallel-loading-workers 1 --cpu-offload-gb 8
--speculative-config {"method":"mtp","num_speculative_tokens":4}
env: NCCL_P2P_DISABLE=1 VLLM_PLE_CPU_OFFLOAD=1 VLLM_QWEN38_PLE_FP8_SCALE=1
```

**Measured** (30.03 s window, live request in flight, warmup not applicable — steady state)

| Metric | Value |
|---|---|
| Generation rate | **24.3 tok/s** (730 tokens / 30.03 s, from `generation_tokens_total`) |
| CPU, aggregate | **96.84 % idle** (2.53 us / 0.62 sy) of 112 logical cores |
| CPU, per-core | **2 cores pegged at 100 %** (cpu14, cpu110); cpu54 62 %, cpu110-adj 38 %; remaining ~108 idle |
| Pegged threads | `VLLM::Worker_TP0` (pid 255532, TID==PID, core 110) and `VLLM::Worker_TP1` (pid 256422, TID==PID, core 14) — **main threads**, i.e. spin-wait, not compute |
| GPU | 99–100 % util, **0–12 % memory util**, **111–116 W of 300 W**, SM 2.57–2.60 GHz |
| Threading | `torch.get_num_threads()=56`, `interop=112`; **no `OMP_NUM_THREADS`/`MKL_NUM_THREADS` set** |
| Spec decode | 1925 accepted / 821 drafts @ 4 spec tokens → **2.34 of 4 accepted (~59 %)** |

**Read:** Reached and held **steady-state serving** — this is not a stage failure but a
characterisation of a healthy run. The thread-level data rules **in** the comm-bound diagnosis and
rules **out** every CPU-side explanation: at 97 % idle with 56 torch threads already available and no
`OMP` cap, there is no CPU parallelism to add. The two pegged cores are the *symptom* of waiting on
the host-bounced collective path, not evidence of CPU saturation.

**Lesson:** **Aggregate CPU idle hides the signature — check per-core before concluding "CPU is fine".**
97 % idle and "2 cores at 100 %" are the same measurement, and only the second one tells you the
workers are spin-waiting. Generalizes to any comm-bound TP run: the spin shows up as a small number of
fully-pegged cores equal to the TP degree, each a worker *main* thread, while the box looks idle.
First thread-level confirmation of the NCCL spin-wait signature previously recorded only as
GPU-side (99 % util / low mem / half power) in KNOWLEDGE §1.

**Next:** Not a CPU variable. The two real levers, both already tracked and both needing a restart
window: **`PROJECT-TODOS.md` A4.1** — `--max-num-seqs` is **1**, so every collective is amortized over
a single sequence (drop `--max-model-len` 262144 → 32–64k, raise concurrency to 2–4; est. 2–4×
aggregate, **P10 estimate, unmeasured**); then **A4.2** — both workers run with `allowed cpus: 0-111`,
**entirely unpinned across 4 NUMA nodes**, and Worker_TP1 was observed on node1 while its GPU (GPU1)
is on node0. Per-process buffer placement was **not** confirmed — `/proc/<pid>/numa_maps` needs root.

**Note:** 24.3 tok/s sits at the top of the 20–24 tok/s FP8 band in KNOWLEDGE §1, so this config is
performing well *for FP8*; the structural ceiling remains the 8.0 GB/s host-bounced all-reduce path
(R-021 does not change that picture, it explains where the time goes on the CPU side).

---

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
