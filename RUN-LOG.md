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
