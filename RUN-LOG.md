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

**~~Next: drop `--kv-cache-dtype fp8` and rerun.~~ — WRONG. Corrected below.**

#### Correction · 2026-09-15, later the same evening `[upstream]`

A concurrent session reading the image's own source found that the proposed fix cannot work, and the
real situation is worse and more decisive. Verified in
`vllm/platforms/cuda.py` and `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`:

1. **On sm_120 there are exactly two MLA backend candidates** — `TRITON_MLA` and
   `FLASHINFER_MLA_SPARSE_SM120`. Triton is filtered out for this sparse/indexer model (the run
   logged `out of potential backends: ['FLASHINFER_MLA_SPARSE_SM120']`), leaving one.
2. **That backend hard-requires fp8**: `if self.kv_cache_dtype != "fp8_ds_mla": raise
   NotImplementedError("FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla KV cache
   layout")`. So `KV_CACHE_DTYPE=auto` does not route around the kernel — it fails earlier, at
   backend construction.
3. **Its kernel hardcodes `pe_dim == 64`**, the DeepSeek rope shape — which is what we hit.

So on this build **both settings fail, for two different reasons, and no flag combination bridges
them.** Worse (and decisively): the `else` branch of the same selector contains explicit handling for
precisely this model shape — `prefer_fi_sm90 = hf.qk_rope_head_dim == 0 and hasattr(hf, "index_topk")`,
commented *"NoPE sparse MLA (GLM-5-Next shape…) prefer FlashInfer's SM90 FA3 path for every KV dtype
— BF16 and FP8 alike"*. **Upstream knows this exact model family and has implemented it for SM90
(Hopper) only.** sm_120 gets the DeepSeek-shaped path.

**Revised conclusion:** GLM-5.3-Flash is **not servable on sm_120 with `vllm/vllm-openai:glm53-flash`
by any flag combination** — a genuine capability gap, not a misconfiguration. This also explains the
family's whole history here: the FP8 sibling's Stage-1 attempts (R-004, R-005) died of unrelated
causes *before* ever reaching a forward pass, so this wall was always there and simply hadn't been
touched yet.

**Revised next steps**, in order of cost:
1. **Confirm empirically** — one `--dummy` with `KV_CACHE_DTYPE=auto`. Expect a *backend-selection*
   failure, not a `pe_dim` assert. Cheap, and converts this entry from `[upstream]` to `[measured]`.
2. **Force the backend** — `--attention-backend TRITON_MLA` (or `FLASHMLA_SPARSE`) to see whether any
   non-SM120 path accepts the model on sm_120. Likely rejected by a feature gate; cheap to find out.
3. **Newer vLLM** — the fix is upstream-shaped (extend the SM90 NoPE path to sm_120). Needs a pull;
   no local image registers `Glm5Next` except this one.
4. **SGLang** — the other runtime with a published GLM-5.3 recipe.

**Revised lesson (this is the transferable one):** when a runtime rejects a model, read the
*backend selector*, not just the failing kernel. `vllm/platforms/cuda.py` branches on
`device_capability.major`, and a model family can be first-class on one capability and unsupported on
another **in the same build**. That check is ten minutes and it retires whole days of flag
permutations. → KNOWLEDGE P-H

**Process note:** this entry was written by a Claude Code session *while* an opencode session was
independently running the same stage. The opencode agent found the contradiction by reading the image
source — the right call, and the reason this correction exists. See the concurrency note at the top
of this file.
