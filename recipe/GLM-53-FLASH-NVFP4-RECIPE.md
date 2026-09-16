# GLM-5.3-Flash-NVFP4 (nvidia) on "pensive" — recipe

Date: 2026-09-14, updated 2026-09-15. Status: **⛔ BLOCKED — not servable on this hardware with this
runtime.** Stage 1 (`--dummy`) ran 2026-09-15 and died at KV-cache init; investigation of the image's
backend selector shows sm_120 has **no NoPE sparse-MLA path at all** (upstream implements this exact
model shape for SM90/Hopper only), so no flag combination will serve it here. The NVFP4 loader,
sparse-MLA selection and TP2 NCCL all proved good on the way. Forward paths: newer vLLM, SGLang, or
Hopper. See §0 and the `RUN-LOG.md` R-014 correction.
Launcher: `recipe/serve-glm-53-flash-nvfp4.sh` (`--check` → `--dummy` → `--serve`).

This is the **NVIDIA ModelOpt NVFP4** variant of the same 320B/18B-active MoE as
`recipe/GLM-53-FLASH-RECIPE.md` (zai-org FP8, 305 GB). NVFP4 ≈ 0.5 B/param on the MoE/dense MLP
weights → **190.4 GiB on disk vs 305 GB** for the FP8 sibling. Consequence: far less CPU
weight-offload (~60 GiB vs ~300 GiB), less sustained DDR traffic per token, and NVFP4 kernels are
the fast path on this box's Blackwell GPUs (proven by Recipe A, Qwen3.8-Flash-Next-NVFP4, 39–55
tok/s). Chosen over `RedHatAI/GLM-5.3-Flash-NVFP4` (the earlier fallback in the FP8 recipe) because
it's NVIDIA's own ModelOpt build with published vLLM/SGLang recipes and benchmark deltas.

---

## 0. Where this stands (read before running anything)

- **Download — DONE.** `nvidia/GLM-5.3-Flash-NVFP4` completed **2026-09-14 14:38 local** (49m22s,
  44/44 files). Verified on disk 2026-09-15: **204,543,782,150 bytes = 204.5 GB / 190.5 GiB, 44
  files, 33 safetensors shards** (~8.3 GiB ×3, ~5.5 GiB ×30) at
  `/trunk/ai/huggingface/models/nvidia/GLM-5.3-Flash-NVFP4` — exactly the predicted size. Un-gated,
  MIT. Zero-byte `.incomplete` stubs left under `.cache/huggingface/download/` are normal residue,
  not a partial transfer; the log's `✅ Successfully downloaded` line is the completion signal.
  (Resumable if it ever needs re-running: same command, completed shards are hash-verified and
  skipped.)
- **Host blocker — CLEARED.** The 2026-09-10 NVIDIA kernel-module/userspace mismatch (module
  580.173.02 vs package 580.178.04) that broke `nvidia-smi` and every GPU container host-wide is
  **resolved**: both sides now read **580.178.04** (verified 2026-09-15 via
  `/proc/driver/nvidia/version` vs `modinfo nvidia`). Re-check that pair before any GPU stage — it
  is the failure that looks model-specific but isn't.
- **Stage 1 (`--dummy`) — ATTEMPTED 2026-09-15 22:27 local, failed at KV-cache init after 5m46s.**
  Clean container exit (code 1), no host reset, no OOM, capture alive throughout. Full entry:
  `RUN-LOG.md` R-014. What it **proved** (both were open questions §6):
  1. **The ModelOpt-NVFP4 loader for `Glm5Next` comes up clean** — `FLASHINFER_CUTLASS` NvFp4 MoE
     backend selected out of 7 candidates. Open question #1: **answered.**
  2. `FLASHINFER_MLA_SPARSE_SM120` sparse-MLA backend initializes on sm_120, and NCCL TP2 init is
     clean on `PYNCCL` (no hang, no cuMem segfault — all NUMA nodes populated, guard not needed).

  What **killed** it: `RuntimeError: concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla`.
  `--kv-cache-dtype fp8` makes vLLM select the `fp8_ds_mla` KV format for this backend (logged
  explicitly), and that kernel hardcodes `pe_dim == 64` — a DeepSeek-V3.2 shape. GLM-5.3's DSA is
  **NoPE**: `qk_rope_head_dim: 0`, `qk_nope_head_dim: 256`, `kv_lora_rank: 512`. Zero can never
  equal 64. **Not a hardware, capacity, or model-support problem — a flag conflict.**

  **⛔ CORRECTION, same evening — the obvious fix does not exist.** Reading the image's own selector
  (`vllm/platforms/cuda.py`) shows sm_120 offers only `TRITON_MLA` and
  `FLASHINFER_MLA_SPARSE_SM120`; Triton is filtered out for a sparse/indexer model, and the survivor
  **requires** `fp8_ds_mla` (`raise NotImplementedError` otherwise) while its kernel **hardcodes
  `pe_dim == 64`**. So `fp8` dies in the kernel and `auto` dies at backend construction — **no flag
  combination serves this model on sm_120 with this image.** The same selector's `else` branch
  implements this exact shape (`prefer_fi_sm90 = qk_rope_head_dim == 0 and hasattr(hf, "index_topk")`,
  commented *"GLM-5-Next shape"*) for **SM90 / Hopper only**. `--kv-cache-dtype` is now the
   `KV_CACHE_DTYPE` env var (default `fp8`, the more informative failure), but that is bookkeeping,
   not a fix. **Forward paths: a newer *vendor-fork* vLLM, SGLang, or Hopper hardware.** Full
   reasoning and the ranked next steps: `RUN-LOG.md` R-014 correction; generalized as KNOWLEDGE.md
   **P-H**.

   **⛔ TESTED 2026-09-16 — public `vllm/vllm-openai:nightly` does NOT fix it (R-015).** Pulled the
   public nightly (vllm 0.29.1rc1.dev187, flashinfer 0.6.18.post1) and re-ran `--dummy` with
   otherwise identical flags: same backend, same `fp8_ds_mla` selection, and the same kernel assert
   (`cache_kernels.cu:937, pe_dim must be 64`). Source inspection: the sm_120 selector branch is
   unchanged, and the compiled kernel still carries both asserts (`strings` on
   `_C_stable_libtorch.abi3.so`). The GLM-5-Next NoPE selector logic in `:glm53-flash` is vendor-fork
   code absent from the public tree. **"Pull a newer public vLLM" is retired as a fix path**; the fix
   must come from the vendor fork (newer `:glm53-flash`-equivalent) or SGLang. See `RUN-LOG.md` R-015.

   **Why the official GB200 recipe doesn't transfer to this box** — the vLLM recipe catalog
   (recipes.vllm.ai, GB200 variant) runs `RedHatAI/GLM-5.3-Flash-NVFP4` with `--tensor-parallel-size 4`
   on **GB200 = B200 = sm_100** with `--kv-cache-dtype fp8`. It works there because the sm_100
   selector branch picks `FLASHINFER_MLA_SPARSE` with **plain fp8** KV — never `fp8_ds_mla`, so the
   `pe_dim==64` assert never fires. On pensive (sm_120, 2 GPUs), the only candidate is
   `FLASHINFER_MLA_SPARSE_SM120`, which mandates `fp8_ds_mla`. TP4 is also not reproducible (we have
   2 GPUs). `--privileged` there is a GB200 host requirement, not something we adopt. The one flag
    worth adopting: `VLLM_ENGINE_READY_TIMEOUT_S=3600` (default is 600s — too tight for a 190 GiB
    ZFS load) — now baked into the launcher as `ENGINE_READY_TIMEOUT_S`.
 - **A working sm_12x NoPE path exists in a community fork (reference only — P12, not a serving
   path).** `samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell` (2026-09-16) runs GLM-5.3-Flash
   on 2× RTX PRO 6000 Blackwell 96 GB (PCIe, working P2P) via the digest-pinned
   `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.6.0` — a custom vLLM fork (same
   `0.1.dev20051` lineage as our `:glm53-flash`) carrying a **`B12X_MLA_SPARSE`** attention backend
   (fork `tpurtell/sparkinfer-glmrt`) that handles the NoPE DSA geometry on Blackwell, *including
   with `KV_CACHE_DTYPE=fp8_ds_mla`* — the exact format our stock kernel rejects (`pe_dim must be 64`).
   It also bundles EXL3 4-bit MoE kernels, a custom PCIe all-reduce (`VLLM_ENABLE_PCIE_ALLREDUCE=1`,
   backend `b12x`) with EP2/DCP2, and DFlash2 (incoai, CC BY-NC-ND 4.0 research license) spec-decode.
   Its model is a third-party EXL3 4-bit quant (127 GiB) that fits 2×96 GB fully GPU-resident (no
   offload) — why its 179→1045 tok/s headline (≈65 % draft acceptance) does not transfer to our
   2×72 GB box or to the 190 GiB NVFP4 weights. **Use:** proves the sm_120 NoPE gap is buildable;
   the fork is the reference for what an upstream fix looks like. Do not serve from it (P12) — read
   the source, port the approach into our own image, or wait for a vendor image.
 - **Also found (affects every real load):** this vLLM build **silently ignores**
  `--max-parallel-loading-workers` (`WARNING [parallel.py:959] ... not supported and will be
  ignored`). That is one of the standing burst-reduction gates in §3.4 — treat it as **not in
  effect** for this image, and rely on `spawn` + the slow ZFS read path instead when Stage 2 runs.
- **RAM/NUMA state (2026-09-15):** all 4 NUMA nodes populated — 129 / 258 / 129 / 254 GB,
  **751 GiB total, ~543 GiB available**, no serve running. Instance 9's memory-less-node OOM
  condition does not apply right now. Re-verify with `numactl -H` / `free -h` immediately before
  `--serve`.
- **What the FP8 sibling already proved on this box** (power-trip-instances.md): Glm5Next arch
  registers, `FLASHINFER_MLA_SPARSE_SM120` initializes, PLE offload allocator works. Its two Stage-1
  deaths were environment/software issues (Instance 8: `0xCF9` software reset at fp8-MoE-finalize;
  Instance 9: OOM under the 4-DIMM config). RAM is back to ~751 GiB / 4 populated NUMA nodes
  (2026-09-10), so Instance-9's cause should not recur at this offload size — but §3 gates still
  apply to every GPU run.

## 1. Model facts (from repo `config.json` + `hf_quant_config.json`, verified 2026-09-14)

| Property | Value |
|---|---|
| Repo | `nvidia/GLM-5.3-Flash-NVFP4` (ModelOpt PTQ of `zai-org/GLM-5.3-Flash`; MIT) |
| Arch | `Glm5NextForConditionalGeneration`, natively multimodal (image+video) |
| Size | 320B total / 18B active; **190.4 GiB on disk** (33 shards) |
| Experts | **288 routed / 8 per token + 1 shared**; `moe_intermediate_size` 2048; 3 dense + 42 sparse layers (45 total) |
| Attention | Hybrid: 34 `linear_attention` (KDA) + 11 `deepseek_sparse_attention` (NoPE sparse-MLA/DSA) |
| MTP | `num_nextn_predict_layers: 1` → speculative decoding available (stage 3) |
| Quant | **NVFP4** (ModelOpt v0.47, group 16); KV cache FP8; exempt (unquantized): embed, lm_head, layers 0–1 self_attn, all mlp.gate + shared_experts, DSA self_attns |
| Context | native **1,048,576** |
| Parsers | `--reasoning-parser glm45 --tool-call-parser glm47` (per model card) |

Quantization cost per the model card (BF16 → NVFP4): GPQA 0.9217→0.9211, SciCode 0.5621→**0.5769**,
MMMU-Pro 0.7688→0.763, AA-LCR 0.71→0.7106, IFBench 0.613→0.6054, Terminal-Bench 2.1 0.8258→**0.8315**
(temp 1.0, top_p 0.95). Negligible for our purposes.

## 2. Fit math on this box (2×72 GB VRAM, ~751 GiB RAM as of 2026-09-10)

- 190.4 GiB weights vs ~128–132 GiB usable VRAM at `--gpu-memory-utilization 0.90` (2×72 GB minus
  KV + activations + CUDA context) → **~60 GiB must live in host RAM** → default
  `--cpu-offload-gb 32` per worker at TP2. Re-verify `free -h` / `numactl -H` before each attempt.
- Compare: the FP8 recipe needed 150–280 GiB offload and measured ~1–2 tok/s (no MTP). NVFP4 halves
  the per-token expert stream (42 sparse layers × 8 experts × 3×4096×2048 × 0.5 B ≈ **~4.3 GB/token**
  across both GPUs), so decode should be markedly better. **Expect ~5–10 tok/s at TP2; measure,
  don't trust this line.**
- KV is cheap (34 KDA layers hold fixed conv/recurrent state; only the 11 DSA layers keep
  compressed MLA KV, fp8). Long context is affordable — start 8192, grow in stage 3.
- The 51B-param n-gram/PLE table (present in this arch family) is CPU-offloaded via
  `VLLM_PLE_CPU_OFFLOAD=1`, exactly like the Qwen-Flash-Next and GLM-5.3-FP8 recipes.

## 3. Power-trip risk + non-negotiable gates

Same channel-G/MM4 history as every other heavy load here (see `power-trip-instances.md` and
`power-trip-diagnosis.md`). A 190 GiB ZFS load + ~60 GiB offload stream is lighter than anything that
has tripped this box, but the gates stand:

1. `powertrip-capture` running **and its klog actively writing** (mtime check, not container liveness).
2. `edac-ce-watch.sh` running; baseline CE counts logged. Use **per-window** CE deltas for
   attribution, never `ras-mc-ctl --summary` lifetime totals.
3. GPU power cap 250 W, **verified** (`power.limit` readback), not just issued.
4. Serialized weight load: `--max-parallel-loading-workers 1`, `spawn`, `NCCL_P2P_DISABLE=1`.
5. No other serve running (launcher refuses while `qwen38-flash-serve` / `glm53-flash-serve` are up).
6. `sync; echo 3 > /proc/sys/vm/drop_caches` before real load.
7. NUMA guard: launcher probes each GPU's local NUMA node and forces `NCCL_CUMEM_HOST_ENABLE=0` if a
   node is memory-less (the segfault path from `ncclCuMemHostEnable`/`cuMemCreate`).

## 4. Staged plan (one variable at a time)

### Stage 0 — download + offline checks (no GPU) — download DONE, `--check` PENDING
```bash
# Download — ALREADY COMPLETE (2026-09-14). Kept for reference / re-run after any loss.
# (resumable; token auto-loaded from /trunk/ai/scripts/HF_TOKEN by the v3 script)
/home/praneet/Workspace/env/python/comfyui/venv/bin/python \
  /home/praneet/Workspace/pensive-ai-research/recipe/hf_bulk_download_v3.py \
  nvidia/GLM-5.3-Flash-NVFP4

# Completion sanity: 33 shards, ~190.5 GiB total  -- PASSED 2026-09-15
du -sb /trunk/ai/huggingface/models/nvidia/GLM-5.3-Flash-NVFP4   # want ~204,543,782,150 bytes
ls /trunk/ai/huggingface/models/nvidia/GLM-5.3-Flash-NVFP4/*.safetensors | wc -l   # want 33

# Image + arch + FlashInfer + modelopt-loader check (CPU-only, safe, no driver needed)
bash recipe/serve-glm-53-flash-nvfp4.sh --check
```

### Stage 1 — dummy-weight TP2 smoke (no real weights, low fabric risk)
```bash
bash recipe/serve-glm-53-flash-nvfp4.sh --dummy
```
Proves: Glm5Next registration, **ModelOpt-NVFP4 loader path**, sparse-MLA backend init on sm_120,
NCCL init under the NUMA guard, offload allocator sizing. CUDA graphs are ON here by default — if
capture is what we want to trust, better to find it on dummy weights.
**Gate:** must reach "model load complete / server up" cleanly. If it dies, that is a software
problem, not a memory problem — grab `docker logs` + the capture klog before trying anything else.
(Watch specifically for a repeat of Instance 8's `0xCF9` software reset, which died at
fp8-MoE-finalize on the FP8 path; NVFP4 exercises a different finalize path but the same box.)

### Stage 2 — real load, TP2, conservative
```bash
bash recipe/serve-glm-53-flash-nvfp4.sh --serve    # TP2, 32 GiB/wkr offload, ctx 8192, seqs 1
```
Expect a 10–30 min ZFS load (resumable: a trip just means relaunching the same command; complete
shards are re-verified, not re-downloaded). To watch it:

```bash
bash recipe/serve-glm-53-flash-nvfp4.sh --wait     # blocks until READY / FAILED / timeout
bash recipe/serve-glm-53-flash-nvfp4.sh --logs     # follow the real vLLM output
bash recipe/serve-glm-53-flash-nvfp4.sh --status   # one-shot: container, GPU, RAM, HTTP, log tail
```

⚠️ **Do not `tail -f /var/tmp/serve-glm53-nvfp4.log` for progress** — `docker run -d` returns
immediately, so that file holds the container ID and nothing else. Real output is in `docker logs`.
Also watch `/var/tmp/edac-ce-watch.out` and the capture klog mtime during a real load.
- Weights don't fit (OOM at load) → raise `CPU_OFFLOAD_GB` in steps of 16 (32→48→64).
- Loads and serves → record: load time, per-GPU weight residency (`nvidia-smi`, `docker logs | grep
  'GPU KV cache size'`), CE delta for the whole window (`recipe/ecc-per-window.sh --boot -1` after,
  or edac-watch output).

### Stage 3 — increments (only after Stage 2 stable, ONE per session)
1. `MAX_MODEL_LEN=32768`, then 131072 (KV is cheap here; check available KV cache after each step).
2. MTP spec decode: `SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":2}'` (1 nextn layer;
   the FP8 sibling hit a ring-capacity crash at 4 — start at 2).
3. `MAX_NUM_SEQS=4` once KV budget allows.
4. `EP=1` (`--enable-expert-parallel --enable-ep-weight-filter`, the model card's config) — may cut
   per-GPU expert footprint; untested at TP2, so test alone.
5. If CUDA-graph capture trips: `ENFORCE_EAGER=1` (accept the ~2–4× decode penalty, same as Qwen A).

## 5. Launcher reference — `recipe/serve-glm-53-flash-nvfp4.sh`

| Env | Default | Notes |
|---|---|---|
| `MODEL` | `/trunk/ai/huggingface/models/nvidia/GLM-5.3-Flash-NVFP4` | |
| `IMAGE` | `vllm/vllm-openai:glm53-flash` | same image as the FP8 recipe (Glm5Next archs + FlashInfer ≥0.6.17, verified by `--check`); fallback `:nightly` |
| `NAME` / `PORT` | `glm53-nvfp4-serve` / `8092` | distinct from the FP8 serve (8091) |
| `TP` | 2 | 2-GPU box; the card's TP4/EP recipe (GB200) is not reproducible here |
| `CPU_OFFLOAD_GB` | auto: 32 (TP2) / 80 (TP1), per worker | raise if load OOMs; measure the actual resident/offloaded split after first serve |
| `MAX_MODEL_LEN` / `MAX_NUM_SEQS` | 8192 / 1 | stage-3 levers |
| `GPU_MEM_UTIL` | 0.90 | |
| `KV_CACHE_DTYPE` | `fp8` | ⛔ **Neither value works on sm_120 with this image** — `fp8` asserts `pe_dim==64` in the kernel, `auto` is rejected at backend construction. Not a tuning knob; see §0 correction. |
| `ENFORCE_EAGER` | 0 (graphs ON) | set 1 if graph capture trips |
| `SPEC_CONFIG` | unset | stage 3, MTP |
| `EP` | 0 | stage 3, expert parallelism |
| `NCCL_CUMEM_HOST_ENABLE` | auto | memory-less-node guard, per-launch probe |
| `BIND_HOST` | `""` | host-side `-p` publish address; `""` = all interfaces (legacy), `100.70.5.43` = tailscale0 only. Engine still binds 0.0.0.0 inside the container (NAT needs it) |
| `API_KEY` | `""` | `""` = unauthenticated (legacy, current clients depend on it); if set → vLLM `--api-key`, clients send `Authorization: Bearer`. Clients are remote tailnet peers (SYSTEM-SPEC banner, 2026-09-16) |

Fixed flags: `--quantization modelopt --disable-custom-all-reduce
--max-parallel-loading-workers 1 --enable-auto-tool-choice --tool-call-parser glm47
--reasoning-parser glm45`; env `NCCL_P2P_DISABLE=1 VLLM_PLE_CPU_OFFLOAD=1
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; caps
`SYS_PTRACE`, seccomp/apparmor unconfined.

## 6. Verdict (provisional, 2026-09-14)

- **This is the right GLM-5.3-Flash target for this box**: 1/3.3 the weight traffic of FP8,
  Blackwell-native NVFP4 (the fastest path we have evidence for — Recipe A), same runtime that
  already passed Stage 0 for the FP8 sibling.
- Open questions, in priority order (updated 2026-09-15 — the old #1, the host-wide driver mismatch,
  is resolved, and the download is complete, so nothing external blocks Stage 0/1 now):
  (1) does the ModelOpt-NVFP4 loader for Glm5Next come up clean under `--dummy`; (2) actual
  offload/KV fit at ctx 8192 (tune `CPU_OFFLOAD_GB` from 32); (3) whether Instance 8's `0xCF9`
  software reset reproduces on a non-fp8 MoE finalize path — if it does, that's a
  software/runtime bug independent of model format and the top thing to chase.
