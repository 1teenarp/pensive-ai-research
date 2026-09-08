# GLM-5.3-Flash on "pensive" — research + staged recipe (NOT yet run)

Date: 2026-09-07. Status: **research complete, staged plan ready, nothing launched.**
Model: `/trunk/ai/huggingface/models/zai-org/GLM-5.3-Flash` (305 GB on disk, 62 shards).
Launcher: `recipe/serve-glm-53-flash.sh` (staged: `--check` → `--dummy` → `--serve`).

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

## 2. Fit math on this box (2×72 GB = ~144 GB VRAM, 1 TiB RAM)

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

## 3. Blockers found during research (why we can't start today)

1. **Runtime support — CONFIRMED MISSING locally.** Registry check (inside each image):
   - `qwen38-flash-next-patched` (vllm 0.1.dev/0.27-era, transformers 5.15.1): no `Glm5Next*` arch.
   - `vllm/vllm-openai:latest` (0.27.1) and `v0.23.0`: no `Glm5Next*` arch.
   - `lmsysorg/sglang:dev-glm52-nvfp4`: has `kimi_linear.py` + `deepseek_v4.py` but **no `glm5_next.py`**.
   - Fix: `vllm/vllm-openai:glm53-flash` — **present on the registry** (verified via
     `docker manifest inspect`); `:nightly` as fallback. This also answers the old GLM-5.2
     "no Blackwell sparse-MLA backend" blocker: the GLM-5.3-Flash path supports NVIDIA
     **Hopper-and-newer** with FlashInfer ≥0.6.17 (Blackwell sm_120 qualifies).
2. **VRAM conflict** — the Qwen3.8 serve currently holds ~67 GB/GPU. It must be stopped; nothing can
   load alongside it. (Stopping also frees ~200 GB host RAM needed for the load.)
3. Prior GLM-5.2 TP2 NCCL hangs — **already root-caused/fixed** (A2): `NCCL_P2P_DISABLE=1` is proven
   on this host by the running Qwen TP2 serve. All stages below set it.
4. The BF16/noPE quirks: KV-cache layout envs (`VLLM_KV_CACHE_LAYOUT=HND`,
   `VLLM_SSM_CONV_STATE_LAYOUT=DS`) are only required for PD-disaggregation; single-instance can omit,
   but they're harmless to set.

## 4. Power-trip risk assessment (channel G / MM4)

Current CE baseline (this boot, `ras-mc-ctl --summary`): **channel#6 = 1367 + 934 (rows 0/1),
channel#7 = 310, channel#5 = 46+64** — the marginal DIMM is already accumulating CEs *while idle*.

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

### Stage 0 — runtime pull + offline check (no GPU, no load) — zero fabric risk
```bash
docker pull vllm/vllm-openai:glm53-flash          # fallback: :nightly
bash recipe/serve-glm-53-flash.sh --check         # asserts Glm5Next arch + FlashInfer >=0.6.17 in image
```

### Stage 1 — dummy-weight TP2 smoke (no disk read, no real weights) — low risk
Proves: arch registers, TP2 doesn't NCCL-hang (with `NCCL_P2P_DISABLE=1`), sparse-MLA backend
initializes on sm_120, offload allocator fits. Same isolation trick as the `--load-format dummy`
test in Instance notes. Requires Qwen serve **stopped**.
```bash
bash recipe/serve-glm-53-flash.sh --dummy         # load-format dummy, ctx 4096, eager
```

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

## 6. Verdict

Servable here **in principle** (runtime now exists as an official image, Blackwell sparse-MLA is
supported upstream, 1 TiB RAM covers the offload), but expect **~1–2 tok/s, ~4–8 tok/s best case
with MTP** — a "does it work" tier, not a daily driver. And it is the **maximal sustained stress on
the failing channel-G DIMM**: capture + EDAC watch armed and Qwen serve stopped are hard gates.
Recommended order: DIMM RMA first; this recipe second.
