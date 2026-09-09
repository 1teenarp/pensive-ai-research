# Model Catalog & Recipes — "pensive" build

Date: 2026-09-06 (RAM figure below is nameplate; **live RAM is currently reduced for DIMM isolation
testing** — see `SYSTEM-SPEC.md`'s hardware-status banner, do not assume 1 TiB is available right now)
Purpose: a catalog of models present under `/trunk/ai/huggingface/models/`, with the **serving
settings that are known to work (or worked in the past)** on this box (2× RTX PRO 5000 72 GB Blackwell
= ~144 GB VRAM, 1 TiB RAM nameplate, EPYC 7663 56c, single socket 4-NUMA (NPS4)).

> Recipe launchers live in the sibling `recipe/` folder. The launch script that reproduces the
> currently-working server is `recipe/serve-qwen38-flash-next-nvfp4.sh`.

---

## Quick reference — the models we've actually served

| Model | Format | On-disk | VRAM needs | Status | Settings |
|---|---|---|---|---|---|
| **nvidia/Qwen3.8-Flash-Next-NVFP4** | NVFP4 MoE (125B total / 6B act, +51B ngram) | 124 GB | ~54 GB/GPU + KV (PLE to RAM) | ✅ Proven, not running now (superseded by FP8) | See "Recipe A" below |
| **unsloth/Qwen3.8-27B-NVFP4** | NVFP4 dense (27B) | 22 GB | ~1 GPU comfortable | ✅ **Worked in past** (llama.cpp/vLLM) | See "Recipe B" below |
| **Qwen/Qwen3.8-Flash-Next-FP8** | FP8 MoE (same arch as Recipe A, heavier weights) | 173 GB | ~60.5 GB/GPU weight (8GB/wkr offloaded) + KV | ✅ **CURRENT — serving, 256K ctx, ~20-24 tok/s** | See "Recipe C" below |

---

## Recipe A — Qwen3.8-Flash-Next-NVFP4 (current, native context, CUDA graphs)

**Model:** `nvidia/Qwen3.8-Flash-Next-NVFP4` — `Qwen4ExpForConditionalGeneration`
(48 layers, hidden 2560, 24 heads / 2 KV heads, 512 experts / 10 per tok, native **262,144** context,
51B n-gram/PLE table, MTP). Routed experts NVFP4; attention/shared BF16; MTP+ngram FP8 (separate
`model-fp8-mtp-ple.safetensors`).

**Working settings (launched via `recipe/serve-qwen38-flash-next-nvfp4.sh`):**
- Image: `vllm/vllm-openai:qwen38-flash-next-patched` (base `qwen38-flash-next`, patched
  `_get_ple_embedding_quant_method` to select the FP8 PLE method under NVFP4).
- TP2 (`--tensor-parallel-size 2`), `--quantization modelopt`.
- `--max-model-len 262144 --max-num-seqs 2` (native context; need low concurrency for KV budget).
- `--gpu-memory-utilization 0.90 --disable-custom-all-reduce`.
- `--max-parallel-loading-workers 1` (serialize weight-load burst).
- **CUDA graphs ON** (do NOT pass `--enforce-eager` — gives ~4× decode).
- Env: `NCCL_P2P_DISABLE=1` (cross-NUMA, no NVLink), `VLLM_PLE_CPU_OFFLOAD=1`,
  `VLLM_QWEN38_PLE_FP8_SCALE=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Docker opts: `--cap-add SYS_PTRACE`, `--security-opt seccomp=unconfined/apparmor=unconfined`
  (for the PLE-offload `pidfd` handshake).
- Host: GPU power cap `nvidia-smi -pl 250` (reduces fabric draw on this marginal DIMM).
- **While DIMM isolation testing is in progress** (see `SYSTEM-SPEC.md` banner): the script also
  auto-detects a memory-less GPU-local NUMA node and sets `NCCL_CUMEM_HOST_ENABLE=0` when needed, to
  avoid a `cuMemCreate` segfault at NCCL init. Launch via the script, not a hand-copied command, so this
  stays correct as DIMMs are added back.

**Measured performance (native context):**
- **~39–55 tok/s** decode with CUDA graphs (~4× faster than `--enforce-eager` ~12 tok/s).
- GPU VRAM during serve: ~65 GB/GPU (262k KV), util 85–91% during decode.

**Hardware caveat:** this model's heavy both-GPU weight-load + PLE host-RAM offload has triggered the
**data-fabric sync-flood reset** multiple times (see `power-trip-instances.md` Instances 2/4/5) on the
marginal DIMM (channel G / MM4). The burst-reduction flags + 250 W cap + PLE offload (**not** `--enforce-eager`)
keep it stable. Expected final fix = memtest + RAM downclock + GPU→Gen3.

---

## Recipe B — Qwen3.8-27B (worked in past)

**Model:** `unsloth/Qwen3.8-27B-NVFP4` (22 GB) — `Qwen3_5ForConditionalGeneration`
(64 layers, hidden 5120, 24 heads / 4 KV heads, native 262,144 context), `compressed-tensors`
quantization. Also on disk: `Qwen/Qwen3.8-27B` (BF16, 52 GB), `Qwen/Qwen3.8-27B-FP8` (29 GB),
`ggml-org/Qwen3.8-27B-GGUF` (57 GB), and abliterated variants.

**Past usage** (from `README-ramoffload-research.md` / earlier work):
- Served via llama.cpp (`Qwen3.8-27B-BF16`, 51 GB GGUF) + draft/MTP model, ~72.7 GB VRAM footprint
  (67.8 GB GPU0 + 5 GB GPU1), using a 2-GPU **speculative decode** split.
- The **NVFP4 (22 GB)** and **FP8 (29 GB)** variants fit comfortably on a **single GPU** — the ideal
  fast/lean tier alongside a larger model.

**Recommended (vLLM, single GPU or TP2):**
```
vllm serve /trunk/ai/huggingface/models/unsloth/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 1 --quantization modelopt --max-model-len 131072 \
  --gpu-memory-utilization 0.90 --enable-auto-tool-choice --tool-call-parser qwen3_xml
```
(For a 27B dense model TP1 is fine — it fits one 72 GB card with headroom; no PLE/offload needed.)

---

## Recipe C — Qwen3.8-Flash-Next-FP8 (current, native 256K context, spec decode)

**Model:** `Qwen/Qwen3.8-Flash-Next-FP8` — same `Qwen4ExpForConditionalGeneration` architecture as
Recipe A (48 layers, 512 experts/10-per-tok, native 262,144 ctx, 51B-param n-gram/PLE table), just
native FP8 (1 byte/param) instead of NVFP4 (0.5 byte/param) — chosen over NVFP4 for precision, at the
cost of ~2x the GPU-resident weight footprint. Launcher: `recipe/serve-qwen38-flash-next-fp8.sh`
(defaults already match the config below — see the script header for the full "what was tried" log).

**Working settings:**
- Same patched image as Recipe A (`vllm/vllm-openai:qwen38-flash-next-patched`) — this checkpoint's
  PLE table has the identical layout the image's FP8-PLE-selector patch targets, confirmed via
  config.json/index.json inspection, no new patching needed.
- TP2, `--cpu-offload-gb 8` per worker (shaves GPU-resident weight to make room for full-context KV —
  without it, `--gpu-memory-utilization 0.95` alone isn't enough headroom at 262144 ctx).
- `--max-model-len 262144 --max-num-seqs 1` (native context; tight VRAM margin needs low concurrency).
- `--gpu-memory-utilization 0.95` (higher than Recipe A's 0.90 — margin is much tighter here).
- **CUDA graphs ON** (do NOT pass `--enforce-eager`) — smaller win than Recipe A's ~4x (only ~1.4x,
  11.5 vs ~8 tok/s), itself evidence the bottleneck is cross-GPU TP communication over this box's
  no-NVLink/no-P2P interconnect, not kernel-launch overhead.
- **MTP speculative decoding**: `--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`.
  Confirmed working despite the comm-bound TP setup (~65-70% avg per-token acceptance). **5 crashes**
  (`QSA ring capacity 12 must divide the attention block size 1616` — the attention block size is
  computed dynamically from mamba/attention page-size alignment, not a simple formula; 4 is the
  tested ceiling, don't assume higher values work without retesting).
- Env: `NCCL_P2P_DISABLE=1`, `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_QWEN38_PLE_FP8_SCALE=1`,
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `NCCL_CUMEM_HOST_ENABLE=0` (auto-detected — GPU0's local NUMA
  node is memory-less during the current DIMM isolation testing).
- **Do NOT** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if also using
  `--kv-offloading-size` — the two are incompatible (vLLM raises a pydantic ValidationError). Not used
  in the final config (see "tried and rejected" below).
- **Pipeline parallelism (TP=1 PP=2) is a dead end for this model** — `VLLM_PLE_CPU_OFFLOAD` has an
  explicit vLLM guard against PP, since the mandatory PLE offload doesn't support it.
- **KV-cache CPU offload (`--kv-offloading-size`) doesn't extend a single sequence's live context** —
  it's for cross-request prefix-cache reuse, not paging one request's active KV beyond GPU capacity.
  Didn't help reach 256K context; weight offload (`--cpu-offload-gb`) is the right lever for that.

**Measured performance (native 262144 context, real weights, steady-state via vLLM `/metrics`):**
- **~20-24 tok/s** with CUDA graphs + spec decode (vs. ~8 tok/s eager baseline, ~11.5 tok/s CUDA
  graphs alone) — a ~2.5-3x overall improvement. **Judge throughput only after a couple of warm-up
  requests** — the first 1-2 requests after a fresh restart are much slower (one-time lazy compilation
  on the spec-decode path), not representative of steady state.
- GPU-resident weight: 54.48 GiB/rank (with the 8GB/worker offload applied); KV cache: ~8.9-10.5 GiB
  available per GPU at `gpu-memory-utilization 0.95`, comfortably covering the ~3.28 GiB/GPU that full
  262144-token context needs.

---

## Catalog — all models present under /trunk/ai/huggingface/models/

Sizes are on-disk. "Fit" is based on ~128–135 GB usable VRAM (single process) unless noted.

| Model (path) | Size | Type / notes | Range | Fit / status |
|---|---|---|---|---|
| **Qwen/Qwen-AgentWorld-35B-A3B** | 65 GB | MoE 35B/3B | tiny | ✅ Fits 1-2 GPU (fast MoE) |
| **Qwen/Qwen3.8-27B** | 52 GB | dense 27B BF16 | small | ✅ 1 GPU (tight) |
| **Qwen/Qwen3.8-27B-FP8** | 29 GB | dense 27B FP8 | small | ✅ 1 GPU comfortable |
| **unsloth/Qwen3.8-27B-NVFP4** | 22 GB | dense 27B NVFP4 | small | ✅ 1 GPU comfortable (Recipe B) |
| **unsloth/Qwen3.6-35B-A3B-NVFP4-Fast** | 22 GB | MoE 35B/3B NVFP4 | small | ✅ 1 GPU |
| **Qwen/Qwen3.5-122B-A10B-FP8** | 115 GB | MoE FP8 | small | ⚠️ 2 GPU (tight) |
| **Qwen/Qwen3.5-122B-A10B** | 234 GB | MoE BF16 | large | ❌ needs split/offload |
| **Qwen/Qwen3.5-397B-A17B-FP8** | 361 GB | MoE FP8 | large | ❌ offload only |
| **nvidia/Qwen3.5-397B-A17B-NVFP4** | 223 GB | MoE NVFP4 | large | ⚠️ offload/RAM (ran at ~1 tok/s) |
| **Qwen/Qwen3-Coder-Next** | 149 GB | MoE coder | large | ⚠️ 2 GPU / offload |
| **Qwen/Qwen3-Coder-Next-FP8** | 75 GB | MoE coder FP8 | small | ✅ 2 GPU |
| **nvidia/Qwen3.8-Flash-Next-NVFP4** | 124 GB | MoE 125B/6B, 262k ctx | large | ✅ Proven (Recipe A) — not running now, superseded by FP8 (Recipe C) for precision |
| **Qwen/Qwen3.8-Flash-Next-FP8** | 173 GB | MoE FP8, 262k ctx | large | ✅ **CURRENT (Recipe C)** — 2 GPU + 8GB/wkr weight offload |
| **Qwen/Qwen3.8-Flash-Next** | 336 GB | MoE BF16 | large | ❌ offload only |
| **DeepSeek-V4-Flash / -FP variants** | 146–153 GB | MoE | large | ⚠️ 2 GPU (tight) |
| **nvidia/DeepSeek-V4-Flash-NVFP4** | 150 GB | MoE NVFP4 | large | ⚠️ just fits 2 GPU |
| **deepseek-ai/DeepSeek-V4-Pro** | 786 GB | MoE | huge | ❌ multi-node/offload |
| **nvidia/DeepSeek-V4-Pro-NVFP4** | 819 GB | MoE NVFP4 | huge | ❌ |
| **nvidia/GLM-5.2-NVFP4** | 426 GB | MoE NVFP4, sparse-attn | huge | ❌ (no Blackwell sparse-MLA backend) |
| **zai-org/GLM-5.3-Flash** | 305 GB | MoE 320B/18B FP8, KDA+sparse-MLA, 1M ctx | huge | ⚠️ RAM-offload only (~1–2 tok/s TP2+MTP; needs vllm glm53-flash image) — see `GLM-53-FLASH-RECIPE.md` |
| **madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid** | 336 GB | MoE | huge | ❌ |
| **nvidia/Kimi-K2.7-Code-NVFP4** | 543 GB | MoE | huge | ❌ |
| **moonshotai/Kimi-K2.7-Code** | 546 GB | MoE | huge | ❌ |
| **moonshotai/Kimi-K3** | 1.4 TB | MoE | huge | ❌ |
| **nvidia/MiniMax-M3-NVFP4** | 232 GB | MoE | large | ❌ offload only |
| **MiniMaxAI/MiniMax-M3** | 664 GB | MoE | huge | ❌ |
| **MiniMaxAI/MiniMax-Music3** | 54 GB | audio | small | ✅ |
| **nvidia/Llama-3.1-Nemotron-70B-Instruct-HF** | 132 GB | dense 70B BF16 | large | ⚠️ 2 GPU (offload) |
| **poolside/Laguna-S-2.1-NVFP4** | 66 GB | MoE | small | ✅ 1-2 GPU |
| **upstage/Solar-Open2-250B** | 467 GB | MoE | huge | ❌ |
| **thinkingmachines/Inkling** | 1.8 TB | MoE | huge | ❌ |
| **tencent/Hy3** | 557 GB | MoE | huge | ❌ |
| **nvidia/LocateAnything-3B** | 7.3 GB | small | tiny | ✅ |
| **Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice** | 4.2 GB | TTS | tiny | ✅ |
| **baidu/Unlimited-OCR** | 6.4 GB | OCR | tiny | ✅ |
| **Wan-AI/Wan-Dancer-14B** | 80 GB | video | small | ✅ 2 GPU |
| **(GGUF variants)** ggml-org/*, HauhauCS/*, heretic-org/*, huihui-ai/*, orcarouter/* | var | GGUF | var | ✅ llama.cpp (Q8/Q4) |
| **unsloth/Qwen3.8-Flash-Next-GGUF** | var | GGUF | large | ⚠️ llama.cpp split |

> "✅" = runs on this box; "⚠️" = tight / needs offload or reduced context; "❌" = too large for a
> single-process 2×72 GB serve (would need multi-node or CPU/RAM offload).

---

## Recurring hardware caveat

The box has a **recurring data-fabric sync-flood reset** (`0x08000a00`, see
`power-trip-instances.md`) traced to a failing/marginal DIMM on **channel G / slot MM4**, aggravated by
high host-RAM/fabric traffic (large MoE weight-loads, PLE host-RAM offload, RAM+VRAM offload serving).
- **Mitigations in use:** burst-reduction flags, PLE CPU offload, GPU power cap (250 W), `NCCL_P2P_DISABLE=1`.
- **Status (2026-09-08):** the suspect DIMM and its NUMA-node half are **physically removed** for
  isolation/RMA testing (memtest86 clean on the remaining ~512 GB) — see `SYSTEM-SPEC.md` and
  `power-trip-diagnosis.md` for the live status and reinstall plan. Not yet the final fix.
- Keep `powertrip-capture` running (auto-restarts) so any future trip is captured.
