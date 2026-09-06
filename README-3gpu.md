# System Capability Diagnostic — LLM Inference (Speculated: 3-GPU Build)

Date: 2026-09-03
Variant: assumes a third identical **NVIDIA RTX PRO 5000 72GB Blackwell** is added to the existing 2-GPU build.
Scope: revised hardware/inference-capability assessment; the primary baseline (2-GPU assumptions) is in `README.md`.

---

## 1. GPU (primary LLM compute)

| Property | GPU 0 | GPU 1 | GPU 2 (added) |
|---|---|---|---|
| Model | NVIDIA RTX PRO 5000 72GB Blackwell | identical | identical (spare) |
| VRAM total | 73,415 MiB (72 GB) | 72 GB | 72 GB |
| Compute capability | 12.0 (Blackwell) | 12.0 | 12.0 |
| Driver | 580.173.02 | same | same |
| CUDA supported | 13.0 | 13.0 | 13.0 |
| Peak power cap | 300 W | 300 W | 300 W |

**Key numbers**
- **Total combined VRAM: ~220 GB** (3 × 72 GB, plus per-card ~1.4 GB stubs).
- Blackwell (`sm_120`) → native **FP8 and NVFP4** are the highest-performance formats here; also FP16/BF16/INT8.
- With 3 cards you can run **distributed (tensor/pipeline/model) parallel** serving, letting one vLLM process span all three GPUs (plus a better path to keep a 4-bit spec-draft on a separate card, as is used today).

## 2. CPU

| Property | Value |
|---|---|
| Model | AMD EPYC 7663 56-Core |
| Cores / threads | 56 cores / 112 threads |
| Sockets | 1 |
| Base/max clock | ~1.5 / 3.54 GHz |
| NUMA nodes | 4 |

With a 3rd GPU the CPU stays the same but becomes relatively **less** of a bottleneck vs. VRAM. CPU offload (`--cpu-offload-gb`) remains a fallback for models that exceed VRAM. Note: 3 GPUs sharing one EPYC means PCIe/NVLink topology matters — check that GPUs are on distinct root ports; otherwise bandwidth to the 3rd card can cap throughput.

## 3. Memory (host RAM)

| Property | Value |
|---|---|
| Total | 1.0 TiB |
| Available | ~925 GiB |

Unchanged — 1 TiB is ample; supports RAM offload, huge KV cache, and multi-model staging.

## 4. Storage

| Mount | Type | Size | Used | Avail |
|---|---|---|---|---|
| `/trunk/ai` | ZFS | 17 T | 14 T (82%) | 3.1 T |
| `/buffer` | LVM (NVMe 990 EVO) | 1 T | 250 G | 706 G |
| `/` (root, NVMe 980 Pro LVM) | ext4 | 512 G | 172 G | 310 G |
| `/trunk` (ZFS root) | ZFS | 3.1 T | ~0 | 3.1 T |

**Constraint to watch:** `/trunk/ai` still 82% full (~3.1 T free). The added GPU raises *VRAM* capacity, not storage — multi-100-GB model dumps (e.g. 397B NVFP4 at 223 GB, or larger) still need storage room and may require pruning existing dumps.

## 5. Current running model context (for reference)

- `llama-server` in Docker container `llama-qwen` (image `llamacpp:main`), port **8081**.
- Model: **Qwen3.8-27B-BF16** (51 GB GGUF) + draft/MTP model.
- VRAM footprint ~72.7 GB (single 27B dense BF16 mostly on one card).
- With a 3rd GPU this current model would easily fit on one card with large headroom on the other two.

## 6. Inference provider software available

| Tool | Status |
|---|---|
| vLLM images | `vllm/vllm-openai:latest`, `:qwen38`, `:minimax-m3` (present) |
| llama.cpp images | `llamacpp:main`, `llamacpp-minimax:latest/uns` (current) |
| CUDA base/devel | `nvidia/cuda:13.0.0-base/devel-ubuntu24.04` |
| NVIDIA Container Toolkit | 1.20.0 (nvidia runtime in `/etc/docker/daemon.json`) |
| Native `vllm` | not installed (only via Docker) |
| Ollama | not present |

**Different-provider recommendation stays:** vLLM (or TRT-LLM) — native Blackwell FP8/NVFP4 and multi-GPU tensor/pipeline parallelism across 3 devices.

## 7. VRAM budget analysis (3-GPU)

Usable combined VRAM with 3 cards ≈ **200–205 GB** after CUDA context, activations, and KV cache (~1.15–1.3× weight multiplier).

| Weight precision | Bytes/param | One 72 GB card | 3×72 GB (~200 GB usable) |
|---|---|---|---|
| BF16/FP16 | 2 | ~25–28 B | ~85–95 B |
| FP8 | 1 | ~50–62 B | ~170–195 B |
| NVFP4 / Q4 | 0.5 | ~100–140 B | ~340–400 B (MoE) / ~180–200 B dense |
| Q8/Q8_K_M | 1 | ~50–60 B | ~170–195 B |

## 8. Which candidate models fit with 3 GPUs

Quantized weight size vs. ~200 GB usable budget (single vLLM serving process across 3 GPUs):

| Model (format) | On-disk size | Fit (3×72 GB) | Notes |
|---|---|---|---|
| Qwen-AgentWorld-35B-A3B GGUF UD-Q8_K_XL | 36 GB | ✅ Comfortable (1 GPU + headroom) | MoE, fast |
| Qwen3.8-27B NVFP4 | 22 GB | ✅ Comfortable (1 GPU + headroom) | |
| Qwen3.8-27B FP8 | 29 GB | ✅ Comfortable (1 GPU + headroom) | |
| Qwen3.8-Flash-Next GGUF UD-Q4 | 104 GB | ✅ Fits on 2 GPUs w/ headroom | |
| DeepSeek-V4-Flash NVFP4 | 150 GB | ✅ Now comfortable on 3 GPUs | Previous bottleneck eliminated |
| GLM-5.3-Flash GGUF UD-Q4 | 186 GB | ⚠️ Fits, but tight (≈186 GB weights) | Verify KV budget; may need reduced context |
| nvidia/GLM-5.2-NVFP4 | 426 GB | ❌ Still too large | Multi-node / offload only |
| Qwen3.5-397B-A17B NVFP4 | 223 GB | ⚠️ Borderline (>200 GB weights) | Needs CPU/RAM offload or reduced precision/context |
| MiniMaxAI/MiniMax-M3 | 664 GB | ❌ | — |
| DeepSeek-V4-Pro | 786 GB | ❌ | — |
| Solar-Open2-250B | 467 GB | ❌ | — |
| Inkling | ~2 TB (108 shards) | ❌ | — |

**New capability unlocked by the 3rd GPU:**
- **~150 GB-class models become comfortable** (e.g. `DeepSeek-V4-Flash NVFP4`) — previously "just fits / tight" on 2 GPUs.
- **~180–200 GB-class** models (GLM-5.3-Flash Q4, borderline 397B NVFP4) now within reach with modest KV/context tuning or offload.
- Dense models up to **~90 B in BF16** or **~180 B in FP8/NVFP4**.
- Leaves room to keep a dedicated small **speculative-draft card** while the main model spans GPUs (as the current build does on 2 cards).

## 9. Recommendation summary (3-GPU)

- **New sweet spot:** ~70–150 B-class models at FP8/NVFP4, or large-MoE (150–200 B) split across 3 GPUs.
- **Best upgrades already downloaded:** `DeepSeek-V4-Flash NVFP4` (150 GB) and `GLM-5.3-Flash GGUF Q4` (186 GB) become runnable; `Qwen3.5-397B-A17B NVFP4` becomes borderline-runner with offload.
- **Best-quality-per-VRAM still:** `Qwen3.8-27B-NVFP4` (22 GB, 1 card) — great as a fast/lean tier alongside a large model.
- **Still out of reach (>400 GB):** GLM-5.2-NVFP4 (426 GB), MiniMax-M3 (664 GB), DeepSeek-V4-Pro (786 GB), Solar-Open2-250B (467 GB), Inkling (~2 TB). Would require multi-node or heavy offload.
- **Storage limitation unchanged:** ~3.1 T free on `/trunk/ai`; the 3rd GPU buys VRAM, not disk — large new dumps need storage space.
- **Physical/thermal consideration:** 3 × 300 W GPUs ⇒ ~900 W peak GPU draw + CPU/system; confirm PSU and cooling before adding the card.

## 10. Next-step plan (3-GPU, vLLM)

1. Install the 3rd GPU, verify `nvidia-smi` shows 3 devices and **check PCIe topology** (NUMA/pcie locality) for good inter-GPU bandwidth.
2. Pick provider/model, e.g. **DeepSeek-V4-Flash NVFP4 @ vLLM** (`vllm/vllm-openai` image) or **GLM-5.3-Flash GGUF** for a more aggressive quant.
3. Configure serving with `--tensor-parallel-size 3` (or 2 + separate draft card).
4. Define container via **docker-compose** (preferred over raw `docker run`), replacing port 8081.
5. Stop/remove current `llama-qwen` container (restart policy `unless-stopped` — must not auto-restart).
6. Validate: `/v1/models` reachable, benchmark tokens/s & first-token latency, and measure VRAM headroom on all 3 cards.
