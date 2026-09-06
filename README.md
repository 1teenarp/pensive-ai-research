# System Capability Diagnostic — LLM Inference

Date: 2026-09-03
Scope: hardware/inference-capability assessment to inform what models can be run locally, and (in follow-up) a move away from llama.cpp to a different provider.

---

## 1. GPU (primary LLM compute)

| Property | GPU 0 | GPU 1 |
|---|---|---|
| Model | NVIDIA RTX PRO 5000 72GB Blackwell | NVIDIA RTX PRO 5000 72GB Blackwell |
| VRAM total | 73,415 MiB (72 GB) | 73,415 MiB (72 GB) |
| VRAM free | ~5.1 GB (in use by current model) | ~67.9 GB |
| Compute capability | 12.0 (Blackwell) | 12.0 (Blackwell) |
| Driver | 580.173.02 | same |
| CUDA supported | 13.0 | 13.0 |
| Peak power cap | 300 W | 300 W |

**Key numbers**
- **Total combined VRAM: ~146.8 GB** (2 × 72 GB, plus the ~1.4 GB stub per card not exposed in the 72 GB query).
- Blackwell (`sm_120`) → full support for **FP8 and NVFP4 (NVIDIA 4-bit float)** native formats. These are the formats vLLM/NVIDIA optimized for Blackwell and give the best performance/VRAM tradeoff here. Also supports FP16/BF16/FP8/INT8.
- Both GPUs already in use by the current `llama-server` container (GPU0 ~67.8 GB, GPU1 ~5.0 GB).

## 2. CPU

| Property | Value |
|---|---|
| Model | AMD EPYC 7663 56-Core |
| Cores / threads | 56 cores / 112 threads |
| Sockets | 1 |
| Base/max clock | ~1.5 / 3.54 GHz |
| NUMA nodes | 4 (0-3) |

CPU matters mainly for: prompt prefill on CPU-only paths, system/tokenizer overhead, and **NOS (No-GPU-offload) not needed here** — the GPUs dominate. vLLM CPU offload (`--cpu-offload-gb`) can extend VRAM for very large models as a fallback.

## 3. Memory (host RAM)

| Property | Value |
|---|---|
| Total | 1.0 TiB |
| Available | ~925 GiB |

Plenty of RAM. Realistically not the constraint; can RAM-offload or use huge KV caches. 1 TiB also means models kept in RAM can be mmap-loaded.

## 4. Storage

| Mount | Type | Size | Used | Avail |
|---|---|---|---|---|
| `/trunk/ai` | ZFS | 17 T | 14 T (82%) | 3.1 T |
| `/buffer` | LVM (NVMe 990 EVO) | 1 T | 250 G | 706 G |
| `/` (root, NVMe 980 Pro LVM) | ext4 | 512 G | 172 G | 310 G |
| `/trunk` (ZFS root) | ZFS | 3.1 T | ~0 | 3.1 T |

**Constraint to watch:** `/trunk/ai` is 82% full with ~3.1 TB free. Very large model dumps (e.g. Multi-TB) will not fit. The NVMe-backed paths `/buffer` and root have more room but are much smaller. Models are staged there (`data-root: /buffer/docker` also hosts container images).

## 5. Current running model context

- `llama-server` in Docker container `llama-qwen` (image `llamacpp:main`), port **8081**.
- Model: **Qwen3.8-27B-BF16** (51 GB GGUF) + draft/MTP model.
- VRAM footprint: **~72.7 GB** (67.8 GB GPU0 + 5.0 GB GPU1). Single 27B dense BF16 model nearly fills one card; 2-GPU speculative decode splits draft model.

## 6. Inference provider software available

| Tool | Status |
|---|---|
| vLLM images | `vllm/vllm-openai:latest`, `:qwen38`, `:minimax-m3` (present) |
| llama.cpp images | `llamacpp:main`, `llamacpp-minimax:latest/uns` (current) |
| CUDA base/devel | `nvidia/cuda:13.0.0-base/devel-ubuntu24.04` |
| NVIDIA Container Toolkit | 1.20.0 (nvidia runtime configured in `/etc/docker/daemon.json`) |
| Native `vllm` | not installed on host (only via Docker) |
| Ollama | not present |

vLLM is the most turnkey "different provider" option already available as an image and is a natural fit on Blackwell (best FP8/NVFP4 support).

## 7. VRAM budget analysis

Rule of thumb for vLLM/llama.cpp (weights ≈ params × bytes/param; add ~1.25–1.3× for KV cache + activations + CUDA context):

| Weight precision | Bytes/param | Usable model size on ONE 72 GB card | On 2×72 GB (tensor- / pipeline-parallel) |
|---|---|---|---|
| BF16/FP16 | 2 | ~25–28 B | ~55–60 B |
| FP8 | 1 | ~50–62 B | ~110–125 B |
| NVFP4 / Q4 | 0.5 | ~100–140 B | ~220–290 B |
| Q8/Q8_K_M | 1 | ~50-60 B | ~110-120 B |

Practical usable **combined VRAM ≈ 128–135 GB** after context/activation/cuda overhead, assuming a single serving process.

## 8. Which candidate models fit (from those already downloaded)

Quantized weight size (excl. KV overhead) and fit verdict for a 2×(72 GB) single-process serve:

| Model (format) | On-disk size | VRAM fit (single proc) | Notes |
|---|---|---|---|
| Qwen-AgentWorld-35B-A3B GGUF UD-Q8_K_XL | 36 GB | ✅ Comfortable on 1 GPU | MoE, fast |
| Qwen3.8-27B NVFP4 | 22 GB | ✅ Comfortable on 1 GPU | High quality/VRAM ratio (Blackwell NVFP4) |
| Qwen3.8-27B FP8 | 29 GB | ✅ Comfortable on 1 GPU | |
| Qwen3.8-Flash-Next GGUF UD-Q4 | 104 GB (4×47G shards) | ⚠️ Fits w/ 2 GPUs (split) | Flash-class; verify it's the full 4-shard set |
| DeepSeek-V4-Flash NVFP4 | 150 GB | ⚠️ Just fits on 2 GPUs (tight) | Needs careful KV/parallel config |
| GLM-5.3-Flash GGUF UD-Q4 | 186 GB (6 shards) | ❌ Too large (>135 GB usable) | Not single-config eligible |
| nvidia/GLM-5.2-NVFP4 | 426 GB | ❌ Too large | Multi-node/offload only |
| Qwen3.5-397B-A17B NVFP4 | 223 GB | ❌ Too large (~200 GB weights) | Offload/partial only |
| MiniMaxAI/MiniMax-M3 | 664 GB | ❌ | — |
| DeepSeek-V4-Pro | 786 GB | ❌ | — |
| Solar-Open2-250B | 467 GB | ❌ | — |
| Inkling | ~2 TB (108 shards) | ❌ | — |

## 9. Recommendation summary

- **Sweet spot for this machine:** 27–40 B dense models at FP8/NVFP4 (one GPU), or ~70 B-class / large-MoE models quantized **across both GPUs**.
- **Best "different provider" path:** **vLLM** (image already present, native Blackwell FP8/NVFP4 support) or **TRT-LLM**. No need for llama.cpp.
- **Best-quality-per-VRAM candidates already downloaded:** `Qwen3.8-27B-NVFP4` (22 GB), `Qwen-AgentWorld-35B-A3B` (36 GB, MoE), `Qwen3.8-Flash-Next` (needs 2 GPUs).
- **Avoid** unless using CPU/RAM offload or multi-node: 300 B+ or full-PRECISION huge dumps (MiniMax-M3, DeepSeek-V4-Pro, Solar-Open2-250B, Inkling).
- **Storage note:** ~3.1 TB free on `/trunk/ai`; large new models (especially >300 GB) may not fit without pruning existing dumps.

## 10. Next-step plan (moving off llama.cpp)

1. Pick target model + provider, e.g. **Qwen3.8-27B-NVFP4 @ vLLM** (`vllm/vllm-openai:qwen38` image exists — verify tag matches the NVFP4 files).
2. Define GPU mapping (both cards; `--tensor-parallel-size 2` if >72 GB needed).
3. Define serving container via **docker-compose** (preferred over raw `docker run`), port + alias to replace 8081.
4. Stop/remove current `llama-qwen` container (restart policy `unless-stopped` — must not be left to auto-restart).
5. Validate: `/v1/models` reachable, prompt-benchmark tokens/s & latency, measure VRAM headroom.
