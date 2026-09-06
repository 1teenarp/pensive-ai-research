# System Specifications — build "pensive"

Date: 2026-09-03 (last verified against live hardware)
Scope: machine hardware/inference-capability assessment that informs what models can run locally on
this build. The system's actual, live spec snapshot lives in `power-debug-collect.sh` output; this file
is the curated reference for planning.

> Hardware caution: this build has a known, recurring **data-fabric sync-flood reset** (`0x08000a00`)
> traced to a marginal DIMM on channel G / slot MM4. See `power-trip-diagnosis.md` and
> `power-trip-instances.md`. Keep `powertrip-capture` running to capture events.

---

## 1. GPU (primary LLM compute)

| Property | GPU 0 | GPU 1 |
|---|---|---|
| Model | NVIDIA RTX PRO 5000 72GB Blackwell | NVIDIA RTX PRO 5000 72GB Blackwell |
| VRAM total | 73,415 MiB (72 GB) | 73,415 MiB (72 GB) |
| Compute capability | 12.0 (Blackwell) | 12.0 (Blackwell) |
| Driver | 580.173.02 | same |
| CUDA supported | 13.0 | 13.0 |
| Peak power cap | 300 W (tuned down to ~250 W for stability) | 300 W (tuned down to ~250 W for stability) |

**Key numbers**
- **Total combined VRAM: ~146.8 GB** (2 × 72 GB, plus the ~1.4 GB stub per card not exposed in the 72 GB query).
- Blackwell (`sm_120`) → full support for **FP8 and NVFP4 (NVIDIA 4-bit float)** native formats. These are
  the formats vLLM/NVIDIA optimized for Blackwell and give the best performance/VRAM tradeoff here.
  Also supports FP16/BF16/FP8/INT8.
- **GPU power caps are lowered to ~250 W** (`nvidia-smi -pl 250`) as a defensive measure to reduce
  peak draw/heat; note this is *not* the root-cause fix for the memory fault (see the diagnosis docs),
  it's a secondary margin.

## 2. CPU

| Property | Value |
|---|---|
| Model | AMD EPYC 7663 56-Core |
| Cores / threads | 56 cores / 112 threads |
| Sockets | 1 |
| Base/max clock | ~1.5 / 3.54 GHz |
| NUMA nodes | 4 (0-3) — **NPS4 mode** (each node has its own ~256 GB) |

CPU matters mainly for: prompt prefill on CPU-only paths, system/tokenizer overhead; the GPUs dominate.
vLLM CPU offload (`--cpu-offload-gb`) can extend VRAM for very large models as a fallback.

> **NUMA / topology note:** the two GPUs are wired to **different root complexes / NUMA nodes**
> (GPU0 on node 3, GPU1 on node 0), with **no NVLink**. This means no GPU↔GPU P2P → NCCL must run
> with `NCCL_P2P_DISABLE=1` (or `NCCL_P2P_LEVEL`). NPS1 would collapse the socket to 1 NUMA node but
> **does not fix P2P** and would interleave the failing channel G into all traffic. See
> `why-databric-syncflood-not-interceptable.md` and the topology discussion in the research docs.

## 3. Memory (host RAM)

| Property | Value |
|---|---|
| Total | 1.0 TiB (8 × 128 GB Micron DDR4-3200 8-rank RDIMM) |
| Available | ~925 GiB |

Plenty of RAM. Realistically not the capacity constraint; can RAM-offload or use huge KV caches. 1 TiB
also means models kept in RAM can be mmap-loaded. **But** the memory subsystem is the source of the
recurring fault: the marginal DIMM on **channel G (slot MM4)** throws corrected-ECC that escalates to an
uncorrectable → sync-flood reset under heavy load.

## 4. Storage

| Mount | Type | Size | Used | Avail |
|---|---|---|---|---|
| `/trunk/ai` | ZFS | 17 T | 14 T (82%) | 3.1 T |
| `/buffer` | LVM (NVMe 990 EVO) | 1 T | 250 G | 706 G |
| `/` (root, NVMe 980 Pro LVM) | ext4 | 512 G | 172 G | 310 G |
| `/trunk` (ZFS root) | ZFS | 3.1 T | ~0 | 3.1 T |

**Constraint to watch:** `/trunk/ai` is ~82% full with ~3.1 TB free. Very large model dumps (e.g.
multi-TB) will not fit. The NVMe-backed paths `/buffer` and root have more room but are much smaller.
Models are staged there (`data-root: /buffer/docker` also hosts container images). Note `/tmp` is
volatile across reboot; use `/buffer/powertrip/` or `/var/tmp` for persistent capture/logs.

## 5. Inference provider software available

| Tool | Status |
|---|---|
| vLLM (patched) | `vllm/vllm-openai:qwen38-flash-next-patched` — required for Qwen3.8-Flash-Next (issue #54765 fix) |
| vLLM images | `vllm/vllm-openai:latest`, `:qwen38`, `:minimax-m3`, `:v0.23.0` |
| llama.cpp images | `llamacpp:main`, `llamacpp-minimax:latest/uns` |
| CUDA base/devel | `nvidia/cuda:13.0.0-base/devel-ubuntu24.04` |
| NVIDIA Container Toolkit | 1.20.0 (nvidia runtime configured in `/etc/docker/daemon.json`) |
| Native `vllm` | not installed on host (only via Docker) |
| Ollama | not present |
| SGLang | `lmsysorg/sglang:dev-glm52-nvfp4` (dev; GLM-5.2 experiments) |

## 6. VRAM budget analysis

Rule of thumb for vLLM/llama.cpp (weights ≈ params × bytes/param; add ~1.25–1.3× for KV cache +
activations + CUDA context):

| Weight precision | Bytes/param | Usable model size on ONE 72 GB card | On 2×72 GB (tensor- / pipeline-parallel) |
|---|---|---|---|
| BF16/FP16 | 2 | ~25–28 B | ~55–60 B |
| FP8 | 1 | ~50–62 B | ~110–125 B |
| NVFP4 / Q4 | 0.5 | ~100–140 B | ~220–290 B |
| Q8/Q8_K_M | 1 | ~50-60 B | ~110-120 B |

Practical usable **combined VRAM ≈ 128–135 GB** after context/activation/CUDA overhead, assuming a
single serving process.

**Context vs. concurrency:** KV scales linearly. The box holds ~24 GiB of KV with PLE offloaded
(`gpu-memory-utilization 0.90`), so native 262k context requires low concurrency
(`--max-num-seqs 2`); higher concurrency only fits ~32k context. See `recipe/MODEL-CATALOG.md`.

## 7. Candidates / fit — see `recipe/MODEL-CATALOG.md`

The full per-model (size + fit verdict) table lives in `recipe/MODEL-CATALOG.md` (Recipe A & B with the
two actually-running models; all ~46 models present on `/trunk/ai`). The summary here: **28-70 B dense
at FP8/NVFP4 (one GPU), ~70 B-class / large MoE across both GPUs** is the sweet spot.
