# System Specifications — build "pensive"

Date: 2026-09-10; RAM/NUMA and driver state re-verified against live hardware **2026-09-15** (RAM
unchanged: 751 GiB across 4 nodes, ~543 GiB available). Re-verify with `numactl -H` / `free -h` before
relying on any figure in §3 — the RAM config has changed twice already this week and may change again
once the 2 bad DIMMs are RMA'd and replacements reinstalled.
Scope: machine hardware/inference-capability assessment that informs what models can run locally on
this build. The system's actual, live spec snapshot lives in `power-debug-collect.sh` output; this file
is the curated reference for planning.

> **Hardware status (2026-09-10): root cause found, 2 bad DIMMs physically removed, 6 of 8 reinstalled.**
> The isolation testing (see `power-trip-diagnosis.md`, `power-trip-instances.md`) identified **two**
> faulty sticks, not one: the original channel-G/MM4 suspect (was throwing memtest86+ errors) **and** a
> second stick found to have physically damaged PCB pads (chipped/missing small components) — silently
> bad, wouldn't have shown up as a clean memtest failure. Both are now pulled for RMA; the other 6
> known-good sticks are back in. **All 4 NUMA nodes have memory again** (uneven population — see §3),
> so the memory-less-GPU-local-node NCCL workaround (§2) no longer triggers on either GPU, though the
> auto-detecting scripts still carry it harmlessly for whenever DIMMs move again. Total live RAM is now
> ~751 GiB (6×128 GB), up from the ~499 GiB isolation-testing low point, still short of the 1 TiB
> nameplate until the 2 RMA replacements arrive and go in.
>
> **RESOLVED 2026-09-15 — NVIDIA driver/library version mismatch (found 2026-09-10).** Both sides now
> read **580.178.04** (`/proc/driver/nvidia/version` and `modinfo nvidia`), so GPU containers launch
> normally again. Kept here because the failure mode is worth recognising instantly:
> an apt upgrade (`nvidia-driver-580-open` 580.173.02 → 580.178.04, `Start-Date: 2026-09-10 09:59:20`,
> 19 min after that morning's post-DIMM-work reboot at 09:40:58) updated the userspace libraries while
> the **kernel module in memory stayed at the old 580.173.02** — nothing reloaded it. That broke
> `nvidia-smi` host-wide (`Failed to initialize NVML: Driver/library version mismatch`) and every GPU
> container (`nvidia-container-cli: initialization error: nvml error: driver/library version mismatch`,
> container stuck in `Created`, exit code 128) — **not specific to any one recipe**, though it reads
> like a model problem. Fixes, needing root: reboot (cleanest — loads the matching module),
> `rmmod`+`modprobe` with no GPU processes active, or pin the package back to the loaded version.
> **Always check this pair first when a container dies at init:**
> ```bash
> cat /proc/driver/nvidia/version        # loaded kernel module
> modinfo nvidia | grep '^version:'      # installed package  -> MUST MATCH
> ```

---

## 1. GPU (primary LLM compute)

| Property | GPU 0 | GPU 1 |
|---|---|---|
| Model | NVIDIA RTX PRO 5000 72GB Blackwell | NVIDIA RTX PRO 5000 72GB Blackwell |
| VRAM total | 73,415 MiB (72 GB) | 73,415 MiB (72 GB) |
| Compute capability | 12.0 (Blackwell) | 12.0 (Blackwell) |
| Driver | 580.178.04, module and package matched (re-verified 2026-09-15; the 2026-09-10 mismatch is resolved — see banner) | same |
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
>
> **Live status (2026-09-10):** both GPU-local NUMA nodes have memory again (GPU0→node 3: 254 GiB,
> GPU1→node 0: 129 GiB — originally confirmed via `/sys/bus/pci/devices/<addr>/numa_node` while
> `nvidia-smi` was broken by the driver mismatch; still current as of 2026-09-15). The memory-less-node NCCL
> segfault (`ncclCuMemHostEnable`→`cuMemCreate` crashing at `ncclCommInitRank` when a GPU's local node
> has 0 MB) hit during the 4-DIMM isolation-testing low point and doesn't apply right now. The launcher
> scripts (`recipe/serve-qwen38-flash-next-nvfp4.sh`, `-fp8.sh`, `serve-glm-53-flash.sh`) all auto-detect
> this per-launch and only force `NCCL_CUMEM_HOST_ENABLE=0` when actually needed, so no flag/doc update
> is required as DIMMs move again — this note documents the mechanism, not a currently-active workaround.
>
> **Reusable check-topology command** (works without `nvidia-smi`, useful right now since it's broken):
> ```bash
> numactl -H   # per-node size/NUMA population
> for f in /sys/bus/pci/devices/*/; do
>   grep -q 0x10de "${f}vendor" 2>/dev/null && echo "$f -> numa_node=$(cat "${f}numa_node")"
> done
> ```
> (`0x10de` = NVIDIA's PCI vendor ID; each GPU shows up as 2 devices — the GPU function and its audio
> function — both report the same NUMA node.)

## 3. Memory (host RAM)

**Re-verify live before relying on any number here** (`numactl -H` for per-node size/NUMA-node
population, `free -h` for total/available) — the config has changed twice this week and will change
again once the 2 RMA replacement DIMMs arrive and go in.

| Property | Nameplate (original, 8 DIMMs) | 2026-09-08 isolation-testing low point | **Live now (2026-09-10, 6-DIMM interim)** |
|---|---|---|---|
| Total | 1.0 TiB (8 × 128 GB Micron DDR4-3200 8-rank RDIMM) | ~499 GiB (4 DIMMs) | **~751 GiB (6 DIMMs)** |
| NUMA nodes with memory | 4 (0–3) | 2 (nodes 0, 1) | **4 (0–3) — all populated, unevenly** |
| Per-node size | ~256 GiB each | nodes 0/1 ~250 GiB, nodes 2/3 = 0 | **node0 129 GiB (1 DIMM), node1 258 GiB (2 DIMMs), node2 129 GiB (1 DIMM), node3 254 GiB (2 DIMMs)** |
| Available | ~925 GiB | ~490 GiB | **~742 GiB (see live `free -h`)** |

Plenty of RAM either way — not the capacity constraint for the model sizes this box targets; can
RAM-offload or use huge KV caches. **Root cause of the recurring fault is now identified as two bad
DIMMs**, not one: the original channel-G/MM4 suspect (memtest86+ errors) and a second stick found to
have physically damaged PCB pads (chipped/missing small components) — a silent failure mode memtest
wouldn't have caught on its own. Both are pulled for RMA; the other 6 known-good sticks are reinstalled
(the uneven per-node sizes above reflect which of the original 8 slots lost a stick). Plan: RMA the 2
bad sticks, reinstall the replacements, return to the full 1 TiB / 8-DIMM config. See the hardware-status
banner at the top of this file and `power-trip-diagnosis.md` for the up-to-date plan.

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
