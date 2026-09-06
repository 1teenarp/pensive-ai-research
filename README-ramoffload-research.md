# RAM + VRAM Hybrid Serving Research — Serving Models Larger Than GPU VRAM

Date: 2026-09-03
Goal: discover available paths (Unsloth / llama.cpp / vLLM / SGLang) for serving a model that does **not** fit in GPU VRAM by combining **system RAM** with **GPU VRAM**. This build has **1 TiB RAM** (largely reclaimable ZFS cache → effectively ~900+ GB usable) and **2× (speculated 3×) 72 GB Blackwell** = **~146–220 GB VRAM**.

Reference point: **Colibri** (`JustVugg/colibri` — "pure C, zero deps, experts streamed from disk") serves **frontier MoE** models that exceed VRAM by streaming **experts from disk** with intelligent caching. The requested analog is the **same idea but with RAM as the fast tier** instead of disk — i.e. the "expert/weight offload to host RAM" pattern below.

---

## 1. Two kinds of memory you can offload

Offloading is not one knob — there are two independent pools of GPU memory big enough to matter for over-VRAM serving:

| Offload target | What it is | Where it lives | Providers |
|---|---|---|---|
| **Model weights / MoE experts** | The parameter tensors (often >VRAM) | host RAM (or disk) | llama.cpp `-ngl`, vLLM `--cpu-offload-gb`, SGLang |
| **KV cache** | Activations / context for concurrent sequences | host RAM → disk/object tiers | vLLM OffloadingConnector, SGLang, llama.cpp CPU KV |

For **serving a model larger than VRAM**, the dominant term is **weights**. For **long-context / high-concurrency on a fitting model**, the dominant term is **KV cache**.

---

## 2. llama.cpp (current provider) — the simplest RAM+VRAM split

llama.cpp's core design already does weight partitioning between GPU VRAM and host RAM.

- **`--n-gpu-layers` / `-ngl`** (or `999`/`all`) — number of layers kept in VRAM; **the rest run on CPU RAM**. This is the canonical RAM+VRAM split. Set it below the max and the remainder is served from system RAM (via mmap).
- **`--split-mode layer`** (default) — pipeline parallelism across GPUs; leftover layers go to CPU.
- **`--split-mode tensor`** (experimental) — tensor-parallel across GPUs.
- **`--tensor-split` / `--device` / `--main-gpu`** — control which GPU(s) get what.
- **`--mmap`** (default) — memory-maps weights so pages are resident in RAM; **`--mlock`** pins them in RAM; **`--no-mmap`** avoids the mapping.
- **`--cache-type-k/v`** — quantize the KV cache (q8_0 / q4_0) to shrink KV in VRAM.
- Plus the existing **speculative decoding** pattern already in use (main model + `--spec-draft-model` on a second GPU).

> Note: llama.cpp has **no expert-level cache abstraction** — it moves whole layers between GPU and CPU RAM. The Colibri-style per-expert caching for MoE is *not* in core llama.cpp; it exists in the third-party Colibri / peregrine forks.

**Practical recipe on this box** (over-VRAM MoE, e.g. 397B/744B-class):
```
llama-server -m model.gguf \
  -ngl <layers-that-fit-on-GPU(s)> \   # remainder of layers+experts served from host RAM
  -sm layer -ts 1,1,1 \                 # 2 or 3 GPUs
  -fa on -ctk q8_0 -ctv q8_0 \          # quantized KV frees VRAM for weights
  --host 0.0.0.0 --port 8081 --alias my-model
```
Tune `-ngl` until weights ~fit in VRAM; the residual runs from the 1 TiB RAM. Performance is bounded by CPU↔GPU transfer (PCIe) and CPU RAM bandwidth.

---

## 3. vLLM — richest official RAM/disk offload stack

vLLM has **two independent offload systems** matching the two pools in §1.

### 3a. Weight offload to host RAM — `--cpu-offload-gb`
- `--cpu-offload-gb <GB>` — moves model **weights** to **system RAM**. vLLM fills GPU VRAM **first**, then spills only the excess to RAM; the value is an **upper bound**, not a reservation.
- Combine with `--tensor-parallel-size <N>` across multiple GPUs.
- Implemented in `vllm/model_executor/offloader` (`prefetch.py`, `uva.py`), including **prefetch offloading** (overlap weight transfer with compute) and **UVA** for MoE.
- **Caveat:** it offloads **weights**, not KV — KV must still fit in VRAM. For MoE, expert weights can be prefetched to hide PCIe latency.

```
vllm serve <model> --tensor-parallel-size 2 --cpu-offload-gb 750 --max-model-len 8192
```

### 3b. KV cache offload to RAM → disk → object store — `OffloadingConnector`
This is vLLM's closest analog to **Colibri's tiered caching** (but for KV, not experts):
- `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":<bytes>,"secondary_tiers":[...]}}'`
- **Single-tier (`CPUOffloadingSpec`, default):** offloads finished KV blocks to **pinned host RAM**, DMA async via `cudaMemcpyAsync` — minimal CPU/GPU overhead.
- **Multi-tier (`TieringOffloadingSpec`):** CPU primary tier **+ secondary tiers** — `fs` (disk), `obj` (S3), `p2p` (RDMA). GPU→CPU→disk, eviction policies `lru`/`arc`.
- Tunables: `cpu_bytes_to_use` (larger than GPU KV for hit-rate), `block_size`, `eviction_policy`, per-request `max_offload_tokens`.

---

## 4. Unsloth — no own hybrid engine; it's quantization + re-export

Unsloth (as a serving layer) does **not** have its own RAM+VRAM serving engine. Its contribution to "bigger-than-VRAM" is **quantization** that shrinks weights so they fit, and it hands serving off to llama.cpp / vLLM / SGLang:

- **Unsloth Dynamic NVFP4** on Blackwell (`sm_120`) — W4A4 quantization with selective FP8/BF16 layers; much smaller VRAM + ~2.5× faster on Blackwell. E.g. `unsloth/Qwen3.8-27B-NVFP4`.
- **Unsloth Dynamic 3.0 GGUFs** — for llama.cpp/Ollama.
- **FP8 KV cache calibration** — 2× longer context in the same VRAM.
- Serving is done via their **vLLM / SGLang / llama-server** deployment guides.

So for the RAM+VRAM goal, **Unsloth's role is to make weights smaller**; the actual RAM spill is performed by the underlying engine (llama.cpp `-ngl` or vLLM `--cpu-offload-gb`). Use Unsloth quants to maximize the fraction that fits in VRAM, then spill the rest to RAM.

```
vllm serve unsloth/Qwen3.8-27B-NVFP4 --tensor-parallel-size <N> --cpu-offload-gb <RAM_GB>
```

---

## 5. SGLang — KV offload + layerwise/weight offload

- KV-cache offload to CPU RAM (`decode_kvcache_offload_manager`) and **layerwise offload** (`layerwise_offload.py`, used for multimodal) to keep activations off the GPU.
- Weight offload to host memory via `--cpu-offload-gb` (host RAM), as with vLLM.
- Good alternative if the vLLM path hits an MoE-backend limit; also supports NVFP4 quants.

---

## 6. Colibri & the "RAM-based expert cache" idea (the real target)

**Colibri** (`JustVugg/colibri`, plus Rust spin-off `peregrine`) serves **frontier MoE models** on consumer hardware by:
- keeping **attention/shared weights + a working set of experts** in GPU/VRAM,
- **streaming the remaining experts from disk** with **intelligent prefetch/caching** of the experts actually used per step (MoE routing is highly sparse → strong locality).

The requested **RAM-based analog** swaps the disk tier for the **1 TiB of host RAM**:

| Tier | Colibri (disk) | Desired (RAM on this box) |
|---|---|---|
| Fast | GPU VRAM | GPU VRAM |
| Cache tier | SSD, expert streaming | **Host RAM (1 TiB, ~900 GB usable)** |
| Benefit | sparse expert routing ⇒ few expert loads | same locality, but RAM bandwidth ≫ disk ⇒ far fewer stalls |
| Provider support | colibri / peregrine (custom) | closest off-the-shelf: **vLLM `--cpu-offload-gb`+prefetch, llama.cpp `-ngl`** |

**Key insight for the machine:** EPYC 7663 (8-channel DDR4-3200) provides a few hundred GB/s of RAM bandwidth plus deep PCIe lanes to the GPUs. Because MoE routing activates only a few experts per token, a **RAM-resident expert cache** can keep ~all experts in RAM and stream just the active one(s) to the GPU per step — exactly the Colibri pattern with a much faster backing tier. Paths that expose per-expert prefetch (vLLM prefetch offload, or a colibri/peregrine-style engine) are the highest-value option here.

---

## 7. What's realistically servable via RAM+VRAM on this build

Budget: ~146 GB (2 GPU) / ~220 GB (3 GPU) VRAM + ~750–900 GB usable host RAM. vLLM/llama.cpp only offload weights, so a model's **weight bytes** can be split VRAM-first, then RAM.

| Model (already on disk) | Weight size | Split | Verdict via RAM offload |
|---|---|---|---|
| Qwen3.5-397B-A17B NVFP4 | ~223 GB | VRAM ~146–220 + RAM remainder | ✅ runnable |
| GLM-5.2-NVFP4 | ~426 GB | VRAM + RAM (needs >420 GB RAM) | ⚠️ runnable, slow |
| MiniMax-M3 | ~664 GB | mostly RAM | ⚠️ same |
| DeepSeek-V4-Pro | ~786 GB | mostly RAM | ⚠️ borderline, very slow decode |
| Qwen-AgentWorld-35B-A3B | 36 GB | fully VRAM | ✅ no offload needed |
| Qwen3.8-27B NVFP4 | 22 GB | fully VRAM | ✅ no offload needed |

> Practical limit: with weight offload, the **KV cache must still fit in VRAM**, so long-context on a huge model is the real constraint (use quantized KV + modest `--max-model-len`). For MoE specifically, **per-expert RAM caching** (Colibri-style) is far more efficient than whole-layer RAM offload, since only a handful of experts are active per token.

---

## 8. Recommended path

1. **Simplest & most supported:** **vLLM** with the model quantized via **Unsloth NVFP4** (Blackwell-native, smallest weights) + `--tensor-parallel-size <N>` + **`--cpu-offload-gb <RAM_GB>`** for the spill.
   ```
   vllm serve <unsloth-nvfp4-model> --tensor-parallel-size 2 --cpu-offload-gb 750
   ```
2. **MoE-specific / largest models:** use a **Colibri- or peregrine-style** engine (expert streaming/prefetch) pointed at RAM, or llama.cpp with tuned `-ngl`.
3. **Long-context on a fitting model:** add vLLM `OffloadingConnector` (CPU-RAM KV tier, optional disk tier) to push KV into the 1 TiB RAM.
4. Measure: tokens/s decode, TTFT, and VRAM/RAM headroom. Expect RAM-offloaded decode to be much slower than fully-on-GPU — only the sparse-expert MoE case mitigates this.

---

## 9. Empirical results — attempted live on this machine (2026-09-04)

What actually happened when trying to serve **GLM-5.2-NVFP4** (426 GB, 753B-total / 40B-active MoE, sparse-attention DSA, NVFP4) to prove the RAM+VRAM offload path.

### Setup context
- VMs used 2× 72 GB Blackwell (146 GB VRAM) + 1 TiB RAM; model staged on `/trunk/ai` (ZFS on WD enterprise HDDs, ~175 MB/s) and copied to `/buffer` (Samsung 990 EVO NVMe) for fast cold-start.
- All attempts used `--shm-size 16g` (container-private) and **no `--ipc=host/--privileged/--cap-add`** (no host IPC exposure).
- **ZFS ARC insight:** `zfs_arc_max=0` and `arcstats c_max≈1.0 TiB` → the whole model can be cached in RAM after one read; ARC caches raw on-disk bytes only (helps cold-start), whereas the engine's **deserialized weight tensors are a separate allocation**.

| # | Provider / config | Result |
|---|---|---|
| 1 | vLLM 0.27.1 `serve` TP2 + `--cpu-offload-gb 400` | ❌ **HANG** — workers 100% CPU right after NCCL init, no "Loading model weights", RAM/GPU static |
| 2 | vLLM 0.23.0 TP2 + `--cpu-offload-gb 400` | ❌ same hang (identical signature) |
| 3 | vLLM 0.27.1 **TP1** + fp8 KV | ❌ fast failure: `No valid attention backend … sparse not supported … compute capability not supported` (no Blackwell sparse-MLA backend in vLLM) |
| 4 | vLLM 0.23.0 TP2, drop fp8 KV | ❌ hang again |
| 5 | vLLM 0.23.0 TP2 + `--enforce-eager` | ❌ hang (not a compile issue) |
| 6 | **SGLang** `dev-glm52-nvfp4` TP2 + `--cpu-offload-gb 400` | ❌ **HANG** at `Init torch distributed / NCCL` (TP0/TP1) |
| 7 | **SGLang TP1** + `--cpu-offload-gb 500` | ⚠️ **got past distributed init, loaded weights into RAM (656 GiB!), then crashed** in `fused_moe_triton._load_w13`: `RuntimeError: size tensor a (3072) must match b (6144)` — NVFP4 fused-MoE loader shape bug |

### Key conclusions from the live test
1. **RAM+VRAM offload mechanism works.** Single-GPU SGLang ramped host RAM to **~656 GiB** during weight load (the `--cpu-offload-gb` path actually offloads weights to host RAM). It's not OOM; it's a code/quant-loader issue for this specific frontier checkpoint.
2. **Multi-GPU (TP2) NCCL init hangs on this host** — reproducible across vLLM AND SGLang. Single-GPU always clears this stage. The old `llama-qwen` used both GPUs only as *independent devices* (no NCCL collective), so it never hit this.
3. **GLM-5.2-NVFP4 itself is not servable here yet**: vLLM lacks a Blackwell sparse-MLA attention backend; SGLang's NVFP4 fused-MoE loader mismatches the expert weight shape.
4. **The on-disk copy to `/buffer` is a cold-start optimization only** (NVMe vs ~175 MB/s HDD read); it does not reduce the engine's own tensor-RAM requirement.

> Workaround used during test: `cp -a /src /dst/GLM-5.2-NVFP4` nested the tree one level too deep (created `/dst/GLM-5.2-NVFP4/GLM-5.2-NVFP4/...`); verify the target path before using it. (Aux to the core task.)

---

## 10. TODO

The project TODO (offload tasks + stashed stability/power-trip tasks) has moved to a dedicated file:
[`PROJECT-TODOS.md`](PROJECT-TODOS.md).

---

## 11. Bottleneck benchmark — live offloaded serving (Qwen3.5-397B-A17B-NVFP4)

Measured while the model was serving on **1 GPU** (single-GPU, TP1) with `--cpu-offload-gb 200` (~201 GB of weights offloaded to host RAM, UVAOffloader).

### Measured hardware state (during active generation)
| Lane / resource | Observed | Verdict |
|---|---|---|
| GPU0 PCIe link | **Gen4 x16** | raw headroom OK |
| GPU0 SM utilization | 100% | busy (small per-token ops) |
| GPU0 HBM (memory) utilization | **~0%** | GPU not compute/memory saturated — it *waits* on weights |
| GPU0 power | ~103 W | low |
| GPU0 VRAM in use | 58.8 GB | weights + KV |
| Host RAM used | ~334 GB (≈201 GB offloaded weights) | plenty |
| Host CPU | ~97% idle (EngineCore ~41%, load avg ~3) | **not** a bottleneck |
| **Host→GPU (RAM→VRAM) H2D** | **~8.6 GB/s (pageable), ~10.2 GB/s (pinned)** | **← THE BOTTLENECK** |
| Host RAM read (single-thread) | ~20.8 GB/s | faster than the PCIe lane; not the choke |

### Per-token weight volume
Qwen3.5-397B **A17B** (60 layers, `moe_intermediate_size=1024`, `hidden=4096`, `num_experts_per_tok=10`, ~526 routed experts/layer):
- Active params per token ≈ **~17 B** (per model name; ≈ 8–17 B by config) → **≈ 5–9.5 GB of NVFP4 weight bytes** must move from host RAM → GPU **every decode step**.

### Root cause of the ~1 token/s
```
active weights/token (~9.5 GB)  /  RAM→GPU H2D rate (~10 GB/s)  ≈  ~0.95 s/token  ≈  ~1 token/s
```
- Measured generation throughput: **~1.0 token/s** (90 tok / ~90 s; matches vLLM's own log `Avg generation throughput: 1.0 tokens/s`).
- Everything else has headroom: RAM read (~21 GB/s) and CPU are faster than the PCIe lane; the GPU is starved on weight delivery, not compute.

### Lane ranking (where the bottleneck is)
1. **Host→GPU PCIe H2D (~9–10 GB/s) — the choke.** Each token streams ~GBs of active expert/attention weights from RAM; the link can't deliver faster.
2. GPU compute — 100% "busy" but HBM ~0% ⇒ it stalls on the weight feed (latency/bandwidth-bound), not throughput-bound.
3. Host CPU / RAM — idle / adequate; not limiting.

### What would raise tokens/s
- **Overlap/pin:** pin the offloaded memory (pageable→pinned gave 8.6→10.2 GB/s, ~18%) and/or use vLLM's **prefetch offloader** to overlap H2D with compute.
- **Fewer active params/token** (e.g. a lower "A" MoE, higher sparsity) ⇒ less bytes/token.
- **Both GPUs (TP2)** would halve per-GPU transfer — but blocked by the **2-GPU NCCL hang (task A2 in `PROJECT-TODOS.md`)**.
- A **Colibri/peregrine-style per-expert cache** with RAM as the hot tier is exactly this scenario; only the ~10 GB/s RAM↔GPU lane bounds it.
> Note: `--enforce-eager` was used (CUDA graphs/compile off, since multi-GPU NCCL also blocked those paths), which adds per-step launch overhead on top.

---

## Sources
- llama.cpp README ("CPU+GPU hybrid inference") + `docs/multi-gpu.md`
- vLLM `docs/features/kv_offloading_usage.md`, forum "Deploy a big LLM when GPU VRAM not enough"
- Unsloth `docs/basics/nvfp4.md` (+ Dynamic GGUF guides)
- JustVugg/colibri ("experts streamed from disk"), s-b-repo/peregrine
- SGLang `offloader` / `decode_kvcache_offload_manager` / `layerwise_offload.py`
- Live results on this machine (the 9-row empirical table above)
