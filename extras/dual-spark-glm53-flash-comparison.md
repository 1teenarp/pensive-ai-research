# Dual-GLM-5.3-Flash-on-Spark vs. pensive — what makes their token speed better

Date: 2026-09-24. Status: research/read-only comparison. **P12:** the upstream setup uses a
third-party Docker image (`eugr/spark-vllm-b12x`) and a third-party NVFP4 checkpoint
(`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`). We read it for what it proves; we do not serve
from it. All speed deltas below are **estimates** (P10) — the upstream repo publishes **no tok/s
number for the GLM-5.3-Flash recipe** (only for e.g. Qwen3.5-397B on 4×Spark: ~37 tok/s
single-user, ~103 tok/s aggregate at 4 concurrent users).

Source: <https://github.com/1teenarp/spark-vllm-docker> (community fork of
`eugr/spark-vllm-docker`), `recipes/glm-53-flash.yaml`, `docs/NETWORKING.md`, Dockerfile/README
changelog (read 2026-09-24, main @ f64f55f). Local counterpart: `nvidia/GLM-5.3-Flash-NVFP4` on
pensive, RUN-LOG R-020 (~2.4–2.5 tok/s measured, serving 2026-09-24).

## 0. Bottom line

Their token speed is higher because of the **machine, not the recipe**:

1. **Zero CPU offload.** 2×128 GB unified LPDDR5x (256 GB) holds the ~190 GiB NVFP4 weights +
   KV cache entirely on-chip. pensive's 144 GB VRAM forces 32 GB/worker host-RAM offload, so every
   decode round streams expert weights at ~10 GB/s pinned H2D — our ~2.4 tok/s ceiling *is* that
   bandwidth limit (`GB_per_token` law, AGENTS.md §B3).
2. **RoCE RDMA interconnect instead of host-RAM-bounced collectives.** Their TP2 all-reduces cross
   a 200 Gb/s-class ConnectX-7 RoCE link (~14 GB/s measured, ~1.5 µs latency). Ours cross the EPYC
   data fabric through DRAM with no P2P path — a documented **~20–25 ms/token floor** and ~104
   blocking all-reduces per decode round (NCCL spin-wait signature).
3. **Full optimized kernel stack + CUDA graphs + MTP×5.** B12X attention/linear/MoE backends, AOT
   artifacts, graphs on (no `--enforce-eager`), and MTP speculative decoding at
   `num_speculative_tokens=5`. Ours runs the ported NoPE sparse-MLA kernel in **eager** (graphs
   `NEVER`), bf16 KV only, **no MTP yet** (stage 3 pending).

Composite estimate, single-user decode: **~15–30 tok/s on 2×Spark (estimate)** vs 2.4–2.5 tok/s
measured on pensive. Levers 1+2 are structural (hardware); lever 3 is partially portable (see §5).

## 1. Hardware comparison

| | 2× DGX Spark (theirs) | pensive (ours) |
|---|---|---|
| Accelerator | 2× GB10 Grace Blackwell Superchip (1 GPU + 20-core Arm Grace per node, sm_121) `[upstream]` | 2× RTX PRO 5000 72 GB Blackwell (sm_120), both in one box |
| Model memory | **256 GB unified LPDDR5x @ 273 GB/s** `[upstream]` — weights + KV fully resident, zero offload | 144 GB VRAM (2×72 GB) → **32 GB/worker host offload**, DDR4-3200 @ ~10 GB/s pinned H2D `[measured]` |
| GPU↔GPU link | **ConnectX-7 RoCE RDMA, up to 200 Gb/s** per cable (two PCIe5×4 "twin" links; `ib_write_bw` ~111.7 Gb/s ≈ 14 GB/s, ~1.5 µs latency, MTU 9000) `[measured in their docs]` | **None usable** — no NVLink, no PCIe switch, different root complexes + NUMA nodes; only path is via CPU data fabric + DRAM (`SYS` in `nvidia-smi topo -m`) `[measured 2026-09-24]` |
| Host CPU | Arm Grace (on the same SoC as GPU + memory) | AMD EPYC 7663, NPS4 (4 NUMA nodes), DDR4 |
| Decode character | compute + low-latency-network-comm bound | **bandwidth-bound (weight offload stream) + comm floor + eager execution** |

The interconnect row is the headline: ~1.5 µs RDMA vs ~20–25 ms host-bounce per collective round.
On pensive the "no P2P" condition is *structural*, not a config accident (see
`KNOWLEDGE.md` §1, 2026-09-24 entries: separate root complexes `0000:00` / `0000:c0`, separate
IOMMU groups 65 / 18, AMD-Vi in `Translated` mode, no `iommu=pt`).

## 2. Their recipe, verbatim (`recipes/glm-53-flash.yaml`)

```yaml
model: local-inference-lab/GLM-5.3-Flash-NVFP4-Spark
container: vllm-node-b12x            # eugr/spark-vllm-b12x nightly, community fork
cluster_only: true
defaults:
  tensor_parallel: 2
  pipeline_parallel: 1
  decode_context_parallel: 1
  block_size: 256
  max_model_len: 500000              # 500k context (model native: 1M)
  max_num_seqs: 4
  max_num_batched_tokens: 4096
  num_speculative_tokens: 5
  kv_cache_memory_bytes: 8G         # deliberately small: "relax memory pressure"
  gpu_memory_utilization: 0.87
env:
  CUTE_DSL_ARCH: "sm_121a"
  SAFETENSORS_FAST_GPU: "1"
  VLLM_ENABLE_ROCE_ALLREDUCE: "1"   # RoCE all-reduce transport
  VLLM_ROCE_ALLREDUCE_MAX_SIZE: "2MB"
  VLLM_WORKER_MULTIPROC_METHOD: "spawn"
  VLLM_SSM_CONV_STATE_LAYOUT: "DS"
  VLLM_USE_AOT_COMPILE: "1"
  VLLM_USE_MEGA_AOT_ARTIFACT: "1"
  VLLM_USE_V2_MODEL_RUNNER: "1"
  VLLM_ENABLE_PCIE_ALLREDUCE: "0"
  B12X_POLICY_MODE: "auto"
command: |
  vllm serve local-inference-lab/GLM-5.3-Flash-NVFP4-Spark \
    --tensor-parallel-size 2 --decode-context-parallel-size 1 \
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
    --dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed \
    --attention-backend B12X --block-size 256 \
    --moe-backend b12x --linear-backend b12x \
    --no-enable-flashinfer-autotune \
    --load-format b12x \
    --max-model-len 500000 --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5,"moe_backend":"humming","attention_backend":"B12X"}' \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --gpu-memory-utilization 0.87
```

Notable vs. our launcher defaults:

- **fp8 KV cache** at 500k context — ours is **bf16 KV only** (the ported NoPE kernel rejects
  `auto`/fp8; R-017/R-018) and we run 8192 context (stage-2 conservative baseline).
- **MTP with 5 speculative tokens** (`humming` MoE backend + B12X attention for the draft) — ours
  has no MTP yet; stage 3 is the next increment, starting at 2.
- **b12x load format** — experimental loader, "faster and more memory efficient than
  InstantTensor on DGX Spark" (changelog 2026-09-06).
- `--decode-context-parallel-size 1` — CP exists in their fork but is idle here.
- Parsers match ours (`glm45` / `glm47`).

**Their residual bottlenecks** (for completeness): (a) every decode round still pays 60–120 TP2
all-reduces over the network (now µs-cheap, not free); (b) the 8 GB KV cap bounds concurrent
long-context sessions; (c) the whole stack is explicitly experimental — nightly forked vLLM,
JIT-compiled B12X kernels, "update the repository often".

## 3. Software stack

| Layer | Spark recipe | pensive (R-020) |
|---|---|---|
| Base image | `eugr/spark-vllm-b12x:latest` — community nightly built from `local-inference-lab/vllm@dev/infernal-invocation` + `lukealonso/b12x` | `pensive/glm53-flash:nope-sm120-617d0cc` — derived from `vllm/vllm-openai:glm53-flash` (vendor fork, vLLM 0.1.dev20051+g487ecf187, FlashInfer 0.6.17) + ported Apache-2.0 `glm53_sparse_mla` NoPE kernel, built in-container for sm_120a |
| B12X kernels | attention, linear, MoE, GDN, sparse indexer; CUTLASS DSL 4.7; PyTorch 2.13; AOT + "mega AOT artifact"; V2 model runner | absent — our NoPE kernel covers sparse-MLA attention only; MoE/linear are stock vLLM/FlashInfer paths |
| NCCL | v2.30u1 built for sm_121; **RoCE RDMA transport** (`NCCL_IB_HCA` set to both twin links); MTU 9000 | stock NCCL with `NCCL_P2P_DISABLE=1`, `--disable-custom-all-reduce` (CUSTOM path CUDA-errors on sm_120), host-shared-memory path; auto `NCCL_CUMEM_HOST_ENABLE=0` guard |
| Load format | b12x (experimental) | stock safetensors via vendor image |
| Spec decode | MTP ×5 (built into model, `humming` MoE backend) | none yet (stage 3) |
| Graphs | on (AOT artifacts) | eager only — CG `NEVER` on this kernel |

## 4. Provider specifics

- **Checkpoint:** `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` — a community NVFP4 export
  (same org that forks vLLM), *not* the vendor-published `nvidia/GLM-5.3-Flash-NVFP4` we serve.
  P12: fine to read, not a serving path for us. Same ModelOpt NVFP4 family, same 1M native context,
  320B/18B-active MoE — the "Spark" suffix marks it as tuned for the Spark memory layout.
- **Runtime:** community fork + nightly prebuilt Docker images (DockerHub `eugr/spark-vllm-b12x`),
  CI-tested on "multiple models in both cluster and solo configuration". They also run an upstream
  `vllm-project/vllm` PR-patching mechanism (`--apply-vllm-pr`) — PRs apply to the *upstream* tree,
  not the fork.
- **Ecosystem:** the same repo carries recipes for DeepSeek-V4-Flash, Qwen3.8 Flash Next (solo +
  cluster), Qwen3.5-397B (2×/4×/8× Spark), Step-3.7-Flash, MiniMax, Gemma4, Nemotron — i.e. a
  maintained sm_12x (GB10) serving ecosystem, not a one-off. GLM-5.2-NVFP4 runs on **8× Spark**
  there — a model that is a hard ❌ on pensive.

## 5. What this proves for pensive (P12: extract the method)

Portable into our own image / strategy (ranked by expected payoff on our 2.4–2.5 tok/s baseline):

1. **MTP speculative decoding** — the single largest soft lever we have (stage 3, already queued).
   Theirs proves MTP×5 is a sane target for *this model*; our own AGENTS.md §C5.2 measured
   2.5–3× from spec decode on comm-bound setups. Start 2, walk up one variable per session
   (P2); our earlier 5-token crash was a ring-capacity/block-size divisibility issue on a
   different config, not a model property.
2. **Watch B12X's upstream landing** — their changelog says B12X support is being integrated into
   upstream vLLM; if it covers sm_120 (their kernels are sm12x-class, `CUTE_DSL_ARCH=sm_121a` is
   arch-specific), CUDA graphs + fp8 KV on our NoPE path could become reachable — both currently
   `NEVER` on the ported kernel.
3. **fp8 KV** — halves KV footprint and buys context/concurrency; blocked only by our kernel's
   bf16-only constraint, same unlock as #2.
4. **Concurrency (`max_num_seqs` 1 → 2–4)** — amortizes fixed per-round cost; but under our offload
   it multiplies H2D traffic, so test *after* offload trim, one variable at a time.
5. **Not portable:** RoCE all-reduce (requires a second node + RDMA NICs; we have one node, no
   usable P2P), b12x loader (fork-specific), zero-offload memory layout (144 GB VRAM < 190 GiB
   weights — physics), 250 W→300 W headroom (theirs is a desktop Superchip; ours is already capped
   for stability reasons).

Host-level comm fix (`iommu=pt`, P2P restoration) remains the largest *structural* lever on
pensive but, per the 2026-09-24 topology probe, the two GPUs are on separate root complexes with
no direct PCIe path — so even with `iommu=pt` the best case is faster host-bounced collectives,
never a Spark-style direct link. Propose, don't just do it (AGENTS.md §C5.7).

## 6. Open questions

- Measured tok/s for their GLM-5.3-Flash recipe (not published anywhere found; a
  `vllm bench serve` sweep with warmup discarded would close the estimate).
- Does B12X attention/MoE backport to sm_120, or is it GB10-only in practice? (affects items 2–3
  above; re-check the upstream vLLM tree + b12x repo when it lands.)
- Whether their `humming` MoE backend for the MTP draft has an upstream/stock equivalent we could
  use for our MTP stage-3 run.
