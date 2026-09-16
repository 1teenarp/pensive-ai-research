# KNOWLEDGE — decision-shaping facts

**Read this before designing a recipe.** It is the accumulated set of facts that change a decision:
what this hardware can and can't do, what each runtime image actually supports, what a whole *model
family* implies before you've run it once, and the patterns general enough to apply to a model
nobody here has touched.

It exists to cut iterations. Every entry is here because not knowing it cost someone a run.

### What belongs here

A fact earns a place if it is **durable** (outlives the attempt that produced it), **decision-shaping**
(it changes a flag, a strategy, or whether you attempt something at all), and **not free to look up**
(you can't get it from `config.json` in ten seconds).

**What does not belong:** per-attempt narrative (that's [`RUN-LOG.md`](RUN-LOG.md)), current machine
state (that's [`SYSTEM-SPEC.md`](SYSTEM-SPEC.md)), a specific model's staged plan (that's its recipe),
or anything plainly readable from the model config.

### Confidence tags

`[measured]` observed directly here, with numbers · `[observed]` seen once, mechanism understood ·
`[upstream]` from vendor docs or source, not verified here · `[inferred]` reasoned, untested — say so.

---

## 1. This system

**Blackwell `sm_120` gives NVFP4 *and* FP8 natively; NVFP4 is the fast path.** `[measured]`
Same architecture at NVFP4 vs FP8 measured 39–55 vs 20–24 tok/s. CUDA graphs bought ~4× on NVFP4 but
only ~1.4× on FP8.
**Use:** default to a vendor NVFP4 build when one exists; reach for FP8 only when precision is the
requirement.

**sm_120 loses two all-reduce paths.** `[measured]` `SymmMemCommunicator: Device capability 12.0 not
supported` (harmless, expected) and the CUSTOM all-reduce raises a CUDA error.
**Use:** always pass `--disable-custom-all-reduce`; vLLM lands on PYNCCL.

**The two GPUs have no NVLink and sit on different root complexes / NUMA nodes** (GPU0→node 3,
GPU1→node 0; `SYS` in `topo -m`). `[measured]` Default NCCL P2P/CUMEM channels complete
`init_process_group` and then hang forever in `all_reduce` — on **both** vLLM and SGLang.
**Use:** `NCCL_P2P_DISABLE=1` on every multi-GPU launch, no exceptions.

**Every TP collective therefore bounces through host RAM, putting a ~20–25 ms/token floor under
TP2.** `[measured]` One decode round is ~104 blocking all-reduces. The signature is GPUs at 99 %
"utilization" drawing ~100 W of 300 W with ~9 % memory throughput — that is **NCCL spin-wait, not
work**, and it persists with no request in flight.
**Use:** for a model that fits on one card, TP1 can beat TP2. Before blaming a kernel for slow
decode, check power draw and memory throughput; low-and-low means you're comm-bound.

**Host→GPU is ~8.6 GB/s pageable, ~10.2 GB/s pinned.** `[measured]`
**Use:** this is the divisor in every offload estimate (§4, bandwidth law).

**`iommu=pt` is NOT set on this host.** `[observed]` Leading suspect for the historic P2P hangs and
the largest untaken architectural lever here.
**Use:** propose it as a host change, don't apply it mid-investigation; re-test P2P with
`p2pBandwidthLatencyTest` / `nccl-tests` afterwards.

**`nvidia-smi -pl` needs interactive sudo, so the 250 W power cap usually does NOT apply in agent
sessions.** `[measured]` The launcher warns and continues — correctly, since the cap is a defensive
margin, not the stabiliser.
**Use:** don't treat the warning as a blocker; don't claim the gate passed either.

**A driver upgrade can leave the old kernel module loaded**, breaking `nvidia-smi` and every GPU
container host-wide with an error that reads as model-specific. `[measured]` 2026-09-10.
**Use:** `cat /proc/driver/nvidia/version` vs `modinfo nvidia | grep ^version:` is the first check
when a container dies at init.

**Storage tiers differ by ~20×**: `/trunk/ai` ZFS ~175 MB/s, `/buffer` NVMe ~3–5 GB/s. `[measured]`
**Use:** ZFS-direct is the *safer* first load on a fabric-fragile box (slow, flat, predictable);
NVMe staging is a cold-start optimization only — it does not reduce the engine's tensor-RAM need.

**Usable VRAM is ~128–135 GB** across both cards for a single process, after KV, activations and
CUDA context. `[measured]`

---

## 2. Runtimes and images

**`vllm/vllm-openai:glm53-flash` (vllm `0.1.dev20051+g487ecf187`, flashinfer 0.6.17) is the ONLY
local image that registers `Glm5Next*`.** `[measured 2026-09-15]` `:latest` (0.27.1) and `:v0.23.0`
return `Glm5Next: False`.
**Use:** there is no local "try a newer vLLM" fallback for GLM-5.3. Changing image means a pull.

**That image silently ignores `--max-parallel-loading-workers`** —
`WARNING [parallel.py:959] ... is currently not supported and will be ignored`. `[measured]`
**Use:** serialized loading is one of the standing burst-reduction gates; on this image it is **not
in effect**. Rely on `spawn` and the slow load path instead, and don't report the gate as armed.

**`vllm/vllm-openai:qwen38-flash-next-patched`** carries the FP8-PLE-selector patch (vLLM #54765) and
serves **both** the NVFP4 and FP8 Qwen3.8-Flash-Next checkpoints — their PLE tables share the
single-global-`weight_scale` layout the patch targets. `[measured]`
**Use:** no new patch needed for a sibling checkpoint in this family; verify the layout in
`index.json` first.

**vLLM's PLE CPU-offload has an explicit guard against `PP>1`.** `[measured]`
**Use:** pipeline parallelism is a hard dead end for any model requiring PLE offload. Not tunable.

**`--kv-offloading-size` is cross-request prefix reuse, not live-context extension**, and it is
incompatible with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (pydantic ValidationError).
`[measured]`
**Use:** to buy context headroom, offload **weights** (`--cpu-offload-gb`).

**SGLang `lmsysorg/sglang:dev-glm52-nvfp4` loads GLM-5.2-NVFP4 weights to host RAM (656 GiB
observed) then fails in `fused_moe_triton._load_w13` on an expert-shape mismatch.** `[measured]`
**Use:** SGLang is a real fallback when vLLM lacks a backend, but its NVFP4 fused-MoE loader is not
a safe assumption for frontier checkpoints.

---

## 3. Model families

### Qwen Flash-Next — `Qwen4ExpForConditionalGeneration`
*(nvidia/Qwen3.8-Flash-Next-NVFP4 124 GB · Qwen/Qwen3.8-Flash-Next-FP8 173 GB · BF16 336 GB)*

48 layers · 512 experts / 10 per token · native **262,144** ctx · a **51B-param n-gram/PLE table** ·
MTP head. `[measured]`

- **The PLE table must live in host RAM** (`VLLM_PLE_CPU_OFFLOAD=1`) — ~52 GB that would otherwise
  eat the VRAM budget. It brings three requirements with it: the patched image +
  `VLLM_QWEN38_PLE_FP8_SCALE=1`, `--cap-add SYS_PTRACE` with unconfined seccomp/apparmor (the
  `pidfd_getfd` handshake), and **no pipeline parallelism**.
- **MTP speculative decoding pays even on a comm-bound box** — ~65–70 % per-token acceptance, ~2.5–3×
  end to end. `num_speculative_tokens=4` is the **tested ceiling**; 5 crashes with
  `QSA ring capacity 12 must divide the attention block size 1616`. That block size is computed
  dynamically from page-size alignment, so there is **no formula** — retest when anything changes.
- KV at full 262k needs `--max-num-seqs 1`; a small weight offload (8 GB/worker) frees the headroom.

### GLM-5.3-Flash — `Glm5NextForConditionalGeneration`
*(zai-org FP8 305 GB · nvidia NVFP4 190.5 GiB — natively multimodal)*

320B total / 18B active · 45 layers (3 dense + 42 sparse) · 288 routed experts / 8 per token + 1
shared · hybrid attention: **34 KDA linear-attention + 11 NoPE sparse-MLA (DSA)** · MTP 1 layer ·
native **1,048,576** ctx · parsers `--tool-call-parser glm47 --reasoning-parser glm45`. `[measured]`

- **The family is NoPE MLA, and both checkpoints are byte-identical on attention geometry:**
  `qk_rope_head_dim: 0`, `qk_nope_head_dim: 256`, `v_head_dim: 256`, `kv_lora_rank: 512`,
  `mla_use_nope: true`, `head_dim: 0`. Verified on **both** the FP8 and NVFP4 configs. `[measured]`
- **⛔ This family is NOT servable on sm_120 with `vllm/vllm-openai:glm53-flash` — by any flag
  combination.** `[upstream: verified in image source, not yet reproduced empirically]`
  On sm_120 the MLA selector offers exactly two candidates (`TRITON_MLA`,
  `FLASHINFER_MLA_SPARSE_SM120`); Triton is filtered out for a sparse/indexer model, leaving one.
  That backend **requires** `fp8_ds_mla` (`raise NotImplementedError` otherwise), and its kernel
  **hardcodes `pe_dim == 64`** — the DeepSeek rope shape. So `fp8` dies in the kernel and `auto` dies
  at backend construction. Decisively: the same selector's `else` branch has explicit NoPE handling
  for this family — `prefer_fi_sm90 = qk_rope_head_dim == 0 and hasattr(hf, "index_topk")`, commented
  *"GLM-5-Next shape … prefer FlashInfer's SM90 FA3 path for every KV dtype"* — implemented for
  **SM90 (Hopper) only**. **Use:** don't burn attempts on flags; the paths forward are a newer vLLM,
  SGLang, or Hopper. → RUN-LOG R-014 + its correction
- **KV is cheap, weights are the constraint.** 34 of 45 layers hold fixed-size conv/recurrent state;
  only the 11 DSA layers keep compressed MLA KV. Long context costs little — spend the budget on
  weights, and don't size this family with the dense KV formula.
- Requires **FlashInfer ≥ 0.6.17**; selects `FLASHINFER_MLA_SPARSE_SM120` on Blackwell. `[measured]`
- Expert stream: 42 sparse layers × 8 experts × 3×4096×2048 → **~8.5 GB/token at FP8, ~4.3 GB/token
  at NVFP4**. Against ~10 GB/s H2D that is the whole performance story (§4).
- Vendor target is **TP4 on GB200**; TP4/EP configs from the model card are not reproducible here.

### Any offloaded MoE
Decode is bandwidth-bound long before it is compute-bound — see the law in §4. The observable: GPU
at 100 % SM with **~0 % HBM utilization** means starved on weight delivery, not busy.

---

## 4. Patterns — the generalized rules

These are the entries that should transfer to a model nobody here has run. Promote a §1–3 fact here
once the *mechanism* explains why it must generalize, or once it's been seen in a second family.

**P-A · A KV-cache dtype selects a kernel, not just a precision.** That kernel carries shape
assumptions inherited from whichever model family it was written for.
**Use:** before setting `--kv-cache-dtype fp8` on any MLA model, read `qk_rope_head_dim` and the
head geometry, and check them against the KV format the backend logs at startup.
*Generalized from the GLM-5.3 NoPE/`fp8_ds_mla` collision (R-014).*

**P-B · The offload bandwidth law.** `tok/s ceiling ≈ H2D GB/s ÷ active weight bytes per token`,
before compute enters the picture.
**Use:** compute it during intake. It predicted 1.0 tok/s for the 397B NVFP4 model and the run
measured 1.0. Halving the bytes (NVFP4 over FP8) roughly doubles the ceiling.

**P-C · "Backend selected" in a log is not "kernel works".** Selection happens at model load;
the kernel's shape assertions fire later, at first forward — memory profiling or graph capture.
**Use:** never record a backend as proven from a load-time log line. Three GLM attempts reported
`FLASHINFER_MLA_SPARSE_SM120` "initializing" before any of them had reached a forward pass; the
fourth got there and the kernel rejected the model immediately.

**P-H · When a runtime rejects a model, read the backend *selector*, not just the failing kernel.**
`vllm/platforms/cuda.py` branches on `device_capability.major`, so **a model family can be
first-class on one compute capability and unsupported on another in the same build** — and the
selector's comments often name the family explicitly.
**Use:** on any "unsupported / assert / no valid backend" failure, spend ten minutes reading the
selector for your capability before permuting flags. It tells you whether a fix exists at all, which
no amount of flag search will.
*Generalized from GLM-5.3 on sm_120 (R-014 correction): upstream implemented this exact NoPE shape
for SM90 and not for SM120, so every flag combination was always going to fail.*

**P-D · Sibling checkpoints share arch-level constraints.** A quantized re-export changes bytes, not
attention geometry.
**Use:** `diff` the two `config.json` files before assuming a variant will behave differently — it
costs seconds and can retire a whole planned attempt. GLM-5.3 FP8 and NVFP4 are identical on every
attention field.

**P-E · Quantization format changes speed structurally, not just footprint.** The FP8 batch-1
grouped-MoE kernel path ran ~3.4× slower than NVFP4 at identical comm settings.
**Use:** don't model a format change as "same speed, less memory".

**P-F · Vendor recipes target vendor hardware.** Model-card flags assume the reference platform
(TP4 GB200, NVLink, 8×GPU EP).
**Use:** treat them as a starting point and re-derive parallelism from the local topology.

**P-G · For a failed run, the metric is how far it got, not pass/fail.** A failure that reaches a
later stage than the last one has retired everything upstream of it.
**Use:** name the last stage reached in every RUN-LOG entry (NCCL init → backend select → weight
load → KV init → graph capture → serving). That sequence is what makes a string of failures add up
to knowledge instead of noise.

---

## 5. Known non-issues

Things that look alarming and are not. Each of these has cost someone time.

| Signal | Reality |
|---|---|
| Zero-byte `*.incomplete` files under `.cache/huggingface/download/` | Normal residue. The completion signal is `✅ Successfully downloaded` in the log, plus shard count and total bytes. |
| `Triton is installed but 0 active driver(s) found` during `--check` | Expected — `--check` runs CPU-only with no GPU in the container. |
| `SymmMemCommunicator: Device capability 12.0 not supported` | Expected on sm_120. vLLM falls back to PYNCCL. |
| `WARNING: gpu N power limit is 300 W, NOT 250 W — cap did not apply` | Known: `nvidia-smi -pl` needs interactive sudo. Non-fatal, less margin. |
| `max_parallel_loading_workers is currently not supported and will be ignored` | Non-fatal, **but** a real gate is silently off (§2). |
| `ras-mc-ctl --summary` showing thousands of CEs | Lifetime cumulative since 2026-08-21; 96 % landed in one historic burst. Use per-boot windows for attribution. |
| `No MLA prefill backend supports this model; sparse MLA will use the top-k MQA path only` | Informational on GLM-5.3; not the failure. |
