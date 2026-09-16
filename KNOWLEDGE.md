# KNOWLEDGE — decision-shaping facts

**Read this before designing a recipe.** It is the accumulated set of facts that change a decision:
what this hardware can and can't do, what each runtime image actually supports, what a whole *model
family* implies before you've run it once, and the patterns general enough to apply to a model
nobody here has touched.

It exists to cut iterations. Every entry is here because not knowing it cost someone a run.

> **Last validated: 2026-09-16.** Re-checked against the live box and the installed images:
> §1 topology, NUMA mapping, compute capability, `iommu`, power-cap state; §2 every image's vLLM /
> FlashInfer version and `Glm5Next` registration, plus the sm_120 selector branch and the
> `ENGINE_READY_TIMEOUT_S` default; §3 the GLM-5.3 attention geometry on both checkpoints.
> **Not re-measured** (would need benchmark runs, unchanged hardware assumed): the H2D bandwidth and
> storage read rates in §1, and the Qwen-family throughput numbers in §3.
> Image contents and host state drift. When an entry disagrees with what you observe, **trust the
> observation and correct the entry** (AGENTS.md §9.2), then move this date.

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

**Serve ports are consumed remotely over Tailscale, not just localhost.** `[observed 2026-09-16]`
Own clients on the tailnet (`pensive` = 100.70.5.43) hit the 809x ports; treat the tailnet as a LAN,
not as authz — any tailnet node can reach an engine as if plugged in. All four launchers now expose
`BIND_HOST` (restricts the host-side `-p` publish address; engine must still bind 0.0.0.0 *inside*
the container for docker-proxy NAT) and `API_KEY` (vLLM `--api-key`, Bearer). Both default to the
legacy behavior (all-ifaces, no auth), so auth is opt-in per launch.
**Use:** before exposing a new engine, decide bind+auth explicitly; remember `--api-key` does NOT
cover `/health` and `/metrics`, and a stale dev image (e.g. the sglang build carrying the
multimodal-RCE CVE-2026-3059) is a different risk once anything beyond localhost can reach it.

**`docker save` silently produces a corrupt tar for images whose compressed layer blobs the daemon
can't re-read** (seen on `lmsysorg/sglang:dev-glm52-nvfp4`: exit 0, 31 KB file; same blob error
Trivy hits in daemon mode). `[observed 2026-09-16]` Running containers and `docker export` are
unaffected. **Use:** to audit such an image, `docker export` + `trivy rootfs`; to move it, re-pull
from the registry instead of `docker save`.

---

## 2. Runtimes and images

**Which local images register `Glm5Next*`** `[measured 2026-09-16]` — re-verify after any pull;
this table went stale within a day the first time.

| Image | vLLM / FlashInfer | `Glm5Next` | Notes |
|---|---|---|---|
| `:glm53-flash` | `0.1.dev20051+g487ecf187` / 0.6.17 | **yes** | **Vendor fork.** Carries the SM90 GLM-5-Next NoPE selector logic that is **absent from the public tree**. |
| `:nightly` | `0.29.1rc1.dev187+gaf1c01499` / 0.6.18.post1 | **yes** | Public. Registers the arch but **cannot serve it on sm_120** — see below. |
| `:latest` | 0.27.1 | no | |
| `:v0.23.0` | 0.23.0 | no | |

**Registering an architecture is not the same as being able to run it.** `[measured]` The public
nightly registers `Glm5Next` and still fails identically to the vendor fork on sm_120 (R-015): its
`device_capability.major == 12` branch is byte-identical (`TRITON_MLA`,
`FLASHINFER_MLA_SPARSE_SM120`), its SM120 backend still raises without `fp8_ds_mla`, and its compiled
kernel still carries the `pe_dim == 64` assert (line moved 866→937). Decisively, **`prefer_fi_sm90`
and the `GLM-5-Next` comment do not exist in the public nightly at all** (grep count 0) — that
handling is vendor-fork code.
**Use:** `--check` passing proves registration only. "Pull a newer public vLLM" is **retired** as a
fix path for GLM-5.3 on sm_120; a fix must come from the vendor fork or another runtime.

**`VLLM_ENGINE_READY_TIMEOUT_S` defaults to 600 s** `[measured 2026-09-16, nightly]` — too tight for
a large cold load from ZFS (190 GiB at ~175 MB/s is well past it).
**Use:** raise it for any big first load; the GLM NVFP4 launcher exposes it as
`ENGINE_READY_TIMEOUT_S`. A load killed at exactly ~10 min with no kernel error is this, not the model.

**`:glm53-flash` silently ignores `--max-parallel-loading-workers`** —
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
(And even without the guard it wouldn't pay at batch-1 decode — see P-I.)

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
- **⛔ This family is NOT servable on sm_120 on any vLLM available to us.**
  `[measured]` for `--kv-cache-dtype fp8`: reproduced at KV init **twice, on two different images** —
  R-014 (`:glm53-flash`) and R-015 (public `:nightly` 0.29.1rc1.dev187) — identical assert, kernel
  line moved 866→937 but the check stands.
  `[upstream]` for the stronger claim that **no** flag combination works: `auto` has never actually
  been run. It is predicted to fail *earlier* (at backend construction, not in the kernel) because
  the SM120 backend raises without `fp8_ds_mla` and is the only candidate left. One `--dummy` with
  `KV_CACHE_DTYPE=auto` would close this — expect a backend-selection error, **not** a `pe_dim`
  assert. If it instead produces a `pe_dim` assert, this entry is wrong and needs rewriting.
  On sm_120 the MLA selector offers exactly two candidates (`TRITON_MLA`,
  `FLASHINFER_MLA_SPARSE_SM120`); Triton is filtered out for a sparse/indexer model, leaving one.
  That backend **requires** `fp8_ds_mla` (`raise NotImplementedError` otherwise), and its kernel
  **hardcodes `pe_dim == 64`** — the DeepSeek rope shape. So `fp8` dies in the kernel and `auto` dies
  at backend construction. Decisively: **in the vendor fork `:glm53-flash`**, the same selector's
  `else` branch has explicit NoPE handling for this family —
  `prefer_fi_sm90 = qk_rope_head_dim == 0 and hasattr(hf, "index_topk")`, commented *"GLM-5-Next
  shape … prefer FlashInfer's SM90 FA3 path for every KV dtype"* — implemented for **SM90 (Hopper)
  only**. That code **does not exist in the public nightly at all** `[measured 2026-09-16, grep
  count 0]`, so the arch is registered publicly but has no NoPE path on any capability there.
  **Use:** don't burn attempts on flags, and don't pull a newer *public* vLLM — that path is retired
  (R-015). What's left: a newer **vendor-fork** image, SGLang, or Hopper/`sm_100` hardware.
  → RUN-LOG R-014 + its correction, R-015
- **KV is cheap, weights are the constraint.** 34 of 45 layers hold fixed-size conv/recurrent state;
  only the 11 DSA layers keep compressed MLA KV. Long context costs little — spend the budget on
  weights, and don't size this family with the dense KV formula.
- Requires **FlashInfer ≥ 0.6.17**; selects `FLASHINFER_MLA_SPARSE_SM120` on Blackwell. `[measured]`
- Expert stream: 42 sparse layers × 8 experts × 3×4096×2048 → **~8.5 GB/token at FP8, ~4.3 GB/token
  at NVFP4**. Against ~10 GB/s H2D that is the whole performance story (§4).
- **Why the official GB200 recipe works and ours can't** `[measured, source-read]` — the published
  vLLM recipe runs this model with `--kv-cache-dtype fp8` on **GB200 = sm_100**, where the selector
  takes the `major == 10` branch and picks `FLASHINFER_MLA_SPARSE` with **plain fp8** KV, never
  `fp8_ds_mla` — so the `pe_dim == 64` assert is never reached. sm_120 is a *different branch* with a
  different, stricter backend. TP4 isn't reproducible here either (2 GPUs).
  **Use:** a vendor recipe "working" is scoped to its compute capability. Read the selector branch for
  *your* capability before assuming a published config transfers. → P-H
- **A working sm_12x NoPE path exists in a community fork** — proof the gap is buildable, not a law
  of the hardware. **Reference only, never a serving path** (AGENTS.md **P12**: official images and
  vendor-published weights only). Read it for method; don't serve from it. → recipe §0

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
*Generalized from GLM-5.3 on sm_120 (R-014 correction): the NoPE shape is implemented for SM90 and
not SM120, so every flag combination was always going to fail.*

**Corollary (R-015) · "try a newer version" is a hypothesis, and the selector tests it in minutes.**
Before pulling a 30 GB image, grep the *installed* one's selector branch for your capability and
compare. The public nightly registered the architecture, looked like progress, and reproduced the
identical failure — because the special-case we needed was **vendor-fork code that was never in the
public tree**. Registration (`--check` passing) proves the arch loads, nothing about the kernels.
**Use:** when a vendor image and a public image disagree, diff their selectors before you diff their
behaviour; and record *which* tree a fix would have to come from, so the next session doesn't re-pull.

**P-I · Pipeline parallelism never helps batch-1 decode, even where it's allowed.** The bubble
consumes exactly what the collectives would have cost: with < #stages in-flight microbatches, stage B
idles while stage A works, so the saved interconnect time reappears as dead time. PP pays only at
concurrency ≥ #stages.
**Use:** treat "switch TP→PP to dodge the all-reduce" as dead twice over on a batch-1 agentic box —
first the bubble (this pattern), then any runtime guard (e.g. §2's PLE `PP>1` block). Revisit only
after raising `--max-num-seqs` above the stage count.
*Mechanized from the Qwen3.8-Flash-Next FP8 optimization review (2026-09-16); the guard alone already
retired the attempt here — see PROJECT-TODOS A4.*

**P-J · On a comm-bound decode, collectives are a FIXED COST PER ROUND, not per request — batch to
amortize them.** The ~104 all-reduces per decode round (→ the 20–25 ms/token floor, §1) are paid once
per step regardless of how many sequences occupy it, so aggregate throughput scales roughly with the
number of concurrent sequences while per-token latency stays ~flat until saturation.
**Use:** when spin-wait forensics (§1 signature) says comm-bound, the cheapest throughput lever is
`--max-num-seqs > 1` at a shorter context (KV permitting), not a faster kernel. Sweep concurrency
1/2/4 and report aggregate + per-client latency separately; they move in opposite directions under
load. Measured baseline for the rule: FP8 Recipe C at 20.8 tok/s @ seqs=1 (2026-09-16 metrics).
*Generalized from this box's TP2 socket-path forensics; expect it on any NVLink-less TP setup.*

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
