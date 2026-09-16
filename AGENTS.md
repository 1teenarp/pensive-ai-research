# AGENTS.md — Operating manual for this workbench

> **Read this first, in full, before touching anything.** It is the contract for any agent
> (Claude Code, opencode, a local model doing computer-use, a human) working in this repo.
> It describes *how we work here*, not just *what broke here*.

---

## 0. What this repo is

A **local LLM serving workbench**. The loop it exists to run, over and over:

```
   read the system  →  intake a model  →  design a recipe  →  run it in stages
        ↑                                                          ↓
        └──────────  document the result  ←  measure & tune  ←─────┘
```

Five standing capabilities, in dependency order:

| # | Capability | Primary artifacts |
|---|---|---|
| **A** | **Read the machine** — know what the hardware can actually do, from cache or from scratch | `SYSTEM-SPEC.md`, `power-debug-collect.sh` |
| **B** | **Intake models** — inventory, acquire, extract facts, compute fit | `recipe/MODEL-CATALOG.md`, `recipe/hf_bulk_download_v3.py` |
| **C** | **Build recipes** — staged, gated, reproducible launchers for a model on this hardware | `recipe/<MODEL>-RECIPE.md` + `recipe/serve-<model>.sh` |
| **D** | **Measure & tune** — turn "it runs" into "it runs well", with evidence | recipe §Verdict, catalog forensics sections |
| **E** | **Triage failures** — classify software vs. host vs. hardware, and prove which | `power-trip-*.md`, `recipe/*-watch.sh`, `evidence/` |

**Capability E is a subsystem, not the mission.** This machine ("pensive") happens to have had a
long-running DIMM fault, so the failure-forensics tooling here is unusually good — reuse it, but do
not assume every problem is hardware. Historically **most run failures here were software or
capacity**, not the bad DIMM (see §10 and the Instance catalog).

---

## 1. Operating principles

These are earned, each from a specific wrong answer this repo produced at some point. Follow them.

| # | Principle | Why (the incident behind it) |
|---|---|---|
| **P1** | **Cached spec is a hypothesis; a live probe is truth.** Re-verify before you compute anything from it. | RAM config changed twice in one week (8→4→6 DIMMs); a driver upgrade silently broke every GPU container host-wide. Docs written on Monday were wrong by Wednesday. |
| **P2** | **One variable per attempt.** Change one flag, one stage, one env at a time; log what changed. | Multi-change attempts produced runs nobody could attribute. |
| **P3** | **Never hand-copy a launch command. Always go through the launcher script.** | Several flags here are *derived from live topology* (NUMA, GPU-local memory). A pasted command from a doc omits them and segfaults. Docs keep raw commands for reference only, marked as such. |
| **P4** | **Liveness ≠ working.** Verify the effect, not the invocation. | `powertrip-capture` was "running" while its klog had been stale for 2.5 h — the crash window was blind (Instances 3/6/7/8). `nvidia-smi -pl 250` "succeeded" while the cap never applied. Check mtime; read the value back. |
| **P5** | **Attribute with windowed evidence, never lifetime counters.** | 96 % of a channel's lifetime ECC count came from one unrelated burst; using the cumulative total as per-run evidence produced a wrong root cause for days. |
| **P6** | **Measure with counters, not impressions. Discard warmup.** | First requests after a restart are 3–5× slow (lazy compilation on the spec-decode path). Judging throughput from request #1 is how you "discover" a regression that isn't there. |
| **P7** | **Cheapest stage that can falsify the hypothesis, first.** | `--check` (CPU-only, seconds) has caught missing architectures and too-old FlashInfer before anyone burned an hour on a 300 GB weight load. |
| **P8** | **Serving is manual-only. Never set `restart: always` on a heavy load path.** | A load-time crash under auto-restart becomes a reboot loop on a machine that resets under load. |
| **P9** | **Negative results are results.** Every attempt gets a `RUN-LOG.md` entry before the next one starts — not a silent retry. | Instances 8 and 9 failed for two *different* reasons; without both written down the second looked like a repeat of the first. Raw serve logs are overwritten by the next launch, so unrecorded means gone. |
| **P10** | **Estimates are labelled as estimates.** Never present a computed guess as a measurement. | Fit math is a starting point; "expect ~5–10 tok/s — **measure, don't trust this line**" is the house style. Keep it. |
| **P11** | **Ask the user only for what you cannot probe.** Physical access, BIOS, interactive sudo, intent, priorities. Everything else: go find out. | Interactive-sudo commands (`ipmitool`, `drop_caches`) silently no-op in agent sessions; say so rather than reporting a gate as passed. |

### Things to ask the user (can't be probed)

Physical slot/board layout and whether hardware can be opened · BIOS/firmware access and which
settings exist on this board · an interactive-sudo password (or a decision to skip root-gated steps)
· HF token location and gated-repo access · **target workload** (latency vs. throughput, real
context length, expected concurrency) · which models matter next · **permission to run a known-risky
stage** on a machine with an open hardware suspicion.

---

## 2. Repo map — read in this order

| File | What it gives you |
|---|---|
| **`AGENTS.md`** (this) | The method. Entry point for any session. |
| **`SYSTEM-SPEC.md`** | Cached hardware/software spec + a **status banner** of currently-broken things. Read the banner first; it is the most-changing part of the repo. |
| **`recipe/MODEL-CATALOG.md`** | Every model on disk, fit verdict, and the proven Recipes A/B/C with measured numbers and bottleneck forensics. |
| **`recipe/<MODEL>-RECIPE.md`** | Per-model staged plan: §0 where it stands → §6 verdict. Template in §9. |
| **`recipe/serve-*.sh`** | The launchers. The only supported way to start a server. |
| **`RUN-LOG.md`** | **The attempt ledger** — every stage run, chronological, pass or fail. Read it to find out what has already been tried and why it was changed; append to it after every attempt (§9.1). |
| **`KNOWLEDGE.md`** | **Decision-shaping facts** — this hardware, each runtime image, each model *family*, and the generalized patterns. **Read at session start and before designing any recipe**; promote durable findings into it (§9.2). |
| `README.md` | Human-facing repo index. |
| `README-ramoffload-research.md` | Offload theory + the empirical RAM→GPU bandwidth results the fit math rests on. |
| `PROJECT-TODOS.md` | Open work, grouped serving vs. stability. |
| `RESUME-NOTE.md` | "Bring the current server back up" quick note. |
| `power-trip-diagnosis.md` / `power-trip-instances.md` | Build-specific hardware fault: root cause + the failure-event catalog (Instances 1–9). |
| `why-databric-syncflood-not-interceptable.md` | Educational: CE vs. UE, why a fabric flood can't be caught in software. |
| `evidence/` | Raw traces backing the above. |

---

## 3. Phase A — Read the system

**Goal:** a trustworthy answer to "what can this box actually run, right now?"

### A1. Preflight (run every session, before any GPU work — ~30 s)

```bash
# Host identity & uptime (did it reboot since the last note?)
uptime -s; uname -r

# GPUs: present, driver matched, memory free
nvidia-smi --query-gpu=index,name,memory.total,memory.used,power.limit,driver_version --format=csv
cat /proc/driver/nvidia/version              # loaded kernel module
modinfo nvidia 2>/dev/null | awk '/^version:/{print}'   # installed package  -> MUST MATCH

# CPU / NUMA / RAM — the numbers most likely to be stale in the docs
numactl -H | grep -E 'node [0-9]+ (size|cpus)'
free -h

# Interconnect & IOMMU (determines the cost of every TP collective)
nvidia-smi topo -m
grep -o 'iommu=[a-z]*' /proc/cmdline || echo "no iommu= flag set"

# GPU -> NUMA node, without nvidia-smi (works even when the driver is broken)
for f in /sys/bus/pci/devices/*/; do
  grep -q 0x10de "${f}vendor" 2>/dev/null && echo "$f -> numa_node=$(cat "${f}numa_node")"
done

# Runtimes available
docker images | grep -Ei 'vllm|sglang|llamacpp|tensorrt'
df -h /trunk/ai /buffer /   # model + image + log space
```

**Verdict to produce:** *ready* / *ready with caveats (list them)* / *blocked (name the blocker)*.
A driver/library mismatch, a memory-less GPU-local NUMA node, or <50 GB free on the model volume are
each **blocking** — say so and stop, don't launch into them.

### A2. Building a spec from scratch (new machine, or the cached spec is stale)

Run `bash power-debug-collect.sh` for the raw dump, then write `SYSTEM-SPEC.md` with these sections —
this structure is what the fit math in Phase B consumes:

1. **GPU** — model, VRAM, compute capability, driver/CUDA, power limits. Derive the **native
   low-precision formats** (`sm_120`/Blackwell → NVFP4 + FP8; `sm_90`/Hopper → FP8; `sm_86`/Ampere →
   INT8/AWQ/GPTQ only). This single line decides the whole quantization ladder in Phase C.
2. **CPU / NUMA** — sockets, cores, NUMA mode (NPS*), and **which NUMA node each GPU is on**.
3. **RAM** — total, per-node, available; DIMM count/type/speed. Mark it "re-verify live".
4. **Storage** — per mount: type, size, free, and **measured sequential read** (`fio` or a timed
   `dd`) for any volume that will serve weights. Load time and fabric stress both follow from it.
5. **Interconnect** — `nvidia-smi topo -m`, NVLink presence, PCIe gen/width, IOMMU mode. If P2P
   matters, measure it: CUDA `p2pBandwidthLatencyTest`, `nccl-tests`' `all_reduce_perf`.
6. **Host→GPU bandwidth** — measured pinned H2D GB/s. **Required** for any offload estimate.
7. **Runtimes** — vLLM/SGLang/llama.cpp/TensorRT-LLM images and versions, container toolkit.
8. **VRAM budget table** — usable-model-size per precision, per-GPU and aggregated (see B4).
9. **Status banner at the top** — dated, listing anything currently broken host-wide.

Every figure gets a **date**. Anything volatile gets an explicit "re-verify with `<command>`".

---

## 4. Phase B — Model intake

### B1. Inventory / acquire

```bash
ls -d /trunk/ai/huggingface/models/*/*                       # what's already here
du -sh /trunk/ai/huggingface/models/<org>/<model>            # on-disk truth

# Acquire (resumable; token resolved from $HF_TOKEN / $HF_TOKEN_FILE / ~/.cache/huggingface/token)
python recipe/hf_bulk_download_v3.py <org>/<model>           # single repo
python recipe/hf_bulk_download_v3.py models.csv --quant NVFP4  # bulk
```

Downloads are **resumable and hash-verified** — an interrupted transfer is re-run with the same
command, never restarted from scratch. Log to `/var/tmp/hf-<slug>.log` and check completion by
**shard count + total size**, not by "the command exited".

### B2. Extract the facts (before any GPU touches it)

**First: look the architecture up in [`KNOWLEDGE.md`](KNOWLEDGE.md) §3.** Most "new model" work is a
family already met here — a different quantization, size, or vendor re-export of something whose
constraints are already written down. A family entry can retire several planned attempts before you
start. If the arch string is new, you'll be writing that entry at the end (§9.2).

Read `config.json`, `hf_quant_config.json` (or `quantization_config`), and
`model.safetensors.index.json`. Produce the **model-facts table** (recipe §1):

architecture string · total / active params · layers, dense vs. sparse split · experts (routed,
per-token, shared) and `moe_intermediate_size` · attention type (dense MHA/GQA, MLA, linear/hybrid,
sparse) and KV geometry · MTP / draft layers · quantization scheme + **which modules are exempt**
(unquantized) · native context · KV-cache dtype support · required runtime versions · tool-call and
reasoning parser names (from the model card).

Also record **what the model card / vendor recipe targets** (e.g. "TP4 on GB200") — and note
explicitly that our box is not that, so their flags are a starting point, not a recipe.

### B3. Fit math

Use on-disk bytes as truth; fall back to `params × bytes_per_param` only when the repo isn't local.

```
bytes_per_param:   BF16/FP16 2 | FP8/INT8 1 | NVFP4/Q4 0.5

usable_vram    ≈ Σ(VRAM_i) × gpu_memory_utilization − ~2-4 GB/GPU (CUDA ctx + activations)
offload_total  ≈ max(0, weights_on_disk − offloadable_tables − usable_vram)
cpu_offload_gb ≈ ceil(offload_total / TP) + headroom     # per worker
```

**KV per token** — dense/GQA: `2 × layers × kv_heads × head_dim × kv_dtype_bytes`. MLA: compressed
`kv_lora_rank` only. Hybrid linear-attention models keep fixed-size recurrent state for most layers
and real KV for a handful → **KV is cheap, weights are the constraint**; do not size these models
with the dense formula.

**MoE offload stream (the decisive number for any offloaded MoE):**

```
GB_per_token   = sparse_layers × experts_per_token × expert_param_bytes × bytes_per_param
tok/s ceiling ≈ measured_H2D_GB/s / GB_per_token          # hard bandwidth ceiling, before compute
```

On this box (~10 GB/s pinned H2D per GPU) that predicted ~1 tok/s for a 305 GB FP8 MoE and ~2× that
for its 190 GB NVFP4 sibling — both matched reality. **Run this before promising anyone a number.**

**Host-RAM check, with NUMA:** total offload across workers must fit in *available* RAM **on the
nodes it will actually land on**. A memory-less GPU-local node pushes that worker's buffer onto
other nodes and a nominally-fitting budget OOMs the host (Instance 9). Check `numactl -H`, not just
`free -h`.

### B4. Verdict and catalog entry

Assign a tier, and add a row to `recipe/MODEL-CATALOG.md`'s quick-reference table:

| Tier | Meaning |
|---|---|
| ✅ **Proven** | Served end-to-end here, with measured tok/s and a working launcher. |
| 🟡 **Should fit** | Fit math says yes, untested. Needs a staged run. |
| 🟠 **Offload tier** | Only runs with host-RAM weight offload; expect bandwidth-bound decode. State the predicted ceiling. |
| 🔴 **Doesn't fit / unsupported** | Weights exceed VRAM+RAM, or no runtime supports the arch on this compute capability. Say which, and what would change it. |

---

## 5. Phase C — Choose the serving strategy

Decide these five, in order, and write the reasoning into recipe §2–3. Each decision names the
constraint that drove it. **Re-read [`KNOWLEDGE.md`](KNOWLEDGE.md) before you start** — §1–2 bound
what this hardware and the available images can do, §4's patterns apply even to a family nobody here
has run, and §5 lists the signals that look like blockers but aren't.

### C1. Quantization ladder (pick the highest the GPU supports natively)

Blackwell `sm_120` → **NVFP4** (fastest path proven here, 39–55 tok/s) → **FP8** → BF16.
Hopper `sm_90` → **FP8** → BF16. Ampere and older → INT8 / AWQ / GPTQ / GGUF.
Prefer **vendor-published quantized builds** (NVIDIA ModelOpt, RedHatAI, Unsloth) over quantizing
locally — they ship `hf_quant_config.json`, published accuracy deltas, and a reference serve recipe.
Note the measured lesson: **format changes decode speed structurally**, not just footprint — the FP8
batch-1 grouped-MoE kernel path ran ~3.4× slower than NVFP4 here at identical comm settings.

### C2. Parallelism

- **TP first** on a single node — it uses all devices on every layer. Standard guidance and our own
  results agree.
- **But TP cost scales with the interconnect.** With no NVLink and no P2P, every collective bounces
  through host RAM; here that put a **~20–25 ms/token floor** on TP2 and made ~104 blocking
  all-reduces per decode round the primary bottleneck. On such a box, seriously consider **one model
  per GPU at TP1** over TP2 for a model that fits.
- **PP** only when TP is blocked — and check for guards: vLLM's PLE CPU-offload path *refuses*
  `PP>1` here (hard dead end, not tunable).
- **EP** (`--enable-expert-parallel`) for large MoE: shards experts instead of replicating, can cut
  per-GPU footprint. Test it **alone**, it interacts with TP.
- **DP-attention + EP** is the large-scale MoE pattern (attention replicated, experts sharded); it
  needs more GPUs than a 2-GPU box to pay off. Know it exists; don't cargo-cult it here.

### C3. Offload

Weight offload (`--cpu-offload-gb`, per worker) when weights exceed VRAM. KV offload
(`OffloadingConnector`, SGLang layerwise) when KV is the constraint. Start with **more offload than
you think** and walk it down — an OOM during load costs 30–60 min, a slightly-oversized offload
costs a few tok/s.

### C4. Platform

**vLLM** is the default. Reach for an alternative when vLLM blocks: **SGLang** when the vendor's
proven path is SGLang or a backend is missing for your compute capability · **llama.cpp** for GGUF,
CPU/GPU layer splits, and small dense models · **KTransformers** for CPU-resident MoE experts +
GPU attention on a big-RAM box · **TensorRT-LLM** for maximum throughput on a fixed, static config.
Record *why* you switched — "vLLM has no sparse-MLA backend for sm_120" is a fact worth keeping.

### C5. Tuning levers, ranked by payoff here

1. **CUDA graphs on** (drop `--enforce-eager`) — up to ~4× decode. Biggest single lever. Its warmup
   burst is the riskiest moment on a load-fragile box, so prove it on dummy weights first.
2. **Speculative decoding** (MTP head, ngram, or draft model) — ~2.5–3× measured here even on a
   comm-bound setup. Start `num_speculative_tokens=2`, raise carefully: **5 crashed** with a ring-
   capacity/block-size divisibility error; 4 was the found ceiling. There is no clean formula — test.
3. **Context vs. concurrency** — KV scales linearly. Full native context forces `--max-num-seqs 1`,
   which exposes every collective serially. Dropping to 32–64k and raising concurrency to 2–4
   amortizes fixed per-round cost and often wins on aggregate throughput.
4. **`--kv-cache-dtype fp8`** — near-free on Blackwell, buys context or concurrency.
5. **Chunked prefill / prefix caching** — smooths TTFT under mixed traffic.
6. **Offload budget trim** — every GB back in VRAM is decode speed on an offloaded model.
7. **Fix the interconnect** — `iommu=pt`, restoring P2P, NUMA-local memory for each GPU. Structural,
   and the largest remaining lever on this box, but it's a host change: propose, don't just do it.

---

## 6. Phase D — The staged ladder (the core algorithm)

**Every new model goes through these stages, in order, no skipping.** Each stage is cheap relative
to the next and falsifies a distinct class of assumption. This is the pattern in
`serve-glm-53-flash.sh` / `serve-glm-53-flash-nvfp4.sh`; copy it.

| Stage | Command | Cost / risk | Proves | Gate to pass |
|---|---|---|---|---|
| **0 · check** | `--check` | seconds, **CPU-only, no GPU, no driver needed** | image exists; architecture is registered in the runtime; required library versions (e.g. FlashInfer ≥ x.y) satisfied; quant loader present | assertions print OK |
| **0.5 · present** | `--check` (model part) | seconds | model dir exists; shard count and total size match the repo | counts match |
| **1 · dummy** | `--dummy` | minutes; **no disk read of weights, no fabric burst** | arch instantiates; distributed init (NCCL) succeeds on this topology; attention/MoE backends select; offload allocator sizes; CUDA-graph capture survives | server reaches "model load complete / up" |
| **2 · serve** | `--serve` | 10–60 min load, **first high-risk event** | real weights load; KV fits; HTTP 200; coherent output | conservative config serves end-to-end |
| **3 · increments** | env overrides | one per session | each individual lever | record metric delta per lever |
| **4 · promote** | — | — | — | recipe §6 verdict + catalog row + launcher defaults updated |

**Stage-2 conservative baseline** (deliberately slow and safe): lowest useful `--max-model-len`,
`--max-num-seqs 1`, generous `--cpu-offload-gb`, `--enforce-eager`, `--max-parallel-loading-workers 1`,
serialized spawn. Load from the **slow, predictable** volume first if the box is load-fragile — a
steady 175 MB/s ZFS read is safer than a 3–5 GB/s NVMe burst.

**Stage-3 increment order** (one per session, measure after each): context → spec decode →
concurrency → KV dtype → CUDA graphs (if eager in stage 2) → EP → offload trim.

**If a stage fails, you stop and classify (§10) before changing anything.** A Stage-1 death is a
*software* finding — grab `docker logs`, the exit code, and the last log line before doing anything
else. Do not "just retry with different flags".

**Every stage ends with a `RUN-LOG.md` entry — pass or fail, before the next attempt** (§9.1). The
raw logs under `/var/tmp/serve-*.log` are overwritten by the very next launch, so an unrecorded
attempt is a permanently lost one. Start each session by reading the ledger's last few rows: it is
the fastest way to know what has already been tried and which variable moved last.

---

## 7. The launcher contract

Every recipe ships a `recipe/serve-<model-slug>.sh`. It is the executable form of the recipe doc and
**the only supported way to start a server**.

### Required interface

```
--check     Stage 0. CPU-only assertions inside the image. Safe to run anytime, on any host state.
--dummy     Stage 1. --load-format dummy. Arms safety, runs topology checks, launches.
--serve     Stage 2/3. Real weights. Everything --dummy does, plus cache drop.
--wait      Poll until the server binds its port, the container dies, or WAIT_TIMEOUT. Never mutates.
--logs      Follow the real runtime output (`docker logs -f`).
--stop      Remove the container. Idempotent.
--status    Container state + GPU memory + host RAM + HTTP readiness + log tail. Never mutates.
```

Long-running proven recipes may additionally expose `--start` / `--restart` / `--no-capture`
(see `serve-qwen38-flash-next-fp8.sh`). Keep the verbs consistent across scripts.

### Required behaviours

1. **Every tunable is an env var with a documented default**, overridable without editing the file:
   `MODEL IMAGE NAME PORT SERVED_MODEL TP CPU_OFFLOAD_GB MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL
   ENFORCE_EAGER SPEC_CONFIG EP GPU_POWER_CAP SERVE_LOG LOAD_FORMAT`. Mirror the list in recipe §5.
2. **Derive topology-dependent flags at launch; never hardcode them.** Probe and decide (the
   memory-less-NUMA-node → `NCCL_CUMEM_HOST_ENABLE=0` guard is the reference example). Log what was
   probed and what was chosen. This is what makes the script survive hardware changes that silently
   invalidate every doc.
3. **Auto-scale coupled defaults.** `CPU_OFFLOAD_GB` depends on `TP`; resolve it rather than letting
   a TP1 default silently double at TP2 (a real bug that was shipped here).
4. **Exclusivity.** Refuse to start while another serve owns the GPUs; name the command to stop it.
5. **Arm safety, then verify the effect** (P4): telemetry running **and writing** (mtime), power cap
   **read back**, CE/health baseline logged before launch.
6. **Preconditions fail loudly and early**, before the expensive path. `die()` with the fix in the
   message.
7. **Header comment = compressed history**: what's proven, what's measured, what crashed and at what
   value, which dead ends were ruled out. Read the top of `serve-qwen38-flash-next-fp8.sh` for the
   standard.
8. **`restart: no`. Always.** (P8)
9. **Conventions:** container `<slug>-serve` · port from a distinct block per model (8090, 8091,
   8092, …) · log `/var/tmp/serve-<slug>.log` · `set -u` · a `log()`/`die()` pair with a `[slug]`
   prefix.
10. **Point at the ledger on exit.** New launchers should remind the operator (or call
    `recipe/run-log.sh --container "$NAME" --stage <stage>` directly) so the attempt gets recorded
    while the container is still inspectable — `docker rm -f` destroys the exit code, the args, and
    the logs the snapshot needs.
11. **`$SERVE_LOG` is not the runtime log.** `docker run -d` returns immediately and prints only the
    container ID, so the redirect target holds a 64-char hash — useful only for errors that stop the
    run from starting at all. **Real output is in `docker logs`.** Say so in the launch message, and
    make `--status` tail `docker logs`, not the file.

### Waiting for a server (never blind-sleep)

A cold load is 5–60 minutes and there is no way to guess it. Do not chain `sleep 60; tail …` — that
burns turns and tells you nothing when the container has already died. Poll the three terminal
states together: **bound**, **dead**, **timed out**.

```bash
bash recipe/serve-<slug>.sh --wait          # preferred: the launcher knows its own port and name
```

The generic form, for a launcher that doesn't have it yet:

```bash
until curl -sf -o /dev/null localhost:$PORT/v1/models \
   || [ "$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null)" != "running" ]; do sleep 15; done
docker inspect -f 'status={{.State.Status}} exit={{.State.ExitCode}}' $NAME
```

Exit non-zero on death so the caller can branch, print the failure signature, and point at
`run-log.sh` — a `--wait` that just returns silently makes the operator go hunting anyway.

### Skeleton

```bash
#!/bin/bash
# serve-<slug>.sh — staged launcher for <org>/<model>.
# <one-paragraph summary: size, format, fit math, why this parallelism>
# Stages: --check (CPU-only) -> --dummy (no real weights) -> --serve (real load). See recipe/<MODEL>-RECIPE.md
# Proven: <measured config + tok/s>.  Dead ends: <what was ruled out, and the exact error>.
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/trunk/ai/huggingface/models/<org>/<model>}"
IMAGE="${IMAGE:-<image:tag>}"
NAME="${NAME:-<slug>-serve}";  PORT="${PORT:-80xx}"
TP="${TP:-2}";  CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-}"      # resolved by TP below
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}";   ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
SERVE_LOG="${SERVE_LOG:-/var/tmp/serve-<slug>.log}"

log(){ echo "[<slug>] $*"; }
die(){ log "ERROR: $*"; exit 1; }

check_image(){ ... }            # image present, else name the pull command
check_runtime(){ ... }          # Stage 0: docker run --entrypoint python3, assert arch + lib versions
check_model_present(){ ... }    # config only (dummy) vs all shards (serve)
require_gpus_free(){ ... }      # refuse while another serve holds the GPUs
resolve_offload_gb(){ ... }     # TP-dependent default
check_numa_topology(){ ... }    # probe GPU-local node memory -> decide NCCL env
check_capture_alive(){ ... }    # telemetry mtime, not container state
arm_safety(){ ... }             # telemetry + watchers + power cap (read back) + health baseline
drop_caches(){ ... }            # real load only; warn if it needs root
launch(){ ... }                 # $1 = dummy|real
status(){ ... }

case "${1:---help}" in
  --check)  check_runtime; check_model_present config ;;
  --dummy)  check_image; check_model_present config; require_gpus_free; resolve_offload_gb; \
            arm_safety; check_numa_topology; launch dummy ;;
  --serve)  check_image; check_model_present all;    require_gpus_free; resolve_offload_gb; \
            arm_safety; check_numa_topology; drop_caches; launch real ;;
  --stop)   docker rm -f "$NAME" >/dev/null 2>&1 && log "stopped" || log "not running" ;;
  --status) status ;;
  *) cat <<EOF
usage: $0 --check | --dummy | --serve | --stop | --status
env: IMAGE TP CPU_OFFLOAD_GB MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL ENFORCE_EAGER GPU_POWER_CAP
see recipe/<MODEL>-RECIPE.md for the staged plan and gates
EOF
  ;;
esac
```

---

## 8. Phase E — Measure and tune

### E1. Benchmark protocol

1. **Warm up and discard.** First requests after a restart carry one-time lazy compilation. Never
   report them.
2. **Use representative shapes** — real input/output lengths for the intended workload, not a toy
   prompt. Synthetic uniform traffic gives numbers that don't survive contact with use.
3. **Sweep concurrency** (1, 2, 4, 8, …) to find the saturation knee, rather than reporting one point.
4. **Record**: output tok/s (steady state), TTFT, inter-token latency, aggregate throughput, KV-cache
   utilisation %, VRAM per GPU, host RAM, and — if offloading — spec-decode acceptance rate.
5. **Prefer the server's own counters** to wall-clock:
   ```bash
   curl -s localhost:<PORT>/metrics | grep -E "^vllm:(inter_token_latency_seconds_(sum|count)|\
   spec_decode_num_(drafts|accepted_tokens)_total|generation_tokens_total|gpu_cache_usage_perc)"
   ```
   Standard harnesses when you want a full sweep: `vllm bench serve` (and `bench latency` /
   `bench throughput`), GuideLLM, NVIDIA GenAI-Perf.

### E2. Bottleneck forensics protocol

When a config works but is slow, do **not** start changing flags. Build the rule-in/rule-out table
(the format used in the catalog's "Generation-speed bottleneck investigation" — copy it):

| Observation | Value | Rules in / out |
|---|---|---|

Cover at minimum: inter-token latency · SM & memory clocks (throttling?) · power draw vs. limit
(compute-saturated?) · `utilization.memory` (VRAM-bandwidth-bound?) · GPU "util" **with no request in
flight** (a constant 99 % at low power is NCCL spin-wait, not work) · KV-cache usage (KV pressure?) ·
`topo -m` + NUMA (comm path cost) · `/proc/cmdline` (IOMMU) · spec-decode acceptance per position.

Then: **decompose one decode round** into its components (collectives × layers, kernel time, H2D
transfers), **rank the levers by payoff/effort**, and **name what's still unmeasured**. Profiling to
settle a two-way ambiguity (`VLLM_TORCH_PROFILER_DIR`, or nsys — the containers already carry
`SYS_PTRACE`) is usually a 10-minute job and beats a day of flag roulette.

---

## 9. Phase F — Document

### 9.1 First, always: the attempt ledger

**Append to [`RUN-LOG.md`](RUN-LOG.md) at the end of every stage, before touching anything else.**
This is the one documentation step that is never optional, including for routine passes and for
tuning increments that moved a number by 2 %.

```bash
bash recipe/run-log.sh --container <name> --stage check|dummy|serve|tune [--note "..."]
```

The helper snapshots the machine-readable half — host state (driver match, RAM, NUMA, repo rev),
exit code and OOM flag, duration, the full launch args, capture liveness, and the failure signature
pulled out of the container logs. You fill the judgement half, which is the part that has value later:

- **Changed vs. last attempt** — the ONE variable (P2). If you can't name it, you changed too many.
- **Read** — how far it got. Name the last stage reached (NCCL init → backend select → weight load →
  KV init → graph capture → serving) and what that **rules in or out**. A failure that got further
  than the last one is progress and must be recorded as such.
- **Lesson** — one transferable line. Would this be true on someone else's box?
- **Next** — the single next variable.

Then add the one-line row to the ledger table. Deep forensics for anything that reset or crashed the
*host* still goes to `power-trip-instances.md`; the ledger row links to it.

### 9.2 Then: promote what's durable to `KNOWLEDGE.md`

The ledger is *what happened*. [`KNOWLEDGE.md`](KNOWLEDGE.md) is *what we now know* — the facts that
change a decision on the next model, so the next recipe takes fewer iterations than this one did.

**Read it at the start of every session, and again before Phase C (strategy).** Most of the time the
right action is to read it and write nothing. Promote deliberately, not reflexively.

**A finding is promotable when all three hold:**

1. **Durable** — it outlives this attempt. "Offload at 32 GB OOM'd today" is not; "this runtime
   ignores `--max-parallel-loading-workers`" is.
2. **Decision-shaping** — it changes a flag, a strategy, or whether to attempt something at all. If
   knowing it wouldn't have changed what you did, leave it in the ledger.
3. **Not free to look up** — you can't get it from `config.json` or `--help` in ten seconds.

**Do not promote:** per-run narrative (ledger), current machine state (`SYSTEM-SPEC.md`), a model's
staged plan (its recipe), or a restatement of the model card.

#### Where it goes — the promotion ladder

```
observation            fact                                   pattern
(RUN-LOG entry)   →    (KNOWLEDGE §1 system / §2 runtime  →   (KNOWLEDGE §4)
                        / §3 model family)
one attempt            scoped: "this image", "this family"     mechanism explains why it generalizes
```

- **§1 System / §2 Runtime** — a property of this hardware or of a specific image. Name the image
  and version; a runtime fact without a version is a trap a year from now.
- **§3 Model family** — a property of an *architecture*, not a checkpoint. Write it under the arch
  string (`Qwen4ExpForConditionalGeneration`, `Glm5NextForConditionalGeneration`) so the next
  quantization, the next size, and the next vendor re-export all inherit it. This is the section
  that saves the most time: most "new model" work is a family you've already met.
- **§4 Pattern** — promote a §1–3 fact here when **the mechanism explains why it must generalize**,
  or when you've seen it in a **second family**. State it as a rule plus a *Use:* line, and cite the
  specific case it came from so a future reader can judge its reach.
- **§5 Known non-issues** — anything that looked like a failure and wasn't. Cheap to write, and it
  directly cancels a future rabbit hole.

#### How to write one

Terse: a bold claim, then the evidence, then **Use:** — what to actually do. Two to six lines. Tag
confidence honestly: `[measured]` here with numbers · `[observed]` once, mechanism understood ·
`[upstream]` from docs or source, unverified here · `[inferred]` reasoned, untested. **An
`[inferred]` entry is allowed and useful — an `[inferred]` entry mislabelled `[measured]` poisons
every decision built on it.**

When a later run contradicts an entry, **correct it in place and say what changed**. A knowledge base
nobody edits becomes a liability faster than one nobody writes.

### Recipe doc template — `recipe/<MODEL>-RECIPE.md`

```markdown
# <Model> on "<host>" — recipe
Date: <date>. Status: <one sentence: exactly which stage is proven, which is next>.
Launcher: recipe/serve-<slug>.sh  (--check → --dummy → --serve)

## 0. Where this stands (read before running anything)
  Stage-by-stage: what ran, when, what happened, link to the Instance entry if it died.
  Host-wide blockers that aren't this model's fault. What changed in the launcher since the last try.
## 1. Model facts        (the B2 table — arch, size, experts, attention, quant, context, parsers)
## 2. Fit math on this box   (the B3 numbers, with the live-state caveat and re-verify command)
## 3. Blockers & non-negotiable gates   (numbered, each with status: resolved / open / untested)
## 4. Staged plan        (Stage 0/1/2/3 command blocks, what each proves, what to watch)
## 5. Launcher reference (env var | default | notes table + the fixed flags and env)
## 6. Verdict            (is it worth running here, expected numbers marked as measured or estimated,
                          open questions in priority order)
```

### Also update

- **`recipe/<MODEL>-RECIPE.md` §0** — roll the ledger entries up into the current per-model status.
  The ledger is the sequence; §0 is the summary of where that sequence has arrived.
- **`recipe/MODEL-CATALOG.md`** — the quick-reference row and, for anything proven, a full
  "Recipe X" section with the measured config.
- **`power-trip-instances.md`** — any run that died, as a new Instance: timestamp, exact config, what
  it reached, last log line, host state at the time, evidence paths, classification (§10), and what
  it rules in/out. **Correct earlier entries** when later evidence contradicts them; there is a
  "Corrected findings" precedent — follow it rather than quietly rewriting history.
- **`PROJECT-TODOS.md`** — mark done, add what's newly open.
- **Status banners** — if you discovered something host-wide, put it at the top of `SYSTEM-SPEC.md`
  where the next session will hit it in the first 20 lines.

**Style:** date every claim · distinguish measured from estimated (P10) · keep the negative results ·
lead with what's currently blocking · when a number can go stale, print the command that re-checks it.

---

## 10. When a run dies — triage

**Classify before you theorise, and snapshot before you clean up.** Run
`bash recipe/run-log.sh --container <name> --stage <stage>` *first* — `docker rm -f` destroys the
exit code, launch args and logs that any later analysis depends on.

Four classes, distinguished by one question: *is the host still up?*

```bash
uptime -s                      # did the host reboot?
docker ps -a --filter name=<slug>-serve --format '{{.Status}}'
docker inspect <slug>-serve --format '{{.State.ExitCode}} OOMKilled={{.State.OOMKilled}}'
```

| Class | Signature | First moves |
|---|---|---|
| **1. Software** | container exited, host uptime unchanged | `docker logs --tail 200`; exit code; **the last log line names the stage** (backend select, MoE finalize, graph capture, KV init). Search the runtime's issue tracker for that line before theorising. |
| **2. Capacity / OOM** | `OOMKilled=true`, or kernel OOM killer in `journalctl -k`, host up | Recompute the B3 budget against **live `numactl -H`**, not nameplate. Cut offload or bind NUMA. Instance 9 is the worked example. |
| **3. Hang** | no exit, no progress, GPUs at 99 % util and low power | Collectives. Check NCCL env, `topo -m`, `NCCL_DEBUG=INFO`, `py-spy dump` on the worker. |
| **4. Host reset** | the box rebooted | Go to the reset-reason decode below. **This is the only class that is possibly hardware.** |

### Host-reset decode (build-agnostic method, AMD specifics below)

```bash
journalctl --list-boots | tail -8
grep -a "Previous system reset" /var/log/syslog /var/log/kern.log | tail -6
bash recipe/reset-reason-decoder.sh --history
bash recipe/ecc-per-window.sh --boot -1     # per-boot CE counts — the ONLY valid per-trip evidence (P5)
journalctl -k -b -1 | grep -aiE "Machine check|CECC|UECC|uncorrect|sync flood|OOM|panic"
```

AMD "Previous system reset reason" classes seen on this platform:

| Code | Meaning | Hardware? |
|---|---|---|
| `0x08000a00` | bit27 uncorrected error → **data-fabric sync flood** (+ bit9 thermal) | **Yes** — memory/fabric integrity fault |
| `0x00080a00` | bit19 **software wrote 0x6 to reset-control register 0xCF9** (+ bit9 thermal) | No — software-initiated warm reset (kernel panic path, NMI or BMC watchdog) |
| `0x00200a00` / `0x00200800` | bit21 **ACPI power-state transition** (± bit9 thermal) | No — power/firmware transition |

Then write the Instance entry (§9) **and** the ledger row (§9.1). A reset with **no fresh per-boot
CEs and a non-sync-flood code is not a memory fault** — say so plainly rather than defaulting to the
known suspect.

---

## 11. Appendix — build-specific: "pensive"

Everything in this section is **this machine's current situation and will go stale**. Treat
`SYSTEM-SPEC.md`'s banner as authoritative and re-verify (P1).

**Hardware:** AMD EPYC 7663 (56c/112t, 1 socket, **NPS4 / 4 NUMA nodes**) · 2× RTX PRO 5000 72 GB
Blackwell `sm_120`, **no NVLink, different root complexes / NUMA nodes** (GPU0→node 3, GPU1→node 0)
· 8× 128 GB DDR4-3200 RDIMM nameplate (1 TiB) · `/trunk/ai` ZFS 17 T (weights), `/buffer` NVMe
(images, telemetry), Seasonic 1600 W.

**Baked-in software workarounds** (all live in the launchers — this is why P3 exists):

1. `NCCL_P2P_DISABLE=1` — cross-NUMA, no NVLink; default P2P/CUMEM channels hang at `all_reduce`.
2. `--disable-custom-all-reduce` — CUSTOM all-reduce CUDA error on sm_120 (falls back to PYNCCL).
3. `--cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined` —
   the PLE-offload `pidfd_getfd` handshake.
4. `VLLM_QWEN38_PLE_FP8_SCALE=1` + patched `ple_layer.py` image — FP8-PLE selector bug (vLLM #54765).
5. **`NCCL_CUMEM_HOST_ENABLE=0`, auto-detected per launch** — `ncclCuMemHostEnable`→`cuMemCreate`
   segfaults (rather than degrading) when a GPU's local NUMA node has 0 MB. Probed every launch, so
   it self-corrects as DIMMs move.

**The long-running hardware story** (root-caused): repeated `0x08000a00` sync-flood resets under heavy
weight loads, traced to **two bad DIMMs** — the channel-G/MM4 suspect (memtest86+ errors) and a second
stick with physically damaged PCB pads (silent; memtest alone would not have caught it). Both pulled
for RMA. Full evidence in `power-trip-diagnosis.md` and `power-trip-instances.md` (Instances 1–9).

**EDAC channel → physical slot** (for RMA paperwork; board-specific):
`A=MM7 B=MM5 C=MM3 D=MM1 E=MM8 F=MM6 G=MM4 H=MM2`.

**Standing gates for any heavy load on this box:** telemetry armed **and writing** · `edac-ce-watch.sh`
running with a logged baseline · GPU power cap 250 W **read back** · serialized loading
(`--max-parallel-loading-workers 1`, `spawn`) · no other serve running · caches dropped before a real
load · NUMA guard probed. The 250 W cap is a defensive margin, **not** the stabiliser — the real
levers are burst reduction and fixing the memory.

**Why the reset can't be caught in software:** an uncorrectable error means the machine can't be
trusted; the fabric sync flood is a fail-safe. The only actionable gate is **pre-fault** — watch
corrected-ECC deltas and act before a CE escalates to a UE. See
`why-databric-syncflood-not-interceptable.md`.

---

## 12. Porting this workbench to another machine

The method (§§1–10) is portable; the appendix is not. To move:

1. Rebuild `SYSTEM-SPEC.md` from §A2 on the new host. **Delete, don't adapt, the old numbers.**
2. Re-derive the quantization ladder from the new compute capability (§C1).
3. Re-measure H2D bandwidth and storage read rates — all offload math depends on them (§B3).
4. Re-probe the interconnect. `NCCL_P2P_DISABLE=1` and friends are **fixes for this box's topology**;
   on an NVLink host they are a large, silent performance loss. Remove what isn't needed and prove it.
5. Replace §11 with the new host's own quirks, and keep the failure catalog going — same format.
6. Keep: the staged ladder, the launcher contract, the documentation templates, and the principles.
   Those are the parts that were expensive to learn.

---

**Sources for the general-practice sections (§5, §8):**
[vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) ·
[vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) ·
[vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) ·
[vLLM Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) ·
[Red Hat — Practical strategies for vLLM performance tuning](https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning) ·
[Google Cloud — vLLM Performance Tuning](https://cloud.google.com/blog/topics/developers-practitioners/vllm-performance-tuning-the-ultimate-guide-to-xpu-inference-configuration).
Everything else is from this repo's own measured results.
