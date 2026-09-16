# pensive-ai-research

**Running frontier-class LLMs on one workstation — the research, the recipes, and the tooling that
makes it repeatable.**

"pensive" is a single-socket AMD EPYC workstation with two NVIDIA Blackwell RTX PRO 5000 72 GB cards
and ~751 GiB of RAM. This repo is the working record of everything it took to get **125B- and
320B-parameter models actually serving on it** — what fits, what it costs, which flags matter, which
blockers were real, and how to do it again for the next model.

> Mostly co-developed with a **locally-hosted LLM driven through opencode**, with the assistant
> authoring most of the scripts and documentation here interactively. The repo is written to be read
> by agents as much as by people — [`AGENTS.md`](AGENTS.md) is the operating manual.

---

## What this repo actually is

Three things, in order of what's most reusable to someone else:

1. **A method.** A tested way to take an arbitrary model from "a folder of safetensors" to "a served
   endpoint with measured throughput" on hardware it doesn't obviously fit on — staged, gated, and
   documented so a failed attempt still teaches you something. → [`AGENTS.md`](AGENTS.md)
2. **Proven recipes with real numbers.** Not "should work" — measured tok/s, measured VRAM residency,
   measured acceptance rates, and the full list of what was tried and rejected.
   → [`recipe/`](recipe/)
3. **Research findings and forensics.** The offload bandwidth law, the cross-NUMA NCCL story, the
   FP8-vs-NVFP4 kernel gap, and a nine-instance hardware failure investigation that ended in a
   root cause. → the research docs below

It is *not* primarily a hardware-fault repo. This machine had a genuine memory fault and the
forensics tooling here is unusually good because of it — but most run failures on this box turned out
to be **software or capacity**, and that is where most of the work went.

---

## The machine

| Component | Spec |
|---|---|
| **GPU** | 2× NVIDIA RTX PRO 5000 72 GB Blackwell (`sm_120`) — ~147 GB VRAM, **~128–135 GB usable** for a single serving process. Native FP8 **and NVFP4**. |
| **Interconnect** | **No NVLink**; the two cards sit on different root complexes / NUMA nodes (`SYS` in `topo -m`). This single fact shapes almost every recipe here. |
| **CPU** | AMD EPYC 7663 — 56c/112t, 1 socket, **NPS4 (4 NUMA nodes)** |
| **RAM** | 1 TiB nameplate (8× 128 GB DDR4-3200 RDIMM); **~751 GiB live** across all 4 nodes (2 sticks out for RMA) |
| **Host→GPU** | **~8.6 GB/s pageable, ~10.2 GB/s pinned** (measured) — the hard ceiling on every offloaded model |
| **Storage** | `/trunk/ai` ZFS 17 T (~175 MB/s, model library) · `/buffer` NVMe 1 T (images, telemetry) · root NVMe 512 G |
| **Software** | Docker + vLLM (incl. a patched image) + SGLang + llama.cpp, NVIDIA Container Toolkit |

Full detail, with a "currently broken" banner at the top: [`SYSTEM-SPEC.md`](SYSTEM-SPEC.md).

---

## What runs today — measured

| Model | Format / size | Config | Measured | Recipe |
|---|---|---|---|---|
| **Qwen3.8-Flash-Next-FP8** | FP8 MoE, 125B/6B, 173 GB | TP2, 8 GB/wkr weight offload, **native 262,144 ctx**, CUDA graphs + MTP spec decode (4) | **~20–24 tok/s** steady state (vs ~11.5 graphs-only, ~8 eager); 54.5 GiB/rank resident | Recipe C |
| **Qwen3.8-Flash-Next-NVFP4** | NVFP4 MoE, 125B/6B, 124 GB | TP2, native 262,144 ctx, CUDA graphs, PLE offloaded to RAM | **~39–55 tok/s** (~4× the eager baseline) | Recipe A |
| **Qwen3.8-27B-NVFP4** | NVFP4 dense, 22 GB | single GPU | comfortable; the fast/lean tier | Recipe B |
| **Qwen3.5-397B-A17B-NVFP4** | NVFP4 MoE, 223 GB | TP1 + 200 GB host-RAM weight offload | **~1.0 tok/s** — and we can tell you exactly why (see below) | offload research |

Recipes A/B/C with the complete flag sets, the rejected alternatives, and the bottleneck forensics:
[`recipe/MODEL-CATALOG.md`](recipe/MODEL-CATALOG.md).

**In flight:** `nvidia/GLM-5.3-Flash-NVFP4` (320B/18B MoE, NVFP4, 1M context) — 190 GiB, 33 shards,
**download completed 2026-09-14**; staged plan and launcher ready, Stage 0 is the next action.
→ [`recipe/GLM-53-FLASH-NVFP4-RECIPE.md`](recipe/GLM-53-FLASH-NVFP4-RECIPE.md)

---

## How work is done here — the staged ladder

Every model goes up the same ladder. Each rung is cheap relative to the next and falsifies a
different class of assumption, so a failure tells you *which* assumption was wrong.

| Stage | Command | Cost | Proves |
|---|---|---|---|
| **0 · check** | `--check` | seconds, **CPU-only** | architecture registered in the runtime, library versions satisfied, quant loader present |
| **1 · dummy** | `--dummy` | minutes, **no weight read** | distributed init on this topology, backend selection, offload sizing, CUDA-graph capture |
| **2 · serve** | `--serve` | 10–60 min, first real risk | weights load, KV fits, HTTP 200, coherent output |
| **3 · increments** | env overrides | one lever per session | context → spec decode → concurrency → KV dtype → graphs → EP |

Plus the rules that made the difference: **one variable per attempt** · **never hand-copy a launch
command** (several flags are derived from live NUMA topology at launch) · **liveness ≠ working**
(check that the telemetry file is growing, read the power cap back) · **windowed evidence, never
lifetime counters** · **discard warmup before quoting throughput** · **write down the failures**.

The full method — reading a system from scratch, model intake and fit math, strategy selection, the
launcher contract, benchmarking, failure triage, and the documentation templates — is
[`AGENTS.md`](AGENTS.md).

---

## Quick start

```bash
# Bring up the current server (arms telemetry + watchers + power cap, then serves)
bash recipe/serve-qwen38-flash-next-fp8.sh --restart
bash recipe/serve-qwen38-flash-next-fp8.sh --status
bash recipe/serve-qwen38-flash-next-fp8.sh --stop

# Take a new model up the ladder (cheap stage first — --check needs no GPU at all)
bash recipe/serve-glm-53-flash-nvfp4.sh --check
bash recipe/serve-glm-53-flash-nvfp4.sh --dummy
bash recipe/serve-glm-53-flash-nvfp4.sh --serve

# Acquire a model (resumable, hash-verified, token auto-resolved)
python recipe/hf_bulk_download_v3.py nvidia/GLM-5.3-Flash-NVFP4

# Something died — classify before theorising
bash recipe/reset-reason-decoder.sh --history   # did the HOST reset, and why?
bash recipe/ecc-per-window.sh --boot -1         # fresh memory errors in that boot only
bash recipe/diag-dimm-fault.sh                  # full, RMA-ready DIMM report
```

Serving is **manual-only by design** (`restart: no`) — an auto-restarting heavy load on a
load-fragile box is a reboot loop. Same for the telemetry containers: armed per session, not on boot.

---

## Research findings worth knowing

The transferable results — the parts that would still be true on someone else's box.

### Interconnect and parallelism
- **Cross-NUMA, no-NVLink GPUs hang NCCL.** Default P2P/CUMEM channels complete
  `init_process_group` and then never finish `all_reduce`. Reproducible on **both vLLM and SGLang**.
  Fix: `NCCL_P2P_DISABLE=1` (falls back to the socket path). → `PROJECT-TODOS.md` A2
- **TP is not free when the wire is slow.** With collectives bouncing through host RAM, one decode
  round is ~104 *blocking* all-reduces — the GPUs sit at 99 % "utilization" drawing 100 W of 300 W
  with 9 % memory throughput. That signature is **NCCL spin-wait, not work**, and it put a
  ~20–25 ms/token floor under everything. On such a box, one model per GPU at TP1 can beat TP2.
- **A memory-less GPU-local NUMA node segfaults NCCL** (`ncclCuMemHostEnable` → `cuMemCreate`)
  instead of degrading. The launchers probe for it every launch and set `NCCL_CUMEM_HOST_ENABLE=0`
  only when needed — a derived flag, not a hardcoded one.
- **Pipeline parallelism is a hard dead end** for models needing vLLM's PLE CPU offload: there's an
  explicit guard against `PP>1`. Not tunable.

### Offload economics
- **The bandwidth law.** For an offloaded MoE, decode speed is predicted by
  `active_weight_bytes_per_token ÷ host→GPU H2D bandwidth`, before compute enters the picture.
  ~9.5 GB/token ÷ ~10 GB/s ≈ **~1 tok/s** — and the 397B model measured exactly 1.0 tok/s. Halving
  the weight bytes (NVFP4 instead of FP8) roughly doubles it. Compute is *not* the constraint: the
  GPU showed 100 % SM with ~0 % HBM utilization — starved on weight delivery.
- **KV-cache offload does not extend one request's context.** `--kv-offloading-size` is for
  cross-request prefix reuse; **weight** offload is the lever that buys you context headroom.
- **Staging weights to NVMe is a cold-start optimization only** — it doesn't reduce the engine's
  tensor-RAM requirement. And ZFS ARC caches raw on-disk bytes, which is a separate allocation from
  the engine's deserialized tensors.
- **Budget against live NUMA, not nameplate RAM.** A nominally-fitting offload budget OOM'd the whole
  host because one worker's buffer had to land on remote nodes. → Instance 9
- Full theory and the empirical table: [`README-ramoffload-research.md`](README-ramoffload-research.md)

### Quantization and kernels
- **Format changes speed structurally, not just footprint.** The FP8 batch-1 grouped-MoE kernel path
  ran ~3.4× slower than NVFP4 at identical comm settings. CUDA graphs bought ~4× on NVFP4 but only
  ~1.4× on FP8 — which is itself evidence about where each one's bottleneck lives.
- **Prefer vendor-published quantized builds** (NVIDIA ModelOpt, RedHatAI, Unsloth): they ship
  `hf_quant_config.json`, published accuracy deltas, and a reference serve recipe.
- Runtime support is checkpoint-specific, not just arch-specific: GLM-5.2-NVFP4 failed **seven
  different ways** across vLLM and SGLang (no Blackwell sparse-MLA backend; a fused-MoE loader shape
  mismatch after successfully loading 656 GiB into host RAM), while GLM-5.3-Flash's newer runtime
  initialized `FLASHINFER_MLA_SPARSE_SM120` cleanly on the same hardware.

### Throughput levers, ranked by what they actually paid here
CUDA graphs (up to ~4×) → **MTP speculative decoding** (~2.5–3×, at 65–70 % per-token acceptance,
even on a comm-bound setup) → trading context for concurrency → FP8 KV → offload trim. Note
`num_speculative_tokens=5` **crashes** (`QSA ring capacity 12 must divide the attention block size
1616` — computed dynamically, no clean formula); 4 was the tested ceiling. Measure with the server's
own `/metrics` counters and throw away the first couple of requests — lazy compilation on the
spec-decode path makes them wildly unrepresentative.

### Host and kernel
- An apt upgrade can update the NVIDIA userspace libraries while leaving the **old kernel module
  loaded** — breaking `nvidia-smi` and every GPU container host-wide, with an error that looks
  model-specific but isn't. Always compare `/proc/driver/nvidia/version` against `modinfo nvidia`.
- `iommu=pt` is absent on this host and is the leading suspect for the historic P2P hangs — the
  largest untaken architectural lever here. → `recipe/MODEL-CATALOG.md`, bottleneck investigation

---

## The model library

~46 model repos (~14 TB) under `/trunk/ai/huggingface/models/`, each with a fit verdict against the
~128–135 GB usable-VRAM budget — from 4 GB TTS models up to a 1.8 TB MoE that will never run here.
The rough shape:

| Tier | Examples |
|---|---|
| **Comfortable, 1 GPU** | 27B dense at NVFP4/FP8, 35B/3B MoE, OCR/TTS/vision models |
| **2 GPUs, proven** | Qwen3.8-Flash-Next at NVFP4 and FP8 (125B/6B, 262k ctx), Qwen3-Coder-Next-FP8 |
| **2 GPUs, tight** | 122B MoE FP8, DeepSeek-V4-Flash NVFP4 (~150 GB) |
| **Offload tier** | 397B-A17B NVFP4 (~1 tok/s), GLM-5.3-Flash FP8 305 GB / NVFP4 190 GiB |
| **Out of reach** | Kimi-K3 (1.4 TB), Inkling (1.8 TB), DeepSeek-V4-Pro (786 GB), MiniMax-M3 (664 GB) |

Full table with sizes, types and per-model verdicts: [`recipe/MODEL-CATALOG.md`](recipe/MODEL-CATALOG.md).

---

## The hardware chapter

The box used to hard-reset under heavy memory load. That investigation ran across **nine documented
failure instances** and ended in a root cause: **two bad DIMMs** — the channel-G/MM4 suspect that
threw memtest86+ errors, and a second stick with physically damaged PCB pads that memtest alone would
never have caught. Both are out for RMA; all four NUMA nodes are populated again at ~751 GiB.

What makes it worth reading even if your hardware is fine:

- **Three distinct reset classes, decoded** — a data-fabric sync flood (`0x08000a00`, real memory
  fault) vs. a software `0xCF9` warm reset (`0x00080a00`) vs. an ACPI transition (`0x00200a00`).
  Conflating them costs days. → `recipe/reset-reason-decoder.sh`
- **Cumulative ECC counters lie for per-event attribution.** 96 % of one channel's lifetime errors
  came from a single unrelated burst. Per-boot windows are the only valid evidence.
  → `recipe/ecc-per-window.sh`
- **Why a fabric sync flood cannot be intercepted in software** — an accessible explanation of
  corrected vs. uncorrectable errors and why the reset is a fail-safe, not a malfunction.
  → [`why-databric-syncflood-not-interceptable.md`](why-databric-syncflood-not-interceptable.md)
- **A reusable fault-isolation runbook** and RMA-ready diagnostic output.
  → [`power-trip-diagnosis.md`](power-trip-diagnosis.md), `recipe/diag-dimm-fault.sh`

Evidence trail: [`power-trip-instances.md`](power-trip-instances.md) (Instances 1–9, including the
corrections where later evidence overturned an earlier conclusion) and [`evidence/`](evidence/).

---

## Tooling

| Tool | What it does |
|---|---|
| `recipe/serve-*.sh` | Staged launchers — the only supported way to start a server. Derive topology-dependent flags live, arm safety, refuse unsafe preconditions. |
| `recipe/hf_bulk_download_v3.py` | Resumable, hash-verified HF downloads; single-repo or CSV bulk; token resolved from env/file/cache without manual pasting. |
| `recipe/run-log.sh` | Snapshots an attempt into a pre-filled [`RUN-LOG.md`](RUN-LOG.md) entry — host state, exit code, launch args, capture liveness, failure signature — so recording a run costs one command. |
| `powertrip-capture.sh` + `run-powertrip-capture.sh` | Per-second crash-safe telemetry to `/buffer/powertrip/` (survives reboot): CPU Tctl + chiplets, RAPL watts, both GPUs, raw dmesg, EDAC table, DIMM map. |
| `recipe/edac-ce-watch.sh` | Corrected-ECC **pre-fault** alerting on per-interval deltas (not cumulative totals). |
| `recipe/ecc-per-window.sh` · `diag-dimm-fault.sh` · `reset-reason-decoder.sh` | Windowed ECC accounting, RMA-ready DIMM diagnostics, AMD reset-reason decoding. |
| `power-debug-collect.sh` | One-shot full system snapshot — the basis for rebuilding `SYSTEM-SPEC.md`. |

A Prometheus-style metrics stack (Grafana Alloy + node, DCGM and RAS exporters) also runs on the
host; it isn't documented in this repo yet.

---

## Repo map

| Path | What it is |
|---|---|
| [`AGENTS.md`](AGENTS.md) | **Operating manual** — read first. Method, staged ladder, launcher contract, triage, templates. |
| [`SYSTEM-SPEC.md`](SYSTEM-SPEC.md) | Hardware/software spec + live status banner of anything currently broken. |
| [`recipe/MODEL-CATALOG.md`](recipe/MODEL-CATALOG.md) | Every model on disk, fit verdicts, Recipes A/B/C with measured numbers and bottleneck forensics. |
| [`recipe/GLM-53-FLASH-RECIPE.md`](recipe/GLM-53-FLASH-RECIPE.md) · [`-NVFP4-`](recipe/GLM-53-FLASH-NVFP4-RECIPE.md) | Staged per-model recipes for the 320B GLM-5.3-Flash pair (FP8 and NVFP4). |
| [`README-ramoffload-research.md`](README-ramoffload-research.md) | Serving models larger than VRAM: theory, provider comparison, empirical results, bottleneck benchmark. |
| [`README-3gpu.md`](README-3gpu.md) | What a third GPU would unlock (~220 GB VRAM) — speculative capability assessment. |
| [`why-databric-syncflood-not-interceptable.md`](why-databric-syncflood-not-interceptable.md) | Educational deep-dive: CE vs. UE, fabric floods, why software can't catch it. |
| [`power-trip-diagnosis.md`](power-trip-diagnosis.md) · [`power-trip-instances.md`](power-trip-instances.md) | Root-cause analysis + fix runbook; the nine-instance failure catalog. |
| [`dmesg-report.md`](dmesg-report.md) | Kernel/dmesg error survey (MCE history, network link-flap). |
| [`powertrip-capture-readme.md`](powertrip-capture-readme.md) | What the telemetry logger captures and how. |
| [`KNOWLEDGE.md`](KNOWLEDGE.md) | **Decision-shaping facts** — this hardware, each runtime image, each model *family*, the generalized patterns, and the known non-issues. Read before designing a recipe. |
| [`RUN-LOG.md`](RUN-LOG.md) | **Attempt ledger** — every stage run against every model, chronological, pass or fail, with the one variable that changed and what was learned. |
| [`RESUME-NOTE.md`](RESUME-NOTE.md) | "Bring the server back up" quick note + standing decisions. |
| [`PROJECT-TODOS.md`](PROJECT-TODOS.md) | Open and closed work, split serving vs. stability. |
| [`evidence/`](evidence/) | Raw traces: telemetry CSVs, kernel and serve logs, reset reasons. |

---

## Status snapshot — 2026-09-15 (live-verified)

- **Host healthy.** NVIDIA driver matched again at 580.178.04 (the 2026-09-10 module/library mismatch
  that blocked every GPU container is resolved). All 4 NUMA nodes populated: 129 / 258 / 129 / 254 GB,
  **751 GiB total, ~543 GiB available**.
- **No server currently running**; `powertrip-capture` is armed. Recipe C
  (`serve-qwen38-flash-next-fp8.sh --restart`) is the one-command path back to a serving box.
- **`nvidia/GLM-5.3-Flash-NVFP4` — blocked, and now understood.** Stage 1 (`--dummy`) ran tonight and
  died at KV-cache init (`pe_dim must be 64 for fp8_ds_mla`). Reading the image's backend selector
  showed the wall is structural: on **sm_120** the only surviving MLA backend requires `fp8_ds_mla`,
  whose kernel assumes the DeepSeek rope shape, while this model is **NoPE** (`qk_rope_head_dim: 0`).
  Upstream implements exactly this shape — for **SM90/Hopper only**. No flag bridges it. Everything
  upstream of attention proved out: the ModelOpt-NVFP4 loader works, TP2 NCCL is clean, backend
  selection succeeds. Forward paths: newer vLLM, SGLang, or Hopper.
  → [`RUN-LOG.md`](RUN-LOG.md) R-014, generalized as [`KNOWLEDGE.md`](KNOWLEDGE.md) P-H
- **Open questions, in priority order:** can any runtime serve GLM-5.3-Flash on sm_120 (newer vLLM or
  SGLang); does Instance 8's `0xCF9` software reset reproduce on a non-FP8 MoE finalize path; would
  `iommu=pt` restore PCIe P2P and lift the TP2 latency floor.
- **Outstanding hardware:** 2 DIMMs out for RMA; back to the full 1 TiB / 8-DIMM config when the
  replacements land.

---

## Notes on safety and honesty

This box has served 125B-parameter models at native 262k context while carrying a known marginal
DIMM, and the docs capture all of it — the evidence, the mitigations, the dead ends, and the
corrections where a confident earlier conclusion turned out to be wrong. Two house rules that the
rest of the repo depends on:

- **Estimates are labelled as estimates.** A computed fit number is never presented as a measurement.
  "Expect ~5–10 tok/s — measure, don't trust this line" is the style.
- **Negative results are kept.** Every failed run gets an entry with its evidence. Two attempts that
  fail for two different reasons are two findings, not one repeat.

Licensed under the terms in [`LICENSE`](LICENSE).
