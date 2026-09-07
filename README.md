# pensive-ai-research

**A living knowledge base for a single-socket EPYC workstation ("pensive") used to run and study
large local LLMs with hosted NVIDIA Blackwell GPUs** — serving, hardware debugging, self-built AI
tooling, and the engineering journey around running frontier models on consumer-grade hardware.

This repo documents **what the machine is, what models it runs (and how), what has failed and why,
and the exact working recipes** — so it can be referenced, shared with LLM agents / humans for
development, and used to unlock **larger and newer models** later.

> Mostly co-developed with a **locally-hosted LLM** used through **opencode** doing computer-use, with
> the assistant authoring most scripts and documentation here interactively.

---

## Project purpose

This is not just diagnostics — it's an **experimentation + learning + development workspace** that
tracks:

- **System specs** and hardware capability (what can actually run here).
- A **model catalog** — all models present on disk and their **fit verdict** for this hardware.
- **Run attempts** — every time a model is served, with the working (or failing) recipe.
- **Hardware troubleshooting** — a recurring, deeply-diagnosed power-trip fault and its evidence trail.
- **Tooling** — capture/monitoring scripts, launch configurations, and DIMM diagnostics.
- **Progress** — TODOs, resume notes, and what has been unlocked over time.

The goal: a long-lived, well-structured reference that makes it easy to (a) know what's here,
(b) reproduce a working setup, and (c) keep pushing to **bigger and newer models**.

---

## Repository layout

| Path | What it is |
|---|---|
| **[`README.md`](README.md)** | This overview/index. |
| **[`AGENTS.md`](AGENTS.md)** | **Power-trip investigation runbook** — the entry point for any fresh session/agent to pick up the hardware-fault analysis without prior history. |
| **[`SYSTEM-SPEC.md`](SYSTEM-SPEC.md)** | Hardware spec: GPUs, CPU, RAM, storage, software stack, VRAM budget, NUMA/topology. |
| **[`recipe/MODEL-CATALOG.md`](recipe/MODEL-CATALOG.md)** | Model catalog + recipes for all models on disk, with fit verdicts for this machine. |
| **[`recipe/serve-qwen38-flash-next-nvfp4.sh`](recipe/serve-qwen38-flash-next-nvfp4.sh)** | One-shot launcher for the currently-working server (all settings baked in). |
| **[`recipe/diag-dimm-fault.sh`](recipe/diag-dimm-fault.sh)** | DIMM fault diagnostic → warranty/RMA-ready output. |
| **[`power-trip-diagnosis.md`](power-trip-diagnosis.md)** | Root-cause analysis + fix runbook for the recurring power-trip. |
| **[`power-trip-instances.md`](power-trip-instances.md)** | Failure-event catalog (Instances 1–5) + evidence archive. |
| **[`why-databric-syncflood-not-interceptable.md`](why-databric-syncflood-not-interceptable.md)** | Educational deep-dive: why a data-fabric sync-flood can't be gracefully intercepted. |
| **[`README-ramoffload-research.md`](README-ramoffload-research.md)** | Research: serving models larger than VRAM via RAM+VRAM offload; empirical results. |
| **[`README-3gpu.md`](README-3gpu.md)** | Speculative 3-GPU build assessment (capability + what it would enable). |
| **[`powertrip-capture-readme.md`](powertrip-capture-readme.md)** | Crash-capture telemetry logger docs (what/how it logs). |
| **[`RESUME-NOTE.md`](RESUME-NOTE.md)** | Live "how to bring the server back up" resume note + decisions. |
| **[`PROJECT-TODOS.md`](PROJECT-TODOS.md)** | Tracked tasks (model serving / stability). |
| Scripts | `power-debug-collect.sh`, `powertrip-capture.sh`, `run-powertrip-capture.sh` (+ `recipe/` scripts). |
| `evidence/` | Raw captured traces (telemetry CSVs, kernel/serve logs, reset reasons) backing the diagnosis. |

---

## Quick start

### The currently-working model server (Qwen3.8-Flash-Next-NVFP4)

```bash
bash recipe/serve-qwen38-flash-next-nvfp4.sh --restart
```

This scripts the full working configuration — patched vLLM image, TP2, native **262,144** context,
CUDA graphs (~39–55 tok/s), PLE CPU offload, and all the fabric-burst-reduction flags — plus it arms
the crash-capture logger and sets the GPU power cap. Customize via `MAX_MODEL_LEN`, `MAX_NUM_SEQS`,
`GPU_POWER_CAP`, `ENFORCE_EAGER`.

### Diagnosing a suspected faulty DIMM

```bash
bash recipe/diag-dimm-fault.sh
```

Prints per-channel CE/UE counts, maps EDAC channel → physical DIMM slot, and the reset-reason history —
enough to file an RMA for the failing stick.

### Monitoring / telemetry

```bash
bash run-powertrip-capture.sh start     # arm the crash-capture logger
bash run-powertrip-capture.sh status
```

Writes per-second CPU/GPU telemetry + raw kernel log + EDAC to `/buffer/powertrip/` (persistent across
reboots). The `powertrip-capture` container is **manual** (`restart: no`) — it is armed by the
`--start` script, not auto-start, so debug/hardware logging only runs during a model-load session.

---

## Key topics

### Hardware / capability
- **System specs** → [`SYSTEM-SPEC.md`](SYSTEM-SPEC.md)
- **Model fit / VRAM budget** → [`recipe/MODEL-CATALOG.md`](recipe/MODEL-CATALOG.md)
- **3-GPU feasibility** → [`README-3gpu.md`](README-3gpu.md)

### The recurring power-trip (the biggest hardware story)
The machine has a **data-fabric sync-flood reset** (`0x08000a00`) caused by a **marginal DIMM on
channel G / slot MM4** that escalates from corrected-ECC to an uncorrectable error under heavy memory
load.
- **Root cause + fix runbook** → [`power-trip-diagnosis.md`](power-trip-diagnosis.md)
- **Failure-event catalog** → [`power-trip-instances.md`](power-trip-instances.md)
- **Why it can't be intercepted (+ CE vs. UE, fabric floods)** → [`why-databric-syncflood-not-interceptable.md`](why-databric-syncflood-not-interceptable.md)
- **Capture tooling** → [`powertrip-capture-readme.md`](powertrip-capture-readme.md), `powertrip-capture.sh`, `run-powertrip-capture.sh`

### Serving & models
- **Working model recipe** → [`RESUME-NOTE.md`](RESUME-NOTE.md), `recipe/serve-qwen38-flash-next-nvfp4.sh`
- **RAM+VRAM offload research** → [`README-ramoffload-research.md`](README-ramoffload-research.md)
- **Model catalog & fit** → [`recipe/MODEL-CATALOG.md`](recipe/MODEL-CATALOG.md)

### Progress
- [`PROJECT-TODOS.md`](PROJECT-TODOS.md) — open/closed tasks.

---

## Status snapshot (as of 2026-09-06)

- **Serving:** `Qwen3.8-Flash-Next-NVFP4` at native **262k** context, CUDA graphs ON, **~39–55 tok/s**
  on 2× RTX Pro 5000 (TP2), PLE table offloaded to host RAM.
- **Software blockers resolved:** NCCL cross-NUMA/no-NVLink hang, FP8-PLE selector bug (vLLM #54765),
  CUSTOM all-reduce CUDA error, and `pidfd` permission — all fixed and baked into the recipe script.
- **Outstanding hardware issue:** the channel-G/MM4 DIMM fault (see the power-trip docs) — mitigate with
  the burst-reduction flags + GPU cap, fix long-term via DIMM replacement/downclock/Gen3.

---

## Notes on the hardware caveat & safety

This machine has served frontier-class 125B-parameter models on a single socket with a known
marginal DIMM. The docs capture everything (evidence, mitigation, and the educational "why"). The
recurring reset is protective (see the "why not interceptable" doc) — the real path is to **fix or
replace the failing DIMM**, which is a legitimate RMA case (see `recipe/diag-dimm-fault.sh`).

**Serve is intentionally manual-only** (see `RESUME-NOTE.md`) to avoid an auto-restart loop from the
trip-prone load. The debug/hardware logging (`powertrip-capture` + `edac-ce-watch`) is also **manual** —
armed only by the `--start` script for a model-load session, not auto-starting on boot.
