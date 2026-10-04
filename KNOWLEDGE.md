# KNOWLEDGE — decision-shaping facts

**Read this before designing a recipe.** It is the accumulated set of facts that change a decision:
what this hardware can and can't do, what each runtime image actually supports, what a whole *model
family* implies before you've run it once, and the patterns general enough to apply to a model
nobody here has touched.

It exists to cut iterations. Every entry is here because not knowing it cost someone a run.

> **Last validated: 2026-10-01** (KT tutorial provenance check: the `1teenarp/ktransformers` fork's
> GLM-5.3-Flash tutorial is content-identical to kvcache-ai main — cite upstream; plus the
> layerwise-prefill/`kt-num-gpu-experts` interaction noted in §2. Prior: **2026-09-30** (live CPU ISA
> probe: EPYC 7663 is **AVX2-only — no AVX-512, no AMX**; KTransformers runtime entry added to §2).
> Prior: **2026-09-24** (live P2P/topology probe: GPU root
> complexes, IOMMU groups, AMD-Vi mode — see the two 2026-09-24 §1 entries; then a **direct P2P
> data-integrity probe** the same day, which corrected them — peer access is advertised and silently
> discards data, and the PCIe root ports are Gen3-capped. Raw traces: `evidence/p2p-probe-2026-09-24/`).
> Full re-check **2026-09-16**:
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
Measured on the fallback path (`NCCL_P2P_DISABLE=1`, 2-rank all-reduce): **6.74 GB/s** busbw at
1 MB, **7.98** at 8 MB, **8.06** at 64 MB, **8.00** at 256 MB `[measured 2026-09-24]` — flat from
8 MB up, i.e. the floor is the link, not the message size.
**The spin also has a CPU-side signature, and aggregate CPU usage hides it.** `[measured 2026-09-25,
R-021]` During steady serving at 24.3 tok/s the host read **96.84 % idle** across 112 logical cores —
while **exactly 2 cores sat pegged at 100 %**, and both were the TP workers' *main* threads
(`VLLM::Worker_TP0/1`, TID==PID). That is `TP`-many fully-pegged cores, one per worker, each blocked
on a collective rather than computing. torch was already at 56 threads / 112 interop with **no
`OMP_NUM_THREADS` set**, so there was no CPU parallelism left to add.
**Use:** when a TP run looks slow, `mpstat -P ALL` before concluding "CPU is fine" — 97 % idle and
"two cores at 100 %" are the same measurement, and only the second one tells you the workers are
spin-waiting. Never answer "can we give it more CPU?" from the aggregate figure: on a comm-bound run
the answer is no, and the per-core view is what proves it.
**Use:** for a model that fits on one card, TP1 can beat TP2. Before blaming a kernel for slow
decode, check power draw and memory throughput; low-and-low means you're comm-bound.

**Host→GPU is ~8.6 GB/s pageable, ~10.2 GB/s pinned.** `[measured]`
**Use:** this is the divisor in every offload estimate (§4, bandwidth law).

**Both GPU root ports are capped at PCIe Gen3 — on Gen5 cards and a Gen4-capable CPU.**
`[measured 2026-09-24]` `00:01.1` and `c0:03.1` each advertise `max_link_speed` **8.0 GT/s** while
the GPUs advertise 32.0 GT/s; `nvidia-smi` reports `pcie.link.gen.max=3`. EPYC Milan does Gen4, so
this is a platform/BIOS setting rather than silicon. Resizable BAR is fully enabled (BAR1 = **64 GB**
on both cards), so the large-BAR prerequisite is already satisfied and is not what limits anything.
**Use:** this throttles the host-bounced collective path *and* H2D offload — the two things every
TP2 decode round depends on — so it is worth a BIOS trip on its own merits, independently of whether
P2P is ever revived. It is also the cheaper of the two, since it pays off even if P2P stays dead.
→ `PROJECT-TODOS.md` A4.6

**`iommu=pt` is NOT set on this host** — AMD-Vi runs in `Translated` mode (kern.log
`Default domain type: Translated`; no `iommu=` flag in `/proc/cmdline`). `[measured 2026-09-24]`
Originally the leading suspect for the P2P hangs; the 2026-09-24 probe (next entry) demoted it to a
secondary factor — the topology, not the IOMMU, is why P2P is dead.
**Use:** still worth proposing as a host change (it removes IOMMU traversal overhead from the
host-bounced collectives every TP2 round uses), but set expectations: an optimisation, **not** a P2P
revival. Don't apply mid-investigation; if tried, re-test with `nccl-tests` afterwards.
**⚠️ CORRECTION 2026-09-24 (data-integrity probe):** the demotion above was reasoning from
topology, not a test — `iommu=pt` was **never actually tried** here. `Translated` mode remains a live
candidate for the dropped peer writes documented two entries down, so treat it as **untested lever
(a)**, not as ruled out.

**GPU↔GPU P2P is structurally absent on this box — not just IOMMU-blocked.** `[measured 2026-09-24]`
Probes: `nvidia-smi topo -m` → GPU0 (BDF 0000:01, root complex 0000:00, NUMA node 3) ↔ GPU1 (BDF
0000:c5, root complex 0000:c0, NUMA node 0) = **SYS**; each GPU hangs off its own PCIe host bridge,
each in a separate IOMMU group (4 AMD-Vi IVHDs); no NVLink, no PCIe switch, no shared root. Any
cross-GPU DMA must traverse the EPYC data fabric (NPS4, cross-node) **via DRAM** — there is no direct
path for P2P to use even in `iommu=pt` mode.
**Use:** don't plan any recipe on P2P coming back or NCCL P2P channels being viable; the comm floor
stays. Realistic levers remain "TP1 if the model fits one card, else accept the floor and buy back
speed with spec decode / concurrency" (or a host change that adds a real interconnect, which none
exists on this board).

  **⚠️ CORRECTION 2026-09-24 (direct P2P data-integrity probe): the conclusion holds, the stated
  mechanism was never measured — and the real one is worse.** "No direct path exists" was inferred
  from topology. What is now measured is that **the P2P path is advertised, accepted, and silently
  discards the data.** `cudaDeviceCanAccessPeer(0,1)` returns **True** both directions and
  `nvidia-smi topo -p2p r/w/p` reports **OK**; every peer write then lands nowhere, **with no error
  returned at any layer**: `cudaMemcpyPeer` (copy engine) → 1048576/1048576 mismatches, all zeros;
  SM-initiated peer store → all zeros; SM peer `atomicAdd` → counter 0, expected 16384
  (`topo -p2p a` = NS); torch cross-device `.copy_()` → all zeros both directions. Same-GPU copies,
  H2D/D2H on both cards and host-staged cross-GPU all return CORRECT in the same harness, so the
  probe is sound. NCCL hangs on **all four** peer transports (`P2P/CUMEM`, `P2P/IPC` via
  `NCCL_CUMEM_ENABLE=0`, `NCCL_PROTO=Simple`, `NCCL_PROTO=LL`) because the flag it spins on is one
  of those discarded writes.
  **Use:** treat `NCCL_P2P_DISABLE=1` as a **correctness** guard, not a throughput tunable — never
  "try removing it to see if it's faster." NCCL is the lucky case: it hangs loudly. Anything that
  trusts `canAccessPeer` and writes peer memory directly gets **zeros and no error**. vLLM's
  `gpu_p2p_access_check` does a real data transfer and correctly returns False both ways, which is
  why `--disable-custom-all-reduce` has held — that guard is load-bearing, not belt-and-braces.
  → `evidence/p2p-probe-2026-09-24/`

**"P2P is structurally impossible here" is not established — two levers are untried.** `[inferred
2026-09-24]` The probe proves peer access is unusable *today*; it does not reach a root cause, and
reading the ACS / AtomicOp control bits needs root (a non-root process gets only the first 64 bytes
of PCI config space). Untested, in cost order: **(a) `iommu=pt`** — AMD-Vi is in `Translated` mode,
and if ACS `RequestRedirect`/`UpstreamForwarding` is set on the root ports, peer TLPs are forced up
to a root complex holding no valid peer mapping and are dropped, which is exactly the observed
signature; one reboot settles it. **(b) Putting both GPUs under one root complex**, which would
remove the fabric crossing entirely — but see the caveat below before costing this one.
**⚠️ Correction 2026-09-25:** root ports are **not** free slots. All five of `0000:c0`'s are populated
by onboard devices (`c1`/`c2` = I226-V NICs, `c3`/`c4` = NVMe, `c5` = GPU1), and `0000:00` exposes
only `00:01.1` (GPU0). Whether *any* two x16 slots on the H12D-8D share a root complex is a
board-layout question that the manual answers for free — settle that before planning a card move.
**Use:** in a plan, cite "confirmed broken, one lever untried and one unverified," not "no
interconnect exists, physics." If either is tried, re-run `evidence/p2p-probe-2026-09-24/` and gate
on **payload**, not bandwidth.
→ `PROJECT-TODOS.md` A4.4 (lever a) · A4.5 (lever b)

**A P2P bandwidth number can look perfect while moving zero bytes.** `[measured 2026-09-24]`
Dropped peer writes still consume wire time, so torch cross-device `.copy_()` benchmarked at
**13.89 GB/s** — a clean, plausible Gen3 x16 figure, ~2× the 7.06 GB/s host-staged path, exactly the
shape of a healthy result — while delivering all zeros. Timing alone endorsed a dead path, and that
number is what made P2P look partly alive here for as long as it did.
**Use:** never qualify an interconnect on throughput. Every transport probe asserts on a known
payload first; a bandwidth figure carrying no correctness assertion is not evidence of transport.

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

**`vllm/vllm-openai:glm53-flash-cu129` is NOT a newer fix — same 2026-09-09 code, CUDA-12.9 base.**
`[measured 2026-09-16, Docker Hub API]` All `glm53*` tags were pushed 2026-09-09 13:31–13:32Z in one
batch; the multi-arch `:glm53-flash` R-014 ran is the same code. The `pe_dim==64` assert lives in
vLLM's own `cache_kernels.cu` and the SM90-only NoPE selector is Python — a CUDA-base rebuild changes
neither (the assert also reproduced on a completely different `:nightly` build).
**Use:** don't pull cu129/cu130 variants expecting a fix.

**SGLang's real DSA-backend flags are `--dsa-prefill-backend` / `--dsa-decode-backend`** with
`choices = ['flashmla_sparse','flashmla_kv','flashmla_auto','fa3','tilelang','aiter','trtllm']`
(`[measured 2026-09-16]` in `lmsysorg/sglang:dev-glm52-nvfp4`; `--dsa-attention-backend` /
`--dsa-moe-backend` do not exist). A `tilelang` DSA path is real and JIT-compiles rather than
requiring prebuilt Hopper kernels — the SGLang hope for NoPE on sm_120 survives — but this image
lacks `Glm5Next` and its NVFP4 MoE loader already failed on GLM-5.2 (R-000b). Existence ≠ sm_120
support ≠ speed; needs a staged test on a free GPU pair.

**The sm_120 NoPE block is solvable without patching vLLM: `Libertai/glm53-flash-vllm-gb10` (Apache-2.0)
ships the fix as a pip-installable `vllm.general_plugins` package.** `[upstream 2026-09-16, repo read]`
Two env-gated entry points (`VLLM_GLM53_CUDA_SPARSE_MLA=1` overrides the `FLASHINFER_MLA_SPARSE_SM120`
enum slot with a hand-written NoPE CUDA kernel, head_size 512 native so the `pe_dim==64` path is never
reached; `VLLM_GLM53_MOE_INPUT_SCALE` for fault 2, below). No vLLM file is edited; the `.so` is built
**inside** the target container (`GLM53_ARCHS=120a`); its `backend.py` explicitly targets the same
`glm53-flash` vendor-fork image we run and feature-detects upstream drift. Kernel is fixed to **32
heads/rank = TP2 on 64 heads** — exactly pensive's config. Measured by them: sm_120 on 4x RTX PRO 6000
TP4 (README claims; `backend.py` enforces 32/rank — verify at port), graphs per README `UNIFORM_BATCH`
(per code `NEVER` — verify), **bf16 KV only per code** (fp8 claim in README unconfirmed).
**Use:** P12-clean path = build a derived image `FROM vllm/vllm-openai:glm53-flash` +
`pip install` their `kernel/`; do not run their image.

**Fault 2 (ModelOpt NVFP4 `w13_input_scale` uninitialised → MoE outputs ×0 → "locklock" garbage)
does NOT fire on `nvidia/GLM-5.3-Flash-NVFP4`.** `[measured 2026-09-16/17]` — 36,297 `input_scale`
tensors present (full 42×288×3 coverage), sampled values nonzero (7.5e-4 … 3.7e-2), and the
`:glm53-flash` base image's `routed_experts.py` contains the ModelOpt-NVFP4 branch that maps the
per-expert `*_proj.input_scale` checkpoint tensors into `w13/w2_input_scale` (init sentinel there
is 1.0, and 1.0 is itself bad per their correction — but ours are loaded). It is a checkpoint
property, not a GPU property; theirs had `"input_activations": null` + zero `input_scale` tensors.
**Use:** leave `VLLM_GLM53_MOE_INPUT_SCALE` unset for the nvidia checkpoint; but since the fault is
silent-garbage, the first serve must include a correctness probe (arithmetic/factual prompts), not
just liveness. → also a standing check for any future third-party NVFP4 quant.

**Build-time facts for kernel-in-image ports on `vllm/vllm-openai` (torch 2.13.0+cu130, nvcc 13.0,
ninja present, no git)** `[measured 2026-09-17]` — (a) `pip install` of a CUDAExtension whose name
is `pkg._C` imports the parent package at metadata time (`find_spec` semantics); if that `__init__`
has a GPU-probing JIT fallback, the build dies unless the source tree is gone or cwd isn't it.
(b) An installed AOT `_C*.so` **loads fine with no driver present** — CPU-only `--check` stages can
assert its presence and even `torch.ops` registration; only real kernel launches need the GPU.
**Use:** in Dockerfiles, `pip install`, delete the source tree, then verify from `cd /`.

**Never `rm -rf` a bind-mount path from inside a container to "clean up"** — it deletes the HOST
directory's contents and fails only at the mount point itself (`Device or resource busy` is not
protection). Cost: the vendor clone's `kernel/` tree, recovered with `git checkout --`.
**Use:** unmount-aware cleanup only; inside probes should treat mounted data as read-only by habit.

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

**KTransformers / kt-kernel is the CPU-experts alternative to vLLM's `--cpu-offload-gb` for MoE
decode.** `[upstream 2026-09-30, kvcache-ai/KTransformers v0.7.0.post4 docs]`
Architecture: routed MoE experts **live in host RAM and execute on the CPU** (AMX / AVX512 / AVX2 /
AMD-BLIS kernels); attention, dense, shared experts, embeds run on GPU via the `sglang-kt` fork
(kvcache-ai's SGLang — *not* vanilla `sglang`; the `ktransformers[sglang]` extra pins the fork).
Install is one command: `pip install "ktransformers[sglang]"` (Py≥3.11; prebuilt wheels carry a
static CUDA runtime and auto-select the CPU variant at import — `kt_kernel.__cpu_variant__`).
GLM-5.3-Flash has native support since 2026-08-26: reads the **official FP8 checkpoint directly,
no conversion**, claims SM89+SM120 GPUs, wants ≥350 GB system RAM, launches via
`sglang.launch_server --kt-method FP8 --kt-cpuinfer <phys-cores> --kt-weight-path <model>`.
**Why it matters on pensive:** the vLLM offload path streams expert weights to the GPU every token
(~8.5 GB/token → ~10 GB/s H2D ceiling → 2.4–2.5 tok/s measured, R-020). KT keeps experts on the CPU,
so the bottleneck becomes host-DRAM bandwidth + CPU GEMM (P10 estimate here: 3–12 tok/s), and a
TP1 single-GPU launch eliminates the TP2 comm floor entirely (no ~104 all-reduces/round).
**Status on pensive (R-024→R-027, 2026-10-01):** (a) the AVX2-only risk did not materialize — the
FP8 CPU-expert path is live **and forward-verified**: `AVX2_FP8_MOE_TP 0..3` pools (one per NUMA
node) for all 42 sparse layers, coherent correct output at decode. The tutorial's "AVX-512" line was
conservative. (b) sm_120 **forward-verified** — startup selects `GPU profile=blackwell_fp8`,
`NSA dispatcher=trtllm … on SM86/SM89/SM120`, graph capture 179.8 s; the model-tutorial's SM89+SM120
claim held (the generic kt-kernel CUDA matrix lists only SM 80–90 — trust the model doc, verify at
Stage 1). (c) **With Layerwise Prefill enabled (the default config for GLM-5.3-Flash), the resident
GPU-expert count is normalized to zero** — `--kt-num-gpu-experts N` only pays off when prefill stays
on the GPU path (prompt ≤ `--kt-gpu-prefill-token-threshold`, default 2048); longer prompts take the
layerwise path and drop GPU experts to 0 regardless of the flag. Decide the prefill threshold
*before* planning expert residency. `[upstream 2026-10-01, tutorial §4]` (d) **Image toolchain —
the real cost of this path (R-024/R-025/R-026):** sglang-kt JIT-compiles **at graph capture, after
the full 305 GB weight load** — a triton C-helper (needs `cc` + `Python.h`) *and* an sgl_kernel CUDA
kernel (needs `nvcc`; compiles for `arch=compute_120a`). Each missing tool was a separate ~30-min
failed load, one JIT stage deeper than the last. Build from a `-devel` CUDA base (or add
`build-essential` + `python3-dev` + nvcc) and gate with `which cc && which nvcc` at build time.
`[measured]`
(e) **`libnuma1` is also a runtime dependency — but lazy, not at startup (R-030/R-031).** The
layerwise-prefill CPU-expert path (first prompt > `--kt-gpu-prefill-token-threshold`, default 2048)
calls `ctypes.CDLL("libnuma.so.1")` for NUMA-aware shared-memory buffers. Without it the server
dies on the first long prompt with the *misleading*
`RuntimeError: KT shared-memory NUMA setup failed on at least one TP rank` — the real
`OSError: libnuma.so.1` sits a few frames up the traceback. Short-prompt decode never touches the
path (R-030 ran ~5 h clean before a user's long prompt crashed it). The launcher's `--check` now
carries the 5-s in-image gate: `python3 -c "import ctypes; ctypes.CDLL('libnuma.so.1')"`.
`[measured]`
Fallbacks if the default path ever breaks: `--kt-method MOE_INT8` + AMD-BLIS source build
(`CPUINFER_ENABLE_BLIS=ON`) + `convert_cpu_weights.py --quant-method moe_int8` (P12-clean, vendor
weights); `--kt-method LLAMAFILE` + GGUF (AVX2, prebuilt wheel — P12-flagged third-party quant, user
OK required).
Provenance note: the `1teenarp/ktransformers` fork's GLM-5.3-Flash tutorial is
content-identical to kvcache-ai main (verified 2026-10-01) — cite kvcache-ai upstream.
**Measured on pensive (R-027, 2026-10-01):** steady-state **5.81 tok/s** (bs=1, ctx 8192,
GPU0/TP1, FP8-AVX2 experts, CUDA-graph bs=1; correctness probes pass) — **2.4× the vLLM NVFP4
baseline (2.4–2.5, R-020)**, low end of the 3–12 P10 band: decode is CPU-DRAM/AVX2-GEMM bound
(GPU0 50 GiB / ~92 W, lightly used; RAM peak ~448 GiB). Live log caveat: FP8 KV scale defaults to
1.0 ("may lead to less accurate results") — probes passed, watch downstream quality.
**Use:** the single-GPU FP8 config is the default GLM-5.3-Flash path here (P12-clean: vendor
weights, our own image from official wheels). Stage-3 levers in order: concurrency
(`--max-running-requests`), short-prompt GPU-expert residency (lever (c)), context.
→ `recipe/GLM-53-FLASH-KT-RECIPE.md`

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
  **⚠️ CORRECTION 2026-09-24 (R-019/R-020): this claim is now scoped to *stock* vLLM images.** Our
  own derived image `pensive/glm53-flash:nope-sm120-617d0cc` (P12-clean port of the Apache-2.0
  `glm53_sparse_mla` NoPE kernel) **serves the nvidia NVFP4 checkpoint end-to-end on sm_120** —
  real weights, coherent output, ~2.4–2.5 tok/s steady (eager, no MTP, comm-bound; bf16 KV,
  block 256, TP2, 32 GB/wkr offload). The ⛔ still holds for stock images and for the zai-org FP8
  checkpoint on any vLLM we have.
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
- **Alternate serving path: KTransformers (kt-kernel), CPU-resident experts.** `[upstream 2026-09-30]`
  Native GLM-5.3-Flash support since 2026-08-26: reads the official zai-org FP8 weights directly
  (no conversion), single-GPU config validated upstream (tp1, `--mem-fraction-static 0.65`, chunked
  prefill 2048, decode CUDA graphs `--cuda-graph-bs 1 2 4`, 501025-ctx). Structurally dodges the TP2
  all-reduce floor that capped R-020 at 2.4–2.5 tok/s: experts execute on CPU, GPU does
  attention/dense/shared, TP1 → no TP collectives. **Measured on pensive 2026-10-01 (R-027):
  5.81 tok/s steady (bs=1, ctx 8192) — 2.4× the vLLM baseline, low end of the 3–12 P10 band,
  CPU-DRAM/AVX2-GEMM bound.** Needs ≥350 GB RAM (have 751 GiB). → `recipe/GLM-53-FLASH-KT-RECIPE.md`

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

**P-K · For offloaded MoE decode, "CPU offload" has two regimes with different bottlenecks.**
Regime A (vLLM `--cpu-offload-gb`): weights live in RAM and are **streamed to the GPU every token**
→ PCIe/H2D-bandwidth bound (P-B law: `tok/s ≈ H2D GB/s ÷ active weight bytes per token`).
Regime B (KTransformers/kt-kernel): routed experts **execute on the CPU**, so the cost moves to
host-DRAM read bandwidth + CPU int8/fp8 GEMM; the GPU only runs attention/dense/shared. On a box
with large RAM and a weak GPU↔GPU / GPU↔host interconnect (this box: no NVLink, P2P silently
discards data, Gen3-capped root ports), regime B is often structurally faster than regime A —
and TP1 removes the all-reduce floor entirely.
**Use:** at intake for any MoE that doesn't fit in VRAM, compute *both* ceilings (P-B H2D ceiling
vs DRAM-bandwidth ceiling) and prefer the regime with the higher ceiling when the runtime supports
it. Measured on pensive 2026-10-01: GLM-5.3-Flash FP8 — regime A (vLLM TP2+offload) 2.4–2.5 tok/s
(R-020) vs regime B (KT CPU-experts) **5.81 tok/s** (R-027): ~2.4× faster, landing at the low end
of the 3–12 P10 band because AVX2-only GEMM runs ~50 % of the DRAM bandwidth ceiling.
*Generalized from the KTransformers pivot (2026-09-30); both regimes now measured (R-020, R-027).*

**P-L · For bandwidth-bound CPU-expert MoE decode, concurrency is not a throughput lever.**
When every decode step streams the expert weights from one shared DRAM stream (~8.5 GB/token here),
adding concurrent requests splits that stream: per-request rate ≈ 1/N, and aggregate grows only as
fast as batching amortizes launches + partial expert overlap. **Measured on pensive 2026-10-01
(R-028):** GLM-5.3-Flash KT, N=1/2/4 — per-request 5.81→3.23→1.76 tok/s, aggregate decode-phase
5.83→6.48→7.05 tok/s (**+21 % at 4× concurrency**), TTFT 6.6→13.0→24.7 s. A *linear* aggregate
gain would have meant compute-bound; the sublinear +21 % is the shared-bandwidth fingerprint.
**Use:** at Stage 3 of any regime-B serve, run the 1→2→4 sweep **once** (it is cheap: one
`--max-running-requests` change + one sweep) and then stop tuning concurrency — invest the session
in structural levers instead (GPU-expert residency for short prompts, context length, DRAM
speed/width, a faster ISA). Leave the server at the highest `--max-running-requests` you swept:
bs=1 users are unaffected (their graph is unchanged) and multi-user traffic batches instead of
queues.
*Generalized from the GLM-5.3-Flash KT concurrency sweep (R-028).*

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
| Client `400 Bad Request` on `/v1/chat/completions` (esp. after a context-length change) | Server-side request-contract validation, not a server bug: sglang rejects when `prompt_tokens + max_tokens > --context-length` (`max_completion_tokens is too large … at most N` / `Requested token count exceeds … maximum context length`), when `top_p` is outside (0,1], when `frequency/presence_penalty` is outside [-2,2], **or when `reasoning_effort` is not one of `low`/`medium`/`high`** (OpenWebUI's "Max" setting sends `max` → `Unsupported reasoning_effort='max' … Expected one of: low, medium, high`; set it to High or off — R-031-era diagnosis, all other params — model-name variants, temperature, `n`, `tools`, `stream_options`, content-parts — probed accepted). First instance: a client sized for GLM's native 1 M context hit the 8192 smoke cap (R-029, fixed by the 501025 serve). **Diagnose by reading `/v1/models` `max_model_len` and matching the client's context/max-tokens/reasoning-effort settings to it.** |
| `RuntimeError: KT shared-memory NUMA setup failed on at least one TP rank` (server exit 137 on a long prompt, host fine) | Not a NUMA-topology or capacity problem — it wraps a bare `OSError: libnuma.so.1: cannot open shared object file`: the layerwise-prefill CPU-expert path lazily dlopen's libnuma, and the image was missing `libnuma1`. Read the root `OSError` a few frames up before classifying as hardware/NUMA. Fix: `+libnuma1` in the image (R-030→R-031); gate: in-image `ctypes.CDLL("libnuma.so.1")` probe in `--check`. |
| `No MLA prefill backend supports this model; sparse MLA will use the top-k MQA path only` | Informational on GLM-5.3; not the failure. |
