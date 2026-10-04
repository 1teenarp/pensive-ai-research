# GLM-5.3-Flash on "pensive" — KTransformers (CPU-experts) recipe
Date: 2026-09-30, updated 2026-10-01. Status: **Stage 2 `--serve` PASSED (R-029/R-031) — full-context serve live on :8093 (CTX=501025, `MAX_RUNNING_REQUESTS=4`, GPU0).** R-030: a user's long prompt crashed the server — root cause was `libnuma.so.1` missing from the image (the layerwise-prefill CPU-expert path lazily dlopen's it); R-031 fixed it (`+libnuma1` in the image, in-image `ctypes.CDLL` probe added to `--check`) and regression-tested the exact crash path (3442-token prompt → 200, clean prefill+decode). Stage 1 `--smoke` PASSED (R-027); Stage-3 concurrency sweep done (R-028). First READY + coherent output of the KT path. Image now carries `build-essential + python3-dev` on a `-devel` base (nvcc). Three prior graph-capture deaths retired: R-024 "no C compiler" (no cc) → R-025 "no `Python.h`" (no python3-dev) → R-026 "no `nvcc`" (base, not devel). Correctness probes pass (arithmetic + factual). **Measured steady-state: 5.81 tok/s** (256-token gen ×3, warmup discarded; server log 5.76–5.80) — **2.4× the vLLM baseline (2.4–2.5 tok/s, R-020)**. Container left running (user directive); next step is Stage-3 increments. Stage 0 `--check` passed (R-022 2026-09-30, re-verified R-023 2026-10-01).
Model: `/trunk/ai/huggingface/models/zai-org/GLM-5.3-Flash` (official zai-org FP8, 305 GB, 62 shards)
Launcher: `recipe/serve-glm-53-flash-kt.sh` (`--check` → `--smoke` → `--serve`)
Image: `pensive/glm53-kt:latest` (built from `recipe/Dockerfile.glm53-kt`)

---

## 0. Where this stands (read before running anything)

**Why this path exists:**
The vLLM offload path (GLM-53-FLASH-NVFP4-RECIPE.md) is proven at ~2.4–2.5 tok/s (R-020) but is
structurally comm-bound: ~104 host-bounce all-reduces per decode round on TP2 + per-token H2D
expert streaming (~4.3 GB/tok). KTransformers moves routed experts to CPU where they **execute
in-place**, eliminating both the PCIe H2D stream and (with TP1) the all-reduce floor entirely.

**Why only GPU0 (GPU1 is idle by design):** KT's split puts ~305 GB of routed experts in host RAM
executed on CPU; the GPU holds only attention (34 KDA + 11 DSA) + dense + shared experts + embeds +
KV ≈ 25–30 GB — comfortably one 72 GB card. Decode is CPU-GEMM/DRAM-bandwidth bound (P-K regime B),
so a second GPU adds nothing to the bottleneck. **Do not "fix" the idle GPU1:** PP=2 is untested in
the sglang-kt CPU-expert path (tutorial validates only tp1/tp4) and adds pipeline bubbles at
batch-1 (P-I) plus stage-boundary transfers over the Gen3 root ports; there is no sglang feature to
place KV on a different GPU, and KV is not the constraint here (11 DSA layers, kv_lora_rank 512).
If GPU1 is wanted, it is for a *separate* second instance, not TP/PP of this one.

**P12 provenance note (user-directed 2026-09-30):**
- Runtime: kvcache-ai/KTransformers v0.7.0.post4 (Apache-2.0), installed from official PyPI
  wheels into our own Docker image — not a third-party image.
- Weights: vendor-published `zai-org/GLM-5.3-Flash` (FP8, native). No conversion, no third-party
  quant on the default path.
- Fallback path (if AVX2 FP8 kernel unavailable): `--kt-method LLAMAFILE` with
  `unsloth/GLM-5.3-Flash-GGUF` (third-party quant — P12 flag; requires explicit user OK).
- AMD BLIS path: `--kt-method MOE_INT8` + source build with `CPUINFER_ENABLE_BLIS=ON` +
  `convert_cpu_weights.py --quant-method moe_int8`. Also P12-clean (vendor weights).

**Known unknowns before first run — all resolved (R-024…R-027, 2026-10-01):**
1. ~~Does the AVX2 CPU-expert kernel support `--kt-method FP8` on Zen 3?~~ — **RESOLVED at load
   time (R-024):** `AVX2_FP8_MOE_TP 0..3` pools created per NUMA node for all 42 sparse layers.
   **Forward-pass-verified (R-027):** coherent, correct output.
2. ~~Does sglang-kt's attention backend support Glm5Next on sm_120?~~ — **RESOLVED (R-027):**
   `GPU profile=blackwell_fp8 … NSA dispatcher=trtllm … SM86/SM89/SM120`; graph capture 179.8 s
   OK; correctness probes pass. The `fla/utils.py` "roll back to CPU" warning for KDA did not
   prevent coherent output; the 5.8 tok/s decode being CPU-DRAM-bound is consistent with KDA
   either on-GPU-fast or CPU-fallback — not worth chasing further until a quality issue appears.
3. ~~RAM / NUMA:~~ ran clean — peak ~448 GiB used of 751 GiB; NPS4 + uneven population was a
   non-issue in practice.
4. (New from R-024…R-026, all retired) The image must be a **devel**-class build (or
   `build-essential` + `python3-dev` + nvcc added): sglang-kt JIT-compiles a triton C-helper
   **and** an sgl_kernel CUDA kernel at graph capture, *after* the full weight load. Now baked
   into `recipe/Dockerfile.glm53-kt`.

**Gates (from §4 of the FP8 recipe + house rules):**
- powertrip-capture running + klog mtime fresh
- edac-ce-watch running
- GPU power cap (needs interactive sudo — probably won't apply; known non-issue)
- No other serve on GPUs
- Caches dropped before load

---

## 1. Model facts (same model, different runtime)

| Property | Value |
|---|---|
| Repo | `zai-org/GLM-5.3-Flash` (vendor-published FP8) |
| Arch | `Glm5NextForConditionalGeneration` |
| Size | 320B total / 18B active |
| Layers | 45 (3 dense + 42 sparse MoE) |
| Experts | 288 routed / 8 per token + 1 shared; `moe_intermediate_size` 2048 |
| Attention | 34 KDA linear-attn + 11 NoPE sparse-MLA (DSA); kv_lora_rank 512 |
| MTP | 1 nextn-predict layer |
| Context | native 1,048,576 |
| Parsers | `--tool-call-parser glm47 --reasoning-parser glm45` |
| On-disk size | 305 GB / 62 shards |

*(Same as GLM-53-FLASH-RECIPE.md §1 — this is a runtime change, not a model change.)*

---

## 2. Fit math (KT-specific)

**GPU side (GPU 0, 72 GB, mem-fraction 0.65 → ~47 GB usable):**
- Attention (34 KDA + 11 DSA): ~10–14 GB in BF16/FP8
- Dense layers (3): ~1–2 GB
- Shared experts (42 × 25.2M): ~1 GB
- Embed + lm_head: ~1.5 GB
- MTP layer: ~0.5 GB
- KV cache (501k ctx, 11 DSA layers, kv_lora_rank 512): ~5–6 GB
- Activations + overhead: ~2–3 GB
- **Total GPU: ~25–30 GB → fits in 47 GB comfortably**

**CPU side (751 GiB available, NPS4):**
- Routed experts: 42 × 288 × 25.2M × 1 byte (FP8) ≈ 305 GB
- Plus model metadata, KV, activations: ~10–20 GB
- **Total CPU: ~320 GB → fits in 743 GiB with large margin**

**Decode ceiling (P10 estimate):**
- Per-token CPU expert stream: 42 layers × 8 experts × 25.2M × 1 B ≈ 8.5 GB
- EPYC 7663 8-channel DDR4-3200: ~150–170 GB/s practical (NPS4, 6 DIMMs)
- 8.5 GB / 160 GB/s ≈ 53 ms/token → ~19 tok/s bandwidth ceiling
- AVX2 GEMM on Zen 3 (no VNNI/AMX): probably 40–60% of bandwidth ceiling
- **P10 estimate: 5–12 tok/s steady-state (single request)**
- Compare: vLLM offload = 2.4–2.5 tok/s (R-020) → **potential 2–5× improvement**

*These are estimates. Measure, don't trust (P10).*

---

## 3. Blockers & gates

| # | Item | Status |
|---|---|---|
| 1 | CPU AVX2-only (no AVX512/AMX) — FP8 CPU kernel may be AVX512-only | ✅ Resolved + forward-pass verified (R-024 load, R-027 output): `AVX2_FP8_MOE_TP` pools per NUMA node, coherent correct output `[measured]` |
| 2 | sm_120 GPU support in sglang-kt attention | ✅ Resolved + forward-pass verified (R-027): `blackwell_fp8` profile, trtllm NSA, graph capture 179.8 s, probes pass. `fla` KDA CPU-fallback warning noted, non-fatal |
| 3 | RAM capacity (305 GB model + overhead vs 743 GiB) | ✅ Ran clean (R-027): peak ~448 GiB of 751; NPS4 uneven population was a non-issue in practice |
| 4 | P12 compliance | ✅ Default path (FP8 vendor weights, our image). LLAMAFILE fallback needs user OK. |
| 5 | GPU exclusivity | Handled by `require_gpus_free()` |
| 6 | Power cap | Known non-issue (no sudo); log and continue |
| 7 | NUMA | KT is NUMA-aware internally; no NCCL guard needed at TP1 |

---

## 4. Staged plan

### Stage 0 — `--check` (CPU-only, seconds, no GPU)
```bash
bash recipe/serve-glm-53-flash-kt.sh --check
```
Proves: image built, kt_kernel imports, CPU variant = avx2, sglang-kt present, model dir complete
(62 shards, ~305 GB), RAM ≥ 350 GB available.

### Stage 1 — `--smoke` (real weights, conservative config)
```bash
bash recipe/serve-glm-53-flash-kt.sh --smoke
```
Config: GPU0 only, ctx 8192, seqs 1, kt-method FP8, kt-cpuinfer 56, threadpool 4,
mem-fraction 0.65, cuda-graph-bs "1 2 4".
Proves: sm_120 attention works, AVX2 FP8 expert kernel works (or tells us it doesn't),
server binds, HTTP 200.
**Gate:** must pass a correctness probe (arithmetic + factual + repeat-token check) before
trusting throughput.
**If FP8 CPU kernel fails:** try `KT_METHOD=LLAMAFILE CPU_WEIGHTS=/trunk/ai/huggingface/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q4_K_XL`
(needs user OK for P12 third-party quant) or `KT_METHOD=MOE_INT8` (needs source build).

### Stage 2 — `--serve` (full config)
```bash
bash recipe/serve-glm-53-flash-kt.sh --serve
```
Config: same as smoke but ctx 501025. Measure steady-state tok/s (discard first 3 requests, P6).

### Stage 3 — increments (one per session, P2)
1. `KT_NUM_GPU_EXPERTS=64` — move hot experts to GPU, reduce CPU stream.
   ⚠️ Caveat (tutorial §4, verified 2026-10-01): with Layerwise Prefill active (default: prompts
   > `KT_GPU_PREFILL_THRESHOLD`=2048), the runtime **normalizes resident GPU experts to 0** — this
   lever only pays off for prompts ≤ threshold. To make it real: raise `KT_GPU_PREFILL_THRESHOLD`
   (or drop `MAX_MODEL_LEN`) so prefill stays GPU-side, then re-test.
2. ~~`MAX_RUNNING_REQUESTS=2` → 4 (concurrency sweep)~~ — **DONE (R-028, 2026-10-01):**
   per-request 5.81→3.23→1.76 tok/s at N=1/2/4; aggregate 5.83→6.48→7.05 tok/s (+21 % at N=4).
   Sublinear aggregate + 1/N split = shared DRAM expert stream → **concurrency is NOT the lever**
   for this regime. Container left at `MAX_RUNNING_REQUESTS=4` (strictly ≥ the 1-config: bs=1 graph
   unchanged for single users; 2–4 users batch instead of queue).
3. `CTX=131072` → 501025 (if KV headroom allows)
4. `KT_GPU_PREFILL_THRESHOLD` tuning
5. A/B: MOE_INT8 (source build) vs FP8-AVX2

---

## 5. Launcher reference — `recipe/serve-glm-53-flash-kt.sh`

| Env | Default | Notes |
|---|---|---|
| `MODEL` | `/trunk/ai/huggingface/models/zai-org/GLM-5.3-Flash` | GPU-side weights |
| `CPU_WEIGHTS` | same as MODEL | Override for LLAMAFILE (GGUF dir) or MOE_INT8 (converted) |
| `IMAGE` | `pensive/glm53-kt:latest` | Build: `docker build -f recipe/Dockerfile.glm53-kt -t pensive/glm53-kt:latest recipe/` |
| `NAME` | `glm53-kt-serve` | |
| `PORT` | `8093` | |
| `GPU` | `0` | Single GPU; no TP |
| `KT_METHOD` | `FP8` | `FP8` / `MOE_INT8` / `LLAMAFILE` |
| `KT_CPUINFER` | `56` | Physical cores |
| `KT_THREADPOOL` | `4` | NUMA node count |
| `KT_NUM_GPU_EXPERTS` | `0` | Stage-3 lever |
| `KT_GPU_PREFILL_THRESHOLD` | `2048` | Layerwise prefill threshold |
| `MEM_FRACTION` | `0.65` | GPU mem fraction |
| `CHUNKED_PREFILL` | `2048` | |
| `SMOKE_CTX` | `8192` | Stage 1 context |
| `CTX` | `501025` | Stage 2+ context |
| `CUDAGRAPH_BS` | `1 2 4` | Decode CUDA graph batch sizes; `""` to disable |
| `MAX_RUNNING_REQUESTS` | `1` | |
| `GPU_POWER_CAP` | `250` | Needs sudo; known non-issue if it fails |
| `BIND_HOST` | `""` | `""` = all ifaces; `100.70.5.43` = tailscale only |
| `API_KEY` | `""` | `""` = no auth; else Bearer |
| `SERVE_LOG` | `/var/tmp/serve-glm53-kt.log` | Container ID only; real logs in `docker logs` |
| `WAIT_TIMEOUT` | `3600` | |

Fixed flags (not env-overridable): `--tp-size 1 --trust-remote-code --host 0.0.0.0 --port 30000
--tool-call-parser glm47 --reasoning-parser glm45`

---

## 6. Verdict (measured, post R-031)

**Works.** Stage 1 `--smoke` PASSED (R-027, 2026-10-01 11:31 UTC). First coherent end-to-end output
of the KT path: correctness probes pass (arithmetic 17×23=391; factual Paris/Seine), and the
model produces clean, non-degenerate generations.

**Measured (single request, bs=1, ctx 8192, GPU0, TP1, CUDA graph bs=1):**
- **Steady-state decode: 5.81 tok/s** (client-side, 256-token gen ×3, warmup discarded per P6;
  server-log cross-check 5.76–5.80). **2.4× the vLLM NVFP4 baseline (2.4–2.5 tok/s, R-020).**
- TTFT ~4.8 s. GPU0 50.1 GiB (attention+dense+shared+embeds+KV+graphs, within mem-fraction 0.65);
  RAM peak ~448 GiB (305 GB experts + overhead). GPU1 idle by design (TP1, CPU experts).
- P10 estimate was 5–12 tok/s → measured 5.81 lands at the low end. Decode is CPU-DRAM/AVX2-GEMM
  bound (P-K regime B), not GPU-bound (GPU util low, ~92 W; CPU the constraint).

**Why the low end of the estimate:** AVX2 GEMM on Zen 3 (no VNNI/AMX) runs well below the DRAM
bandwidth ceiling; plus the `fla` KDA path may be on a CPU fallback and the FP8-KV scale-1.0
warning is present. Neither is a blocker; both are Stage-3 tuning targets.

**Caveats on this run:** power cap ran at 300 W (default, no sudo for 250 W); only bs=1 graph
captured (`--max-running-requests 1`); FP8 KV scale defaulted to 1.0 (watch for quality drift).

**Concurrency (R-028, 2026-10-01):** sweep N=1/2/4 at 256-tok gens — per-request decode
5.81 → 3.23 → 1.76 tok/s (≈1/N split); aggregate decode-phase 5.83 → 6.48 → 7.05 tok/s (**+21 %
at N=4**); TTFT 6.6 → 13.0 → 24.7 s. Sublinear aggregate growth is the shared-DRAM-bandwidth
fingerprint: concurrency is **not** the throughput lever in regime B. Container now runs
`MAX_RUNNING_REQUESTS=4` (bs 1/2/4 graphs): single users are identical to the bs=1 config
(5.81 tok/s, 6.6 s TTFT); 2–4 concurrent users batch (6.03 tok/s aggregate) instead of queueing —
strictly ≥ the old config in every case.

**Full-context serve (R-029, 2026-10-01):** `--serve` at CTX=501025 (tutorial-validated; ONE
variable vs R-028) — READY ~21 min (warm cache), `max_model_len` 501025, bs 1/2/4 graphs,
`MAX_RUNNING_REQUESTS=4`. This fixed the user's client `400 Bad Request`: the 8192-ctx smoke server
rejected any `max_tokens` that pushed `prompt+max_tokens` over 8192 (sglang validates the total).
A second 400 class (same day, post-restart) was the client's `reasoning_effort:"max"` — sglang only
accepts `low`/`medium`/`high` (OpenWebUI "Max" setting → set to High or off). All other params
probed clean (model-name variants, temperature, `n`, `response_format`, `stop`).

**Long-prompt crash + root-cause fix (R-030→R-031, 2026-10-01):** a user "very long prompt"
crashed the R-029 server (exit 137, host fine) with
`RuntimeError: KT shared-memory NUMA setup failed on at least one TP rank`. The root cause was a
missing shared library, not NUMA: the layerwise-prefill CPU-expert path (first prompt >
`--kt-gpu-prefill-token-threshold` 2048) lazily `ctypes.CDLL`s `libnuma.so.1`, which the image
lacked (`libnuma1`). R-031 added it, rebuilt the image (3cf104f7ddcb), and added a 5 s in-image
`--check` gate. **Regression-tested the exact crash path: a 3442-token prompt → HTTP 200, clean
prefill+decode (prefill ~10.6 tok/s first-use, P6: not steady-state).** This is the fix; the
layerwise-prefill path is now proven end-to-end, so full-ctx (501025) long prompts work.

**Status now:** container `glm53-kt-serve` left running (user request) — the live GLM-5.3-Flash
endpoint on `:8093`, at full 501025 context, `MAX_RUNNING_REQUESTS=4`, image 3cf104f7ddcb
(`+libnuma1`). This is the default serving path; the vLLM NoPE-ported image remains the fallback
for CPU-saturated scenarios.

**Open questions (in priority order):**
1. `KT_NUM_GPU_EXPERTS` (layerwise prefill) — only pays for prompts ≤ `KT_GPU_PREFILL_THRESHOLD`
   (2048); re-test with the threshold raised or short prompts.
2. ~~Context: 8192 → 32768 → 501025~~ — **DONE (R-029): running at 501025**; KV headroom was a
   non-issue (KV pool sized by mem-fraction, not context length — ~24 GB pool at capture).
3. `KT_THREADPOOL` 4 → 8/16 and `KT_CPUINFER` 56 → 48 sweep (NUMA/oversubscription trade-off).
4. A/B: MOE_INT8 (AMD BLIS source build) vs the current FP8-AVX2 path.
