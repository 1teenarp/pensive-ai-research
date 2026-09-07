# RESUME NOTE — Qwen3.8-Flash-Next-NVFP4 serving + power-trip capture
Last updated: 2026-09-06 (manual-only serve decision)

## One-line status
✅ **The model now SERVES successfully on this box (TP2, PLE CPU offload), and this run did NOT
power-trip.** The software path is fully validated end-to-end (weight load → PLE registration → KV init
→ CUDA-graph capture → server startup → real inference with correct answers).

## IMPORTANT — serve is MANUAL-ONLY (do NOT auto-start)
- **`qwen38-flash-serve` is intentionally `restart: no`** and must NOT be set to auto-restart.
  Reason: the recurring hardware fault (channel G / MM4 data-fabric sync-flood, see
  `power-trip-instances.md`) fires during the heavy ~80 GB weight-load burst. If the serve auto-started
  on boot it could trip the box and loop reboots.
- `powertrip-capture` (the telemetry logger) and `edac-ce-watch` are **also manual** (`restart: no`) —
  they are armed by `recipe/serve-qwen38-flash-next-nvfp4.sh --start` for a model-load/debug session,
  and do NOT auto-start on boot. This keeps all specialized metric containers on-demand.
- To run the server after a boot/reboot, launch **manually** with the command in "The exact serve
  command" below. The patched image and patched `ple_layer.py` survive reboots (they're in Docker layer
  cache), so no rebuild is needed unless the image is pruned.

## What worked / how to bring the server back up
The software fixes for the Qwen3.8-Flash-Next-NVFP4 TP2 serve (all validated; reached weight-load →
PLE registration → CUDA-graph capture):
1. **NCCL P2P hang** — GPUs on different NUMA nodes, no NVLink → `NCCL_P2P_DISABLE=1`.
2. **FP8 PLE selector** (vLLM issue #54765) — patch `_get_ple_embedding_quant_method` in
   `ple_layer.py` to select the FP8 method under ModelOpt NVFP4; opt-in via `VLLM_QWEN38_PLE_FP8_SCALE=1`.
3. **CUSTOM all-reduce CUDA error** — add `--disable-custom-all-reduce` (falls back to PYNCCL).
4. **pidfd_getfd permission** — add `--cap-add SYS_PTRACE --security-opt seccomp=unconfined
   --security-opt apparmor=unconfined`.

### The exact serve command (patched image) — CURRENT BEST (native context, CUDA graphs)
```bash
MODEL=/trunk/ai/huggingface/models/nvidia/Qwen3.8-Flash-Next-NVFP4
docker run -d --name qwen38-flash-serve --gpus all --shm-size 16g --ipc=host \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -v "$MODEL":/model:ro -p 8090:8000 \
  -e NCCL_P2P_DISABLE=1 -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_QWEN38_PLE_FP8_SCALE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm/vllm-openai:qwen38-flash-next-patched \
  /model --tensor-parallel-size 2 --quantization modelopt \
  --max-model-len 262144 --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 --disable-custom-all-reduce \
  --max-parallel-loading-workers 1 \
  --host 0.0.0.0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --trust-remote-code
```

> **Tool-choice flags are required** for the opencode client: opencode sends `tool_choice: "auto"`, which
> vLLM rejects unless the server is launched with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`.
> Without these you get: `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`.
>
> `--host 0.0.0.0` binds to all interfaces (docker `-p 8090:8000` already publishes on 0.0.0.0).

### Context length vs. concurrency (KV budget)
Native context is **262,144**. KV scales linearly; the box has ~24 GiB KV budget (PLE offloaded, gpu-util
0.90). To run full native context you MUST lower concurrency: `--max-model-len 262144 --max-num-seqs 2`
(verified working; KV ~15 GiB). `--max-num-seqs 64` only fits ~32k context. Use
`--max-model-len 131072 --max-num-seqs 4` for a middle ground.

### CUDA graphs vs `--enforce-eager` (measured)
- **CUDA graphs ON (no `--enforce-eager`):** ~**39–55 tok/s** (214 tokens in ~3.9 s).
- **`--enforce-eager` ON:** ~**12 tok/s** (~17.5 s for the same).
- **Conclusion:** drop `--enforce-eager` to enable CUDA graphs for ~4× decode speed. It is SAFE as long as
  the other burst-reduction flags (below) keep the warmup burst low; verified no trip while capturing graphs.

### Data-fabric burst-reduction flags (why these matter)
The sync-flood reset (see `power-trip-instances.md`, Instances 2/4/5) is triggered by the **weight-loading
burst** (~80 GB host→GPU over the Infinity Fabric / PCIe). These reduce the instantaneous fabric load and
were validated by the `--load-format dummy` isolation test (structure-only load did NOT trip):
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` — workers launch via `spawn` instead of `fork` (avoids the big
  copy-on-write memory storm of the 1 TB map at worker startup).
- `--max-parallel-loading-workers 1` — serialize concurrent disk→VRAM weight workers (default parallelizes
  across GPUs = massive simultaneous fabric burst).
- Plus an off-container **GPU power cap** (`nvidia-smi -pl` e.g. 250 W) to reduce peak draw.
- (Optional/fallback only if CUDA-graph warmup trips: add `--enforce-eager` — but it costs ~4× decode speed.)

**Still a hardware-figure gap:** these reduce *load* but do not fix the marginal DIMM (channel G / MM4) /
fabric timing. Long-term fix = memtest + RAM downclock + GPU→Gen3 (see `power-trip-instances.md`).

### Patched image / build
- Image: `vllm/vllm-openai:qwen38-flash-next-patched` (based on `qwen38-flash-next`).
- Build context lives in `/tmp/patched-build/` ({Dockerfile, ple_layer.py}) — **note `/tmp` is volatile
  across reboot**; if the image is gone, rebuild from `qwen38-flash-next` + `ple_layer.py` patch
  (patch is also described in `power-trip-instances.md` Instance 4 and `README-ramoffload-research.md`).

## Capture (manual; armed by the --start script)
- Container `powertrip-capture` (`restart: no`), image `powertrip-capture:local` — started on-demand via
  `recipe/serve-qwen38-flash-next-nvfp4.sh --start` (or `run-powertrip-capture.sh start`). Not auto-start.
- Also armed by the script: `edac-ce-watch` (corrected-ECC pre-trip monitor) and the GPU power cap.
- Launcher: `bash /home/praneet/Workspace/pensive-ai-research/run-powertrip-capture.sh start`.
- Captures to **`/buffer/powertrip/`** (persistent). Readable docs: `powertrip-capture-readme.md`.
- It writes telemetry CSV, raw klog (dmesg), edac CSV, summary. Sample interval ~1 s.
- **Manual, not auto-start** (per decision). Start it with the serve script (`--start`) or
  `run-powertrip-capture.sh start` before a model-load/debug session.

## Evidence / next deep-dive pointers
- Instance catalog: `power-trip-instances.md` (Instances 1–4). Evidence: `evidence/power-trips/`.
- Instance 4 crash-window telemetry: `evidence/power-trips/run-20260905-trip/telemetry-crashwindow.csv`
  and `ecc-mce-evidence.txt` (**channel 6 = channel G = slot MM4** CECC escalation → UC → sync-flood reset).
- Reset-reason check after any boot: `grep -a "Previous system reset" /var/log/syslog` and
  `journalctl -k -b -1 | grep -iE "mce|edac|channel#6|sync flood"`.

## Hardware fix (the gating item)
- **Reseat/swap DIMM slot MM4 (channel G)**, downclock memory 3200→2933/2666, run memtest86 on channel G.
  See `power-trip-diagnosis.md`. Until fixed, heavy both-GPU loads will keep tripping.

## On resume, run in this order
1. `bash run-powertrip-capture.sh status` (or `start`) → confirm capture live.
2. Verify reset reason from the prior boot (`journalctl -k -b -1 | grep -iE "mce|edac|reset"`).
3. Rebuild/confirm the patched image exists (`docker images | grep qwen38-flash-next-patched`).
4. Launch the serve command above; watch `/var/tmp/serve.log` (persistent, not `/tmp`).
5. Watch capture via `tail /buffer/powertrip/telemetry-*.csv` (pkgW=$13, tctl=$4, g0W=$23,g1W=$27).
6. Verify serving: `curl -s localhost:8090/v1/models` and a chat completion (use `max_tokens>=150` so
   the qwen3 reasoning block finishes and `content` is emitted).

## Proven-signature success run (2026-09-06 ~00:18 UTC)
- Startup reached `Started server process` + `Application startup complete` + `/v1/models` -> 200.
- Chat test: "What is 2+2?" -> `content: '4'` (29 reasoning tokens), both GPUs ~85-91% util.
- Software flags that made this work (all required together):
  `NCCL_P2P_DISABLE=1`, `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_QWEN38_PLE_FP8_SCALE=1`,
  `--disable-custom-all-reduce`, `--cap-add SYS_PTRACE` + unconfined seccomp/apparmor,
  `--quantization modelopt`.
- Startup is **slow** (~14 min load + ~1 min graph capture on this host; ZFS disk-bound), so don't
  mistake the long 82% shard plateau for a hang — sample worker CPU TIME to confirm it's advancing.
