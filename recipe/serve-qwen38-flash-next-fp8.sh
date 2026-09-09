#!/bin/bash
# serve-qwen38-flash-next-fp8.sh
# Launcher for Qwen/Qwen3.8-Flash-Next-FP8 (173 GB on disk) — the FP8-quantized sibling of the
# already-proven nvidia/Qwen3.8-Flash-Next-NVFP4 (recipe/serve-qwen38-flash-next-nvfp4.sh, "Recipe A").
# SAME architecture (Qwen4ExpForConditionalGeneration, 48 layers, 512 experts/10-per-tok, native
# 262144 ctx, 51B-param n-gram/PLE table) — just fp8 (1 byte/param routed-expert weights) instead of
# NVFP4 (0.5 byte/param). Confirmed 2026-09-09 via config.json/index.json inspection: this checkpoint's
# PLE table has the identical "single global ngram_embedding.weight_scale + per-shard .weight" layout
# the qwen38-flash-next-patched image's FP8-PLE-selector patch (VLLM_QWEN38_PLE_FP8_SCALE=1) targets —
# so this reuses that same patched image and flag, no new patch needed.
#
# VRAM math (see recipe/MODEL-CATALOG.md / SYSTEM-SPEC.md for the box's ~135 GB usable budget):
#   173 GB on disk - ~52 GB PLE table (offloaded to host RAM, same as NVFP4) ~= ~121 GB GPU-resident
#   weight, vs NVFP4's ~124-57=~67 GB. That's ~60.5 GB/GPU of weight alone (TP2), leaving only
#   ~11-12 GB/GPU for KV+activations+CUDA context — MUCH tighter than the NVFP4 recipe's headroom.
#   Full 262144 context needs a small model-weight offload to free real headroom for KV (see
#   CPU_OFFLOAD_GB default below) — validated 2026-09-09.
#
# CONFIRMED WORKING CONFIG (2026-09-09, real weights, this script's defaults): TP2, CUDA graphs ON,
# --cpu-offload-gb 8/worker, MTP speculative decoding (4 tokens), full 262144 ctx, seqs=1.
# Steady-state throughput ~20-24 tok/s (measured via vLLM's own /metrics — inter_token_latency and
# spec_decode_num_accepted/drafts counters, cross-checked against wall-clock; the *first* couple of
# requests after a fresh restart are much slower — one-time lazy-compilation warmup on the spec-decode
# path, not representative of steady state, don't judge throughput from request #1). That's ~2.5-3x
# over the original --enforce-eager, no-spec-decode baseline (~8 tok/s). Breakdown of what was tried:
#   - CUDA graphs alone (no spec): ~11.5 tok/s — a real but modest win (NVFP4's sibling recipe saw a
#     ~4x win from graphs alone; this being far smaller is itself evidence the bottleneck here is the
#     cross-GPU TP all-reduce over this box's no-NVLink/no-P2P interconnect, not kernel-launch overhead).
#   - Pipeline parallelism (TP=1 PP=2): DEAD END. vLLM raises `VLLM_PLE_CPU_OFFLOAD does not support
#     the requested configuration. Unsupported settings: PP=2` — PLE offload (mandatory here) has an
#     explicit guard against PP. Not fixable without patching vLLM; not attempted further.
#   - MTP speculative decoding (SPEC_METHOD=mtp): real win despite the comm-bound TP setup — the draft
#     path saw ~65-70% average per-token acceptance in testing, avg ~3.6-3.9 output tokens/round.
#     num_speculative_tokens=5 CRASHES: `QSA ring capacity 12 must divide the attention block size
#     1616` — the attention block size is computed dynamically (tied to mamba/attention page-size
#     alignment), so which spec_tokens values satisfy the divisibility isn't a simple formula; 4 is
#     the confirmed-working ceiling found by testing, don't assume 5+ works without retesting.
#
# Usage:
#   bash serve-qwen38-flash-next-fp8.sh              # stop any existing, then start
#   bash serve-qwen38-flash-next-fp8.sh --restart    # same as above (stop+start)
#   bash serve-qwen38-flash-next-fp8.sh --start      # start (leaves existing running)
#   bash serve-qwen38-flash-next-fp8.sh --stop       # stop the serve container
#   bash serve-qwen38-flash-next-fp8.sh --status     # status of serve + capture
#   bash serve-qwen38-flash-next-fp8.sh --no-capture # skip arming the capture logger
# Env overrides:
#   GPU_POWER_CAP=250   nvidia-smi -pl cap (needs root; this session found it fails under sudo -n —
#     verify manually with `nvidia-smi -pl 250` before a real run if you want the defensive margin)
#   MAX_MODEL_LEN=32768  MAX_NUM_SEQS=1     conservative starting point; raise once VRAM fit is proven
#   ENFORCE_EAGER=1     default ON here (unlike the NVFP4 recipe) — tighter VRAM margin means the
#     CUDA-graph warmup burst is riskier untested; flip to 0 once a stable eager run confirms headroom
#   CPU_OFFLOAD_GB=0    escape valve if weights don't fit even at low context — shaves GPU-resident
#     weight onto host RAM (same offload path GLM uses), 0 = disabled (rely on VRAM only)
#   LOAD_FORMAT=auto    set to "dummy" for a fast, low-risk smoke test (no 173 GB ZFS read, no real
#     weight I/O) before committing to a real load — mirrors recipe/serve-glm-53-flash.sh's --dummy
#   NCCL_CUMEM_HOST_ENABLE=auto|0|1   same memory-less-NUMA-node guard as the NVFP4/GLM recipes
#   TP=2 PP=1   tensor- vs pipeline-parallel split across the 2 GPUs (TP*PP must be 2). No NVLink/P2P
#     on this box (NCCL_P2P_DISABLE=1 always set) — TP needs an all-reduce at ~2 sync points per layer
#     (~96 round-trips/token over the slow socket fallback); PP needs only 1-2 handoffs/token. Set
#     TP=1 PP=2 to try pipeline parallelism instead. Confirmed 2026-09-09: the nvidia/model.py backend
#     properly declares SupportsPP and threads get_pp_group() through the forward pass — not a stub.
#   SPEC_METHOD=   (empty=disabled) set to "qwen3_8_flash_next_mtp" to enable this model's native MTP
#     speculative decoding (confirmed present: vllm/v1/spec_decode/qwen3_8_flash_next.py, built on the
#     standard EagleProposer; draft model config has mtp_num_hidden_layers=1)
#   SPEC_TOKENS=2   --speculative-config num_speculative_tokens, only used when SPEC_METHOD is set
#
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/trunk/ai/huggingface/models/Qwen/Qwen3.8-Flash-Next-FP8}"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next-patched}"
NAME="${NAME:-qwen38-flash-fp8-serve}"
PORT="${PORT:-8092}"
SERVED_MODEL="${SERVED_MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}"
GPU_POWER_CAP="${GPU_POWER_CAP:-250}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"             # CUDA graphs ON by default — validated 2026-09-09
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"          # model WEIGHT offload (OffloadConfig, --cpu-offload-gb); 8GB/worker needed to fit full 262144 ctx KV
KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-0}"  # KV CACHE offload to host RAM (CacheConfig, --kv-offloading-size), GiB, 0=disabled
KV_OFFLOADING_BACKEND="${KV_OFFLOADING_BACKEND:-native}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-auto}"
TP="${TP:-2}"
PP="${PP:-1}"                                   # PP>1 is a dead end here — see header note (PLE offload guard)
SPEC_METHOD="${SPEC_METHOD:-mtp}"               # confirmed working; set empty to disable
SPEC_TOKENS="${SPEC_TOKENS:-4}"                 # confirmed ceiling; 5 crashes (QSA ring-capacity divisibility)
SERVE_LOG="${SERVE_LOG:-/var/tmp/serve-qwen38-fp8.log}"
QWEN_NVFP4_NAME="qwen38-flash-serve"
GLM_NAME="glm53-flash-serve"

log(){ echo "[qwen-fp8] $*"; }
die(){ log "ERROR: $*"; exit 1; }

check_image(){
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "image $IMAGE missing (should already be pulled for the NVFP4 recipe): docker pull $IMAGE"
}

require_gpus_free(){
  for other in "$QWEN_NVFP4_NAME" "$GLM_NAME"; do
    if docker ps --filter name="$other" --format '{{.Names}}' | grep -q "$other"; then
      die "$other is running (owns VRAM). Stop it first."
    fi
  done
}

arm_capture(){
  if [ "${1:-}" = "--no-capture" ]; then log "skipping capture"; return 0; fi
  local cap="powertrip-capture"
  if docker ps --filter name="$cap" --format '{{.Names}}' | grep -q "$cap"; then
    log "capture already running ($cap)"
  else
    log "starting capture ($cap)"
    bash "$REPO_DIR/run-powertrip-capture.sh" start || log "warn: capture start failed (continuing)"
  fi
}

arm_edac_watch(){
  if [ "${1:-}" = "--no-edac" ]; then log "skipping EDAC CE watch"; return 0; fi
  if pgrep -f ".../edac-ce-watch.sh" >/dev/null 2>&1 || pgrep -x edac-ce-watch.sh >/dev/null 2>&1; then
    log "EDAC CE watch already running"
  else
    log "arming EDAC CE watch"
    nohup bash "$REPO_DIR/recipe/edac-ce-watch.sh" >/var/tmp/edac-ce-watch.out 2>&1 &
  fi
}

apply_power_cap(){
  log "setting GPU power cap to ${GPU_POWER_CAP} W (nvidia-smi -pl)"
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | while read -r i; do
    nvidia-smi -pl "$GPU_POWER_CAP" -i "$i" >/dev/null 2>&1 || log "warn: cap command failed on gpu $i (needs root — this session found it requires interactive sudo; apply manually if you want the margin)"
  done
  while IFS=, read -r i limit; do
    i="$(echo "$i" | tr -d ' ')"; limit="$(echo "$limit" | tr -d ' ')"
    limit_int="${limit%%.*}"
    if [ -n "$limit_int" ] && [ "$limit_int" -gt "$GPU_POWER_CAP" ] 2>/dev/null; then
      log "*** WARNING: gpu $i power limit is ${limit} W, NOT the requested ${GPU_POWER_CAP} W cap — cap did not apply ***"
    fi
  done < <(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits)
}

check_capture_alive(){
  local capdir="/buffer/powertrip"
  local latest; latest="$(ls -t "$capdir"/klog-*.log 2>/dev/null | head -1)"
  if [ -z "$latest" ]; then
    log "WARNING: no klog-*.log found under $capdir — capture may not be writing"
    return 0
  fi
  local age; age=$(( $(date +%s) - $(stat -c %Y "$latest" 2>/dev/null || echo 0) ))
  if [ "$age" -gt 60 ]; then
    log "WARNING: $latest last wrote ${age}s ago — capture may have gone quiet"
  else
    log "capture alive: $latest updated ${age}s ago"
  fi
}

check_numa_topology(){
  # Same guard as serve-qwen38-flash-next-nvfp4.sh / serve-glm-53-flash.sh: GPU0's local NUMA node
  # is currently memory-less (DIMM isolation testing) which segfaults NCCL's cuMemHostEnable probe
  # unless forced off.
  log "NUMA topology check (GPU-local node memory):"
  local empty_gpu_node=0
  local busid sysbus node mem_kb mem_mb
  while IFS=, read -r busid; do
    busid="$(echo "$busid" | tr -d ' \r')"
    [ -z "$busid" ] && continue
    sysbus="$(echo "${busid/#0000/}" | tr 'A-Z' 'a-z')"
    node="$(cat "/sys/bus/pci/devices/${sysbus}/numa_node" 2>/dev/null)"
    if [ -z "$node" ] || [ "$node" -lt 0 ] 2>/dev/null; then
      log "  gpu $busid: no NUMA affinity reported"
      continue
    fi
    mem_kb="$(awk '/MemTotal/{print $4}' "/sys/devices/system/node/node${node}/meminfo" 2>/dev/null)"
    mem_kb="${mem_kb:-0}"
    mem_mb=$(( mem_kb / 1024 ))
    log "  gpu $busid -> numa node $node (${mem_mb} MB local memory)"
    if [ "$mem_mb" -eq 0 ]; then
      log "  WARNING: gpu $busid's local NUMA node $node is memory-less (0 MB)"
      empty_gpu_node=1
    fi
  done < <(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader)

  case "$NCCL_CUMEM_HOST_ENABLE" in
    auto)
      if [ "$empty_gpu_node" = "1" ]; then
        RESOLVED_CUMEM_HOST_ENABLE=0
        log "-> memory-less GPU-local NUMA node detected; forcing NCCL_CUMEM_HOST_ENABLE=0"
      else
        RESOLVED_CUMEM_HOST_ENABLE=""
        log "-> all GPU-local NUMA nodes have memory; leaving NCCL_CUMEM_HOST_ENABLE at its default"
      fi
      ;;
    *)
      RESOLVED_CUMEM_HOST_ENABLE="$NCCL_CUMEM_HOST_ENABLE"
      log "-> NCCL_CUMEM_HOST_ENABLE explicitly overridden to $RESOLVED_CUMEM_HOST_ENABLE"
      ;;
  esac
}

drop_caches(){
  log "dropping page caches (reduce load-time RAM peak); needs root"
  sync && (echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || log "warn: drop_caches needs root (continuing)")
}

start_serve(){
  local extra=()
  [ "$ENFORCE_EAGER" = "1" ] && extra+=(--enforce-eager)
  [ "$LOAD_FORMAT" != "auto" ] && extra+=(--load-format "$LOAD_FORMAT")
  local offload_args=()
  if [ "${CPU_OFFLOAD_GB:-0}" != "0" ]; then
    offload_args+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
  fi
  if [ "${KV_OFFLOADING_SIZE:-0}" != "0" ]; then
    offload_args+=(--kv-offloading-size "$KV_OFFLOADING_SIZE" --kv-offloading-backend "$KV_OFFLOADING_BACKEND")
  fi
  local cumem_env=()
  if [ -n "${RESOLVED_CUMEM_HOST_ENABLE:-}" ]; then
    cumem_env+=(-e "NCCL_CUMEM_HOST_ENABLE=${RESOLVED_CUMEM_HOST_ENABLE}")
  fi
  # vLLM's KV-offloading connector (--kv-offloading-size) is incompatible with
  # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the CUDA VMM allocator can remap KV cache
  # virtual addresses, invalidating pinned/registered KV memory) unless the cumem allocator is
  # separately enabled — vLLM raises a pydantic ValidationError at startup otherwise. Simplest fix:
  # don't set expandable_segments when KV offload is in use.
  local alloc_conf_env=()
  if [ "${KV_OFFLOADING_SIZE:-0}" = "0" ]; then
    alloc_conf_env+=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
  else
    log "KV offload enabled: omitting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (incompatible with OffloadingConnector)"
  fi
  local spec_args=()
  if [ -n "$SPEC_METHOD" ]; then
    spec_args+=(--speculative-config "{\"method\":\"${SPEC_METHOD}\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
  fi
  docker rm -f "$NAME" >/dev/null 2>&1
  log "launching $NAME (model=$MODEL, served=$SERVED_MODEL, TP=$TP PP=$PP, ctx=$MAX_MODEL_LEN, seqs=$MAX_NUM_SEQS, eager=$ENFORCE_EAGER, load_format=$LOAD_FORMAT, cpu_offload=${CPU_OFFLOAD_GB}GB, kv_offload=${KV_OFFLOADING_SIZE}GiB/${KV_OFFLOADING_BACKEND}, spec=${SPEC_METHOD:-none}/${SPEC_TOKENS}, NCCL_CUMEM_HOST_ENABLE=${RESOLVED_CUMEM_HOST_ENABLE:-<default>})"
  nohup docker run -d --name "$NAME" \
    --gpus all --shm-size 16g --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$MODEL":/model:ro -p "$PORT":8000 \
    -e NCCL_P2P_DISABLE=1 -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_QWEN38_PLE_FP8_SCALE=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "${alloc_conf_env[@]}" \
    "${cumem_env[@]}" \
    "$IMAGE" \
    /model --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
    --served-model-name "$SERVED_MODEL" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --disable-custom-all-reduce \
    --max-parallel-loading-workers 1 \
    --host 0.0.0.0 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 --trust-remote-code \
    "${offload_args[@]}" "${spec_args[@]}" "${extra[@]}" \
    > "$SERVE_LOG" 2>&1 &
  log "serve launching; log=$SERVE_LOG (docker run -d returns fast; watch actual vLLM output with: docker logs -f $NAME)"
}

status(){
  docker ps -a --filter name="$NAME" --format 'serve: {{.Names}} {{.Status}}'
  docker ps -a --filter name=powertrip-capture --format 'capture: {{.Names}} {{.Status}}'
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
}

ACTION="${1:---restart}"
SKIP_CAP=""
SKIP_EDAC=""
case "$ACTION" in
  *--no-capture*) ACTION="--restart"; SKIP_CAP="--no-capture" ;;
  *--no-edac*)    ACTION="--restart"; SKIP_EDAC="--no-edac" ;;
esac

case "$ACTION" in
  --start|--restart)
    check_image; require_gpus_free
    arm_capture "$SKIP_CAP"; arm_edac_watch "$SKIP_EDAC"
    apply_power_cap; check_capture_alive; check_numa_topology
    [ "$LOAD_FORMAT" = "auto" ] && drop_caches
    docker rm -f "$NAME" >/dev/null 2>&1
    start_serve
    ;;
  --stop)
    docker rm -f "$NAME" >/dev/null 2>&1 && log "stopped $NAME" || log "no $NAME running" ;;
  --status)
    status ;;
  *)
    die "usage: $0 [--start|--restart|--stop|--status] [--no-capture] [--no-edac]" ;;
esac
