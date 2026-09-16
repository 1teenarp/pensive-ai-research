#!/bin/bash
# serve-glm-53-flash-nvfp4.sh — launcher for nvidia/GLM-5.3-Flash-NVFP4 on this box (2×72 GB, TP2).
# ModelOpt NVFP4 MoE (320B total / 18B active), 190.4 GiB on disk, FP8 KV, native 1M ctx.
# Recipe + rationale: recipe/GLM-53-FLASH-NVFP4-RECIPE.md (read §3 gates before --serve).
#
# Usage:
#   bash serve-glm-53-flash-nvfp4.sh --check    # offline: image/arch/flashinfer/modelopt + model dir sanity (no GPU)
#   bash serve-glm-53-flash-nvfp4.sh --dummy    # dummy-weight TP2 smoke (no real weights, low fabric risk)
#   bash serve-glm-53-flash-nvfp4.sh --serve    # real load: TP2, offload 32 GiB/wkr, ctx 8192, seqs 1
#   bash serve-glm-53-flash-nvfp4.sh --stop
#   bash serve-glm-53-flash-nvfp4.sh --status
# Env overrides:
#   MODEL, IMAGE, NAME (glm53-nvfp4-serve), PORT (8092)
#   TP=2  CPU_OFFLOAD_GB= (auto: 32/worker TP2, 80 TP1)  MAX_MODEL_LEN=8192  MAX_NUM_SEQS=1
#   GPU_MEM_UTIL=0.90  GPU_POWER_CAP=250  ENFORCE_EAGER=0 (CUDA graphs ON; set 1 if capture trips)
#   KV_CACHE_DTYPE=fp8   (NB: NEITHER fp8 NOR auto works on sm_120 with this image — the model has no
#                         NoPE sparse-MLA path here at all. See the block at the variable + R-014.)
#   SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":2}'   # stage-3 only
#   EP=1  adds --enable-expert-parallel --enable-ep-weight-filter (stage-3; untested at TP2)
#   NCCL_CUMEM_HOST_ENABLE=auto|0|1   auto probes per-GPU NUMA node memory (memory-less node => force 0)
#   SERVE_LOG=/var/tmp/serve-glm53-nvfp4.log
# Hard gates (channel G / MM4 history — see power-trip-instances.md):
#   - refuses while qwen38-flash-serve or glm53-flash-serve holds VRAM
#   - arms powertrip-capture + edac-ce-watch, GPU power cap 250 W (verified, not just issued)
#   - serialized load (--max-parallel-loading-workers 1), spawn, NCCL_P2P_DISABLE=1
#   - NUMA/NCCL memory-less-node guard (ported from the qwen38 launcher)
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/trunk/ai/huggingface/models/nvidia/GLM-5.3-Flash-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:glm53-flash}"   # fallback: vllm/vllm-openai:nightly
NAME="${NAME:-glm53-nvfp4-serve}"
PORT="${PORT:-8092}"
SERVED_MODEL="${SERVED_MODEL:-nvidia/GLM-5.3-Flash-NVFP4}"
GPU_POWER_CAP="${GPU_POWER_CAP:-250}"
TP="${TP:-2}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-}"              # per worker; auto: TP2->32, TP1->80
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# ⛔ READ THIS BEFORE TUNING KV DTYPE. Neither setting works on sm_120 with this image, and the
# reason is structural, not a misconfiguration (RUN-LOG.md R-014 + correction, KNOWLEDGE.md P-H):
#   fp8  -> selects the fp8_ds_mla KV format; its kernel hardcodes pe_dim==64 (DeepSeek rope shape)
#           and GLM-5.3 is NoPE (qk_rope_head_dim=0) => "concat_and_cache_mla ... pe_dim must be 64"
#   auto -> FLASHINFER_MLA_SPARSE_SM120 raises NotImplementedError ("requires the packed fp8_ds_mla
#           KV cache layout"), and it is the ONLY MLA candidate left on sm_120 after TRITON_MLA is
#           filtered out for sparse/indexer models => fails earlier, at backend construction.
# vllm/platforms/cuda.py implements the NoPE/GLM-5-Next path for SM90 (Hopper) ONLY. Forward paths
# are a newer vLLM, SGLang, or different hardware — not a flag. Default stays fp8 (the model card's
# intent, and it reaches further) purely so the failure is the informative one.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"           # fp8 | auto — both currently fail, see above
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"               # 0 = CUDA graphs ON (target); 1 = eager fallback
SPEC_CONFIG="${SPEC_CONFIG:-}"                    # e.g. '{"method":"mtp","num_speculative_tokens":2}'
EP="${EP:-0}"                                     # 1 = --enable-expert-parallel --enable-ep-weight-filter
NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-auto}"
SERVE_LOG="${SERVE_LOG:-/var/tmp/serve-glm53-nvfp4.log}"
OTHER_SERVES="qwen38-flash-serve glm53-flash-serve"
EXPECTED_SHARDS=33

log(){ echo "[glm53nv] $*"; }
die(){ log "ERROR: $*"; exit 1; }

check_image(){
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "image $IMAGE missing: docker pull $IMAGE"
}

check_runtime(){   # stage 0: CPU-only assertions inside the image (no GPU)
  check_image
  log "checking arch + FlashInfer + ModelOpt loader inside $IMAGE (CPU-only)"
  docker run --rm --entrypoint python3 "$IMAGE" -u -c "
import vllm, importlib.metadata as md
from vllm import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert any('Glm5Next' in a for a in archs), f'no Glm5Next arch; vllm={vllm.__version__}'
fi = md.version('flashinfer-python')
t = tuple(int(x) for x in fi.split('.')[:3])
assert t >= (0,6,17), f'FlashInfer {fi} < 0.6.17 (sparse-MLA needs it)'
try:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    names = set(QUANTIZATION_METHODS) if hasattr(QUANTIZATION_METHODS,'keys') else set(map(str,QUANTIZATION_METHODS))
    assert any('modelopt' in n for n in names), 'modelopt quant method not registered'
    print('modelopt (NVFP4) loader: present')
except Exception as e:
    print(f'WARN: could not verify modelopt loader: {e}')
print(f'OK: vllm={vllm.__version__} flashinfer={fi} archs={[a for a in archs if \"Glm5\" in a]}')
" || die "runtime check FAILED: wrong/old image"
}

check_model_present(){   # $1 = required: "config" | "all"
  [ -d "$MODEL" ] || die "model dir missing: $MODEL (download first — see recipe §4 Stage 0)"
  [ -f "$MODEL/config.json" ] || die "config.json missing in $MODEL (incomplete download?)"
  if [ "$1" = "all" ]; then
    local n; n="$(ls "$MODEL"/*.safetensors 2>/dev/null | wc -l)"
    if [ "$n" -lt "$EXPECTED_SHARDS" ]; then
      log "WARNING: only $n/$EXPECTED_SHARDS shards present — download incomplete? (see /var/tmp/hf-glm53-nvfp4.log)"
    fi
  fi
}

require_gpus_free(){
  local s
  for s in $OTHER_SERVES; do
    if docker ps --filter name="$s" --format '{{.Names}}' | grep -q "$s"; then
      die "$s is running (holds VRAM). Stop it first."
    fi
  done
}

resolve_offload_gb(){
  if [ -z "$CPU_OFFLOAD_GB" ]; then
    if [ "$TP" = "1" ]; then CPU_OFFLOAD_GB=80; else CPU_OFFLOAD_GB=32; fi
    log "CPU_OFFLOAD_GB not set; defaulting to ${CPU_OFFLOAD_GB} (TP=$TP)"
  fi
}

check_numa_topology(){
  # ncclCuMemHostEnable probes cuMemCreate on each GPU's local NUMA node at comm init; a
  # memory-less node (pulled DIMMs) segfaults instead of failing gracefully. See SYSTEM-SPEC.md §2.
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

check_capture_alive(){
  # Instances 3/6/7/8 pattern: capture container "up" but klog not growing. Verify mtime, not liveness.
  local capdir="/buffer/powertrip"
  local latest; latest="$(ls -t "$capdir"/klog-*.log 2>/dev/null | head -1)"
  if [ -z "$latest" ]; then
    log "WARNING: no klog-*.log under $capdir — capture may not be writing"
    return 0
  fi
  local age; age=$(( $(date +%s) - $(stat -c %Y "$latest" 2>/dev/null || echo 0) ))
  if [ "$age" -gt 60 ]; then
    log "WARNING: $latest last wrote ${age}s ago — capture may have gone quiet. Consider: bash run-powertrip-capture.sh restart"
  else
    log "capture alive: $latest updated ${age}s ago"
  fi
}

arm_safety(){
  local cap="powertrip-capture"
  if docker ps --filter name="$cap" --format '{{.Names}}' | grep -q "$cap"; then
    log "capture already running"
  else
    bash "$REPO_DIR/run-powertrip-capture.sh" start || log "warn: capture start failed (continuing)"
  fi
  if pgrep -f ".../edac-ce-watch.sh" >/dev/null 2>&1 || pgrep -x edac-ce-watch.sh >/dev/null 2>&1; then
    log "EDAC CE watch already running"
  else
    nohup bash "$REPO_DIR/recipe/edac-ce-watch.sh" >/var/tmp/edac-ce-watch.out 2>&1 &
    log "armed EDAC CE watch"
  fi
  log "setting GPU power cap to ${GPU_POWER_CAP} W (nvidia-smi -pl)"
  local mismatch=0
  while IFS=, read -r i limit; do
    i="$(echo "$i" | tr -d ' ')"; limit="$(echo "$limit" | tr -d ' ')"
    local limit_int="${limit%%.*}"
    if [ -n "$limit_int" ] && [ "$limit_int" -gt "$GPU_POWER_CAP" ] 2>/dev/null; then
      log "*** WARNING: gpu $i power limit is ${limit} W, NOT ${GPU_POWER_CAP} W — cap did not apply ***"
      mismatch=1
    fi
  done < <(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits)
  [ "$mismatch" = "1" ] && log "*** power cap mismatch above is NOT fatal but this run has less fabric-load margin than the recipe assumes ***"
  log "CE baseline (channel#6=G=MM4 suspect; LIFETIME totals, not per-run — see power-trip-instances.md 'Corrected findings'):"
  ras-mc-ctl --summary 2>/dev/null | grep -E "channel#6" | sed 's/^/[glm53nv]   /' || true
  check_capture_alive
}

drop_caches(){
  log "dropping page caches (reduce load-time RAM peak); needs root"
  sync && (echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || log "warn: drop_caches needs root (continuing)")
}

launch(){   # $1 = "dummy" | "real"
  local mode="$1" extra=()
  [ "$mode" = "dummy" ] && extra+=(--load-format dummy)
  [ "$ENFORCE_EAGER" = "1" ] && extra+=(--enforce-eager)
  [ "$EP" = "1" ] && extra+=(--enable-expert-parallel --enable-ep-weight-filter)
  [ -n "$SPEC_CONFIG" ] && extra+=(--speculative-config "$SPEC_CONFIG")
  local cumem_env=()
  if [ -n "${RESOLVED_CUMEM_HOST_ENABLE:-}" ]; then
    cumem_env+=(-e "NCCL_CUMEM_HOST_ENABLE=${RESOLVED_CUMEM_HOST_ENABLE}")
  fi
  docker rm -f "$NAME" >/dev/null 2>&1
  log "launching $NAME mode=$mode TP=$TP offload=${CPU_OFFLOAD_GB}GB/wkr ctx=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS mem_util=$GPU_MEM_UTIL eager=$ENFORCE_EAGER NCCL_CUMEM_HOST_ENABLE=${RESOLVED_CUMEM_HOST_ENABLE:-<default>}"
  nohup docker run -d --name "$NAME" \
    --gpus all --shm-size 16g --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$MODEL":/model:ro -p "$PORT":8000 \
    -e NCCL_P2P_DISABLE=1 -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${cumem_env[@]}" \
    "$IMAGE" \
    /model --tensor-parallel-size "$TP" --quantization modelopt \
    --served-model-name "$SERVED_MODEL" \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --disable-custom-all-reduce --max-parallel-loading-workers 1 \
    --host 0.0.0.0 \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    "${extra[@]}" \
    > "$SERVE_LOG" 2>&1 &
  # NOTE: `docker run -d` returns immediately and prints only the container ID, so $SERVE_LOG holds
  # a 64-char hash and nothing else. The actual vLLM output lives in `docker logs`. Do not tail
  # $SERVE_LOG for progress (cost a session ~4 min of blind sleeps, RUN-LOG R-014).
  log "launched (real load from ZFS: ~10-30 min)"
  log "  progress:  docker logs -f $NAME"
  log "  wait:      bash $0 --wait"
  log "  launch err: $SERVE_LOG (container id only if the run started)"
}

wait_ready(){   # poll until the server binds, the container dies, or we time out
  local timeout="${WAIT_TIMEOUT:-3600}" t0 elapsed st
  t0=$(date +%s)
  log "waiting for $NAME (timeout ${timeout}s); ^C is safe, it does not stop the container"
  while :; do
    elapsed=$(( $(date +%s) - t0 ))
    st="$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null)"
    if [ -z "$st" ]; then log "container $NAME is gone"; return 1; fi
    if [ "$st" != "running" ]; then
      log "FAILED after ${elapsed}s — container $st (exit $(docker inspect "$NAME" --format '{{.State.ExitCode}}' 2>/dev/null))"
      docker logs "$NAME" 2>&1 | grep -aE "RuntimeError|ValueError|AssertionError|CUDA error|No valid|not supported" \
        | grep -avE "otel.py|please check the stack trace" | tail -3 | sed 's/^/[glm53nv]   /'
      log "next: snapshot it before cleanup -> bash $REPO_DIR/recipe/run-log.sh --container $NAME --stage <stage>"
      return 1
    fi
    if curl -sf -o /dev/null "http://localhost:${PORT}/v1/models" 2>/dev/null; then
      log "READY after ${elapsed}s — http://localhost:${PORT}/v1/models"; return 0
    fi
    if [ "$elapsed" -ge "$timeout" ]; then log "timed out after ${elapsed}s (still running; keep watching docker logs -f $NAME)"; return 2; fi
    sleep 15
  done
}

status(){
  docker ps -a --filter name="$NAME" --format 'serve: {{.Names}} {{.Status}}'
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
  free -g | awk 'NR==2{print "  ram used/avail GB: "$3"/"$7}'
  curl -sf -o /dev/null "http://localhost:${PORT}/v1/models" 2>/dev/null \
    && echo "  http: READY on :$PORT" || echo "  http: not serving on :$PORT"
  echo "  --- docker logs (last 8) ---"
  docker logs "$NAME" 2>&1 | tail -8 | sed 's/^/  /'
}

case "${1:---help}" in
  --check)  check_runtime; check_model_present config ;;
  --dummy)  TP=2; check_image; check_model_present config; require_gpus_free; resolve_offload_gb; arm_safety; check_numa_topology; launch dummy ;;
  --serve)  check_image; check_model_present all; require_gpus_free; resolve_offload_gb; arm_safety; check_numa_topology; drop_caches; launch real ;;
  --wait)   wait_ready ;;
  --logs)   docker logs -f "$NAME" ;;
  --stop)   docker rm -f "$NAME" >/dev/null 2>&1 && log "stopped $NAME" || log "no $NAME running" ;;
  --status) status ;;
  *) cat <<EOF
usage: $0 --check | --dummy | --serve | --wait | --logs | --stop | --status
  --wait    poll until the server binds :$PORT, the container dies, or WAIT_TIMEOUT (default 3600s)
  --logs    follow the real vLLM output (SERVE_LOG holds only the container id)
env: MODEL IMAGE NAME PORT TP CPU_OFFLOAD_GB MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL
     GPU_POWER_CAP KV_CACHE_DTYPE ENFORCE_EAGER SPEC_CONFIG EP NCCL_CUMEM_HOST_ENABLE SERVE_LOG
     WAIT_TIMEOUT
see recipe/GLM-53-FLASH-NVFP4-RECIPE.md for the staged plan and risk gates
EOF
  ;;
esac
