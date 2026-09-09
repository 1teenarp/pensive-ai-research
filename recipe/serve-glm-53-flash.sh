#!/bin/bash
# serve-glm-53-flash.sh — STAGED launcher for zai-org/GLM-5.3-Flash on this box (TP2-class host).
# 320B/18B-active FP8 MoE, ~306 GiB weights vs ~144 GB VRAM => CPU weight-offload mandatory.
# Stages (one variable at a time; see recipe/GLM-53-FLASH-RECIPE.md):
#   --check    offline-only: assert Glm5Next arch + FlashInfer>=0.6.17 inside the image (no GPU, safe)
#   --dummy    dummy-weight TP2 smoke: arch + NCCL + sparse-MLA backend, NO real weights / no disk burst
#              (CPU_OFFLOAD_GB auto-scales to 150/worker for TP2 unless overridden)
#   --serve    real load. Default TP1 + --cpu-offload-gb 280, ctx 8192, seqs 1, eager (~1 tok/s expected)
#              override TP=2 for the two-GPU step (CPU_OFFLOAD_GB auto-scales to 150)
#   --stop / --status
# Hard gates (power-trip on marginal DIMM channel G/MM4):
#   - refuses to run while qwen38-flash-serve is up (it owns all VRAM)
#   - arms powertrip-capture + edac-ce-watch, GPU power cap 250W, serialized loading
#   - logs EDAC CE baseline before launch, then verifies capture is actively writing (not just "up")
#   - probes GPU-local NUMA node memory and forces NCCL_CUMEM_HOST_ENABLE=0 if memory-less
#     (see check_numa_topology; ported from serve-qwen38-flash-next-nvfp4.sh — see SYSTEM-SPEC.md)
# History: Instance 8 (power-trip-instances.md) already ran --dummy once (2026-09-07, under the
# then-full 8-DIMM config) and died to a software 0xCF9 reset during fp8-MoE finalize — NOT the
# classic sync-flood. Root cause still open; see recipe/GLM-53-FLASH-RECIPE.md §3.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/trunk/ai/huggingface/models/zai-org/GLM-5.3-Flash}"
IMAGE="${IMAGE:-vllm/vllm-openai:glm53-flash}"   # fallback: vllm/vllm-openai:nightly
NAME="${NAME:-glm53-flash-serve}"
PORT="${PORT:-8091}"
SERVED_MODEL="${SERVED_MODEL:-zai-org/GLM-5.3-Flash}"
GPU_POWER_CAP="${GPU_POWER_CAP:-250}"
TP="${TP:-1}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-}"              # per worker; default resolved below (TP1=280, TP2=150)
NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-auto}"  # auto probes NUMA; see check_numa_topology()
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
SERVE_LOG="${SERVE_LOG:-/var/tmp/serve-glm53.log}"
QWEN_NAME="qwen38-flash-serve"

log(){ echo "[glm53] $*"; }
die(){ log "ERROR: $*"; exit 1; }

check_image(){
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "image $IMAGE missing: docker pull $IMAGE"
}

check_runtime(){   # stage 0: CPU-only assertions inside the image
  check_image
  log "checking arch + FlashInfer support inside $IMAGE (CPU-only, no GPU)"
  docker run --rm --entrypoint python3 "$IMAGE" -u -c "
import vllm
from vllm import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert any('Glm5Next' in a for a in archs), f'no Glm5Next arch; vllm={vllm.__version__}'
import importlib.metadata as md
fi = md.version('flashinfer-python')
t = tuple(int(x) for x in fi.split('.')[:3])
assert t >= (0,6,17), f'FlashInfer {fi} < 0.6.17 (sparse-MLA needs it)'
print(f'OK: vllm={vllm.__version__} flashinfer={fi} archs={[a for a in archs if \"Glm5\" in a]}')
" || die "runtime check FAILED: wrong/old image"
}

require_gpus_free(){
  if docker ps --filter name="$QWEN_NAME" --format '{{.Names}}' | grep -q "$QWEN_NAME"; then
    die "$QWEN_NAME is running (owns all VRAM + ~200GB RAM). Stop it first: recipe/serve-qwen38-flash-next-nvfp4.sh --stop"
  fi
}

resolve_offload_gb(){   # sets CPU_OFFLOAD_GB if the user didn't override it
  if [ -z "$CPU_OFFLOAD_GB" ]; then
    if [ "$TP" = "2" ]; then CPU_OFFLOAD_GB=150; else CPU_OFFLOAD_GB=280; fi
    log "CPU_OFFLOAD_GB not set; defaulting to ${CPU_OFFLOAD_GB} (TP=$TP)"
  fi
}

check_numa_topology(){
  # Same fix as recipe/serve-qwen38-flash-next-nvfp4.sh: NCCL's ncclCuMemHostEnable() probes
  # cuMemCreate on each GPU's *local* NUMA node during ncclCommInitRank. If that node is
  # memory-less (DIMMs pulled for the isolation/RMA testing — see SYSTEM-SPEC.md) the probe
  # segfaults instead of failing gracefully. Detect it and force NCCL_CUMEM_HOST_ENABLE=0.
  # NOTE: Instance 8's dummy TP2 run (power-trip-instances.md) predates this — it ran while all
  # 8 DIMMs were still populated, so it never hit this path. A rerun today WILL hit it unless
  # this guard is in place, since GPU0's local node (3) is currently memory-less.
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
  # Instances 3/6/7 (power-trip-instances.md) all found the capture logger silently stopped
  # writing mid-run, leaving the eventual trip/crash uncaptured. Instance 8 (the last GLM53
  # attempt) had the same gap: powertrip-capture's dmesg session went quiet ~27 min into the
  # ~3h run, well before the container actually died — the crash window itself was blind.
  # Verify the *current* klog file is actively growing, not just that the container is up.
  local capdir="/buffer/powertrip"
  local latest; latest="$(ls -t "$capdir"/klog-*.log 2>/dev/null | head -1)"
  if [ -z "$latest" ]; then
    log "WARNING: no klog-*.log found under $capdir — capture may not be writing"
    return 0
  fi
  local age; age=$(( $(date +%s) - $(stat -c %Y "$latest" 2>/dev/null || echo 0) ))
  if [ "$age" -gt 60 ]; then
    log "WARNING: $latest last wrote ${age}s ago — capture may have gone quiet (Instance 3/6/7/8 pattern). Consider: bash run-powertrip-capture.sh restart"
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
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | while read -r i; do
    nvidia-smi -pl "$GPU_POWER_CAP" -i "$i" >/dev/null 2>&1 || log "warn: cap failed on gpu $i"
  done
  log "CE baseline (channel#6=G=MM4 is the suspect; note ras-mc-ctl totals are LIFETIME cumulative, not per-run — see power-trip-instances.md 'Corrected findings'):"
  ras-mc-ctl --summary 2>/dev/null | grep -E "channel#6" | sed 's/^/[glm53]   /' || true
  check_capture_alive
}

drop_caches(){
  log "dropping page caches (reduce load-time RAM peak); needs root"
  sync && (echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || log "warn: drop_caches needs root (continuing)")
}

launch(){   # $1 = "dummy" | "real"
  local mode="$1" extra=()
  [ "$mode" = "dummy" ] && extra+=(--load-format dummy)
  local cumem_env=()
  if [ -n "${RESOLVED_CUMEM_HOST_ENABLE:-}" ]; then
    cumem_env+=(-e "NCCL_CUMEM_HOST_ENABLE=${RESOLVED_CUMEM_HOST_ENABLE}")
  fi
  docker rm -f "$NAME" >/dev/null 2>&1
  log "launching $NAME mode=$mode TP=$TP offload=${CPU_OFFLOAD_GB}GB/wkr ctx=$MAX_MODEL_LEN eager=1 NCCL_CUMEM_HOST_ENABLE=${RESOLVED_CUMEM_HOST_ENABLE:-<default>}"
  nohup docker run -d --name "$NAME" \
    --gpus all --shm-size 16g --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$MODEL":/model:ro -p "$PORT":8000 \
    -e NCCL_P2P_DISABLE=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_KV_CACHE_LAYOUT=HND -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
    "${cumem_env[@]}" \
    "$IMAGE" \
    /model --tensor-parallel-size "$TP" \
    --served-model-name "$SERVED_MODEL" \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --disable-custom-all-reduce \
    --max-parallel-loading-workers 1 --enforce-eager \
    --host 0.0.0.0 \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    "${extra[@]}" \
    > "$SERVE_LOG" 2>&1 &
  log "launched; log=$SERVE_LOG (real load from ZFS: expect 30-60+ min; watch: tail -f $SERVE_LOG)"
}

status(){
  docker ps -a --filter name="$NAME" --format 'serve: {{.Names}} {{.Status}}'
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
  free -g | awk 'NR==2{print "  ram used/avail GB: "$3"/"$7}'
  tail -5 "$SERVE_LOG" 2>/dev/null | sed 's/^/  /'
}

case "${1:---help}" in
  --check)  check_runtime ;;
  --dummy)  TP=2; check_image; require_gpus_free; resolve_offload_gb; arm_safety; check_numa_topology; launch dummy ;;
  --serve)  check_image; require_gpus_free; resolve_offload_gb; arm_safety; check_numa_topology; drop_caches; launch real ;;
  --stop)   docker rm -f "$NAME" >/dev/null 2>&1 && log "stopped $NAME" || log "no $NAME running" ;;
  --status) status ;;
  *) cat <<EOF
usage: $0 --check | --dummy | --serve | --stop | --status
env: IMAGE TP CPU_OFFLOAD_GB MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL GPU_POWER_CAP
see recipe/GLM-53-FLASH-RECIPE.md for the staged plan and risk gates
EOF
  ;;
esac
