#!/bin/bash
# serve-glm-53-flash.sh — STAGED launcher for zai-org/GLM-5.3-Flash on this box (TP2-class host).
# 320B/18B-active FP8 MoE, ~306 GiB weights vs ~144 GB VRAM => CPU weight-offload mandatory.
# Stages (one variable at a time; see recipe/GLM-53-FLASH-RECIPE.md):
#   --check    offline-only: assert Glm5Next arch + FlashInfer>=0.6.17 inside the image (no GPU, safe)
#   --dummy    dummy-weight TP2 smoke: arch + NCCL + sparse-MLA backend, NO real weights / no disk burst
#   --serve    real load. Default TP1 + --cpu-offload-gb 280, ctx 8192, seqs 1, eager (~1 tok/s expected)
#              override TP=2 CPU_OFFLOAD_GB=150 for the two-GPU step
#   --stop / --status
# Hard gates (power-trip on marginal DIMM channel G/MM4):
#   - refuses to run while qwen38-flash-serve is up (it owns all VRAM)
#   - arms powertrip-capture + edac-ce-watch, GPU power cap 250W, serialized loading
#   - logs EDAC CE baseline before launch
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/trunk/ai/huggingface/models/zai-org/GLM-5.3-Flash}"
IMAGE="${IMAGE:-vllm/vllm-openai:glm53-flash}"   # fallback: vllm/vllm-openai:nightly
NAME="${NAME:-glm53-flash-serve}"
PORT="${PORT:-8091}"
SERVED_MODEL="${SERVED_MODEL:-zai-org/GLM-5.3-Flash}"
GPU_POWER_CAP="${GPU_POWER_CAP:-250}"
TP="${TP:-1}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-280}"           # per worker; TP2 wants ~150
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
  log "CE baseline (channel#6=G=MM4 is the suspect):"
  ras-mc-ctl --summary 2>/dev/null | grep -E "channel#6" | sed 's/^/[glm53]   /' || true
}

drop_caches(){
  log "dropping page caches (reduce load-time RAM peak); needs root"
  sync && (echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || log "warn: drop_caches needs root (continuing)")
}

launch(){   # $1 = "dummy" | "real"
  local mode="$1" extra=()
  [ "$mode" = "dummy" ] && extra+=(--load-format dummy)
  docker rm -f "$NAME" >/dev/null 2>&1
  log "launching $NAME mode=$mode TP=$TP offload=${CPU_OFFLOAD_GB}GB/wkr ctx=$MAX_MODEL_LEN eager=1"
  nohup docker run -d --name "$NAME" \
    --gpus all --shm-size 16g --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$MODEL":/model:ro -p "$PORT":8000 \
    -e NCCL_P2P_DISABLE=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_KV_CACHE_LAYOUT=HND -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
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
  --dummy)  check_image; require_gpus_free; arm_safety; TP=2 launch dummy ;;
  --serve)  check_image; require_gpus_free; arm_safety; drop_caches; launch real ;;
  --stop)   docker rm -f "$NAME" >/dev/null 2>&1 && log "stopped $NAME" || log "no $NAME running" ;;
  --status) status ;;
  *) cat <<EOF
usage: $0 --check | --dummy | --serve | --stop | --status
env: IMAGE TP CPU_OFFLOAD_GB MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL GPU_POWER_CAP
see recipe/GLM-53-FLASH-RECIPE.md for the staged plan and risk gates
EOF
  ;;
esac
