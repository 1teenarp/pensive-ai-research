#!/bin/bash
# serve-qwen38-flash-next-nvfp4.sh
# One-shot launcher for the Qwen3.8-Flash-Next-NVFP4 server on this build.
# Brings up (or restarts) everything with the CURRENT (proven, fast) settings:
#   - crash-capture telemetry (powertrip-capture) auto-armed
#   - GPU power cap (default 250 W)
#   - native 262k context, CUDA graphs ON (~4x faster decode)
#   - all data-fabric burst-reduction flags + PLE CPU offload
#
# Usage:
#   bash serve-qwen38-flash-next-nvfp4.sh              # stop any existing, then start
#   bash serve-qwen38-flash-next-nvfp4.sh --restart    # same as above (stop+start)
#   bash serve-qwen38-flash-next-nvfp4.sh --start      # start (leaves existing running)
#   bash serve-qwen38-flash-next-nvfp4.sh --stop       # stop the serve container
#   bash serve-qwen38-flash-next-nvfp4.sh --status     # status of serve + capture
#   bash serve-qwen38-flash-next-nvfp4.sh --no-capture # skip arming the capture logger
# Env overrides:
#   GPU_POWER_CAP=250   nvidia-smi -pl cap to apply to both GPUs
#   MAX_MODEL_LEN=262144  MAX_NUM_SEQS=2   context/concurrency (native context needs low concurrency)
#   ENFORCE_EAGER=1     fall back to --enforce-eager (slower ~1/4 decode) if CUDA-graph warmup trips
#
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/trunk/ai/huggingface/models/nvidia/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next-patched}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
NAME="${NAME:-qwen38-flash-serve}"
PORT="${PORT:-8090}"
SERVED_MODEL="${SERVED_MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
GPU_POWER_CAP="${GPU_POWER_CAP:-250}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
SERVE_LOG="${SERVE_LOG:-/var/tmp/serve.log}"   # persistent, not /tmp

# Pull-in the patched ple_layer.py from the patched-build dir if the image is missing.
PATCHED_PL="${PATCHED_PL:-/tmp/patched-build/ple_layer.py}"

log(){ echo "[serve] $*"; }
die(){ log "ERROR: $*"; exit 1; }

# ---------- helpers ----------
ensure_image(){
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "image $IMAGE present"
    return 0
  fi
  log "image $IMAGE missing; building from $BASE_IMAGE ..."
  if [ ! -f "$PATCHED_PL" ]; then
    log "extracting pristine ple_layer.py from $BASE_IMAGE"
    mkdir -p /tmp/patched-build
    docker run --rm --entrypoint bash "$BASE_IMAGE" \
      -c 'cat /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py' \
      > "$PATCHED_PL" || die "could not extract ple_layer.py"
  fi
  # Apply the FP8-PLE selector patch (vLLM issue #54765) if not present.
  if ! grep -q "VLLM_QWEN38_PLE_FP8_SCALE" "$PATCHED_PL"; then
    log "patching _get_ple_embedding_quant_method for FP8 PLE selection"
    python3 - "$PATCHED_PL" <<'PY'
import sys,sysconfig,re,os
p=sys.argv[1]
s=open(p).read()
if "import os" not in s.split("\n")[:8] and "import os" not in s:
    s=s.replace("import math","import math\nimport os",1)
needle='''def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):
        return None
'''
if needle in s:
    s=s.replace(needle, needle + '''
    # ModelOpt NVFP4 (MIXED_PRECISION) checkpoints can carry an FP8 PLE table
    # with one global ngram_embedding.weight_scale; opt in via env.
    if os.getenv("VLLM_QWEN38_PLE_FP8_SCALE","").lower() in ("1","true"):
        ignored = getattr(quant_config, "ignored_layers", None) or []
        sp = f"{prefix}.shard_"
        if not any(prefix.startswith(n.rstrip("*")) for n in ignored) and \\
           not any(n.startswith(sp) for n in ignored):
            return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
    ''', 1)
open(p,"w").write(s)
print("patched")
PY
  fi
  cd /tmp/patched-build
  printf 'FROM %s\nCOPY ple_layer.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py\nRUN python3 -c "import ast;ast.parse(open(\x27/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py\x27).read());print(\x27ok\x27)"\n' "$BASE_IMAGE" > Dockerfile
  docker build -t "$IMAGE" . || die "image build failed"
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

apply_power_cap(){
  log "setting GPU power cap to ${GPU_POWER_CAP} W (nvidia-smi -pl)"
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | while read -r i; do
    nvidia-smi -pl "$GPU_POWER_CAP" -i "$i" >/dev/null 2>&1 || log "warn: cap failed on gpu $i"
  done
}

arm_edac_watch(){
  if [ "${1:-}" = "--no-edac" ]; then log "skipping EDAC CE watch"; return 0; fi
  # Only match a running *loop* process (args = the script with no sub-args), not the current
  # launcher's own command line or a one-shot --status/--once invocation.
  if pgrep -f ".../edac-ce-watch.sh" >/dev/null 2>&1 || pgrep -x edac-ce-watch.sh >/dev/null 2>&1; then
    log "EDAC CE watch already running"
  else
    log "arming EDAC CE watch (pre-trip corrected-ECC monitor)"
    nohup bash "$REPO_DIR/recipe/edac-ce-watch.sh" >/var/tmp/edac-ce-watch.out 2>&1 &
  fi
}

start_serve(){
  local extra=()
  [ "$ENFORCE_EAGER" = "1" ] && extra+=(--enforce-eager)
  docker rm -f "$NAME" >/dev/null 2>&1
  log "launching $NAME (model=$MODEL, served=$SERVED_MODEL, ctx=$MAX_MODEL_LEN, seqs=$MAX_NUM_SEQS, eager=$ENFORCE_EAGER)"
  nohup docker run -d --name "$NAME" \
    --gpus all --shm-size 16g --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$MODEL":/model:ro -p "$PORT":8000 \
    -e NCCL_P2P_DISABLE=1 -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_QWEN38_PLE_FP8_SCALE=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$IMAGE" \
    /model --tensor-parallel-size 2 --quantization modelopt \
    --served-model-name "$SERVED_MODEL" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization 0.90 --disable-custom-all-reduce \
    --max-parallel-loading-workers 1 \
    --host 0.0.0.0 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 --trust-remote-code "${extra[@]}" \
    > "$SERVE_LOG" 2>&1 &
  log "serve launching; log=$SERVE_LOG  (takes ~10-15 min to load; watch with: tail -f $SERVE_LOG)"
}

status(){
  docker ps -a --filter name="$NAME" --format 'serve: {{.Names}} {{.Status}}'
  docker ps -a --filter name=powertrip-capture --format 'capture: {{.Names}} {{.Status}}'
}

# ---------- main ----------
ACTION="${1:---restart}"
SKIP_CAP=""
SKIP_EDAC=""
# normalize --no-capture / --no-edac (with or without extra args) to --restart
case "$ACTION" in
  *--no-capture*) ACTION="--restart"; SKIP_CAP="--no-capture" ;;
  *--no-edac*)    ACTION="--restart"; SKIP_EDAC="--no-edac" ;;
esac

case "$ACTION" in
  --start)
    ensure_image; arm_capture "$SKIP_CAP"; arm_edac_watch "$SKIP_EDAC"; apply_power_cap; start_serve ;;
  --restart)
    ensure_image; arm_capture "$SKIP_CAP"; arm_edac_watch "$SKIP_EDAC"; apply_power_cap; start_serve ;;
  --stop)
    docker rm -f "$NAME" >/dev/null 2>&1 && log "stopped $NAME" || log "no $NAME running" ;;
  --status)
    status ;;
  *)
    die "usage: $0 [--start|--restart|--stop|--status] [--no-capture] [--no-edac]" ;;
esac
