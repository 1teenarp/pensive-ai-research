#!/bin/bash
# serve-glm-53-flash-kt.sh — KTransformers (CPU-experts) launcher for zai-org/GLM-5.3-Flash.
#
# Why: vLLM offload path is comm-bound at ~2.4 tok/s (R-020). KTransformers moves routed experts
# to CPU (execute-in-place), GPU does attention/dense/shared, TP1 → no all-reduce floor.
# P10 estimate: 5–12 tok/s on this AVX2-only EPYC.
#
# Provenance (P12, user-directed 2026-09-30): kvcache-ai/KTransformers v0.7.0.post4 (Apache-2.0)
# from official PyPI into our own image. Weights: vendor zai-org FP8 (no conversion, no 3rd-party
# quant on default path). LLAMAFILE/GGUF fallback is P12-flagged (third-party quant).
#
# Recipe: recipe/GLM-53-FLASH-KT-RECIPE.md
# Image:  pensive/glm53-kt:latest (Dockerfile: recipe/Dockerfile.glm53-kt)
#
# Usage:
#   bash serve-glm-53-flash-kt.sh --check    # CPU-only: image, imports, model dir, RAM
#   bash serve-glm-53-flash-kt.sh --smoke    # Stage 1: real weights, ctx 8192, GPU0
#   bash serve-glm-53-flash-kt.sh --serve    # Stage 2: real weights, ctx 501025
#   bash serve-glm-53-flash-kt.sh --wait     # poll until bind/die/timeout
#   bash serve-glm-53-flash-kt.sh --logs     # docker logs -f
#   bash serve-glm-53-flash-kt.sh --stop     # remove container
#   bash serve-glm-53-flash-kt.sh --status   # one-shot snapshot
#
# Env overrides:
#   MODEL, CPU_WEIGHTS, IMAGE, NAME, PORT, GPU, KT_METHOD, KT_CPUINFER, KT_THREADPOOL,
#   KT_NUM_GPU_EXPERTS, KT_GPU_PREFILL_THRESHOLD, MEM_FRACTION, CHUNKED_PREFILL,
#   SMOKE_CTX, CTX, CUDAGRAPH_BS, MAX_RUNNING_REQUESTS, GPU_POWER_CAP,
#   BIND_HOST, API_KEY, SERVE_LOG, WAIT_TIMEOUT
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/trunk/ai/huggingface/models/zai-org/GLM-5.3-Flash}"
CPU_WEIGHTS="${CPU_WEIGHTS:-$MODEL}"
IMAGE="${IMAGE:-pensive/glm53-kt:latest}"
NAME="${NAME:-glm53-kt-serve}"
PORT="${PORT:-8093}"
SERVED_MODEL="${SERVED_MODEL:-zai-org/GLM-5.3-Flash}"
GPU="${GPU:-0}"
KT_METHOD="${KT_METHOD:-FP8}"
KT_CPUINFER="${KT_CPUINFER:-56}"
KT_THREADPOOL="${KT_THREADPOOL:-4}"
KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-0}"
KT_GPU_PREFILL_THRESHOLD="${KT_GPU_PREFILL_THRESHOLD:-2048}"
MEM_FRACTION="${MEM_FRACTION:-0.65}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
SMOKE_CTX="${SMOKE_CTX:-8192}"
CTX="${CTX:-501025}"
CUDAGRAPH_BS="${CUDAGRAPH_BS:-1 2 4}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
GPU_POWER_CAP="${GPU_POWER_CAP:-250}"
SERVE_LOG="${SERVE_LOG:-/var/tmp/serve-glm53-kt.log}"
BIND_HOST="${BIND_HOST:-}"
API_KEY="${API_KEY:-}"
OTHER_SERVES="qwen38-flash-serve qwen38-flash-fp8-serve glm53-flash-serve glm53-nvfp4-serve"
EXPECTED_SHARDS=62

log(){ echo "[glm53kt] $*"; }
die(){ log "ERROR: $*"; exit 1; }

probe_url(){
  local ip host="localhost"
  ip="$(docker inspect "$NAME" --format '{{range $p, $conf := .NetworkSettings.Ports}}{{if $conf}}{{(index $conf 0).HostIP}}{{end}}{{end}}' 2>/dev/null | head -1)"
  if [ -n "$ip" ] && [ "$ip" != "0.0.0.0" ] && [ "$ip" != "::" ]; then host="$ip"; fi
  echo "http://${host}:${PORT}/v1/models"
}
probe_key(){
  if [ -n "$API_KEY" ]; then echo "$API_KEY"; return; fi
  docker inspect "$NAME" --format '{{join .Args " "}}' 2>/dev/null \
    | awk '{for(i=1;i<NF;i++) if($i=="--api-key"){print $(i+1); exit}}'
}
http_probe(){
  local url key; url="$(probe_url)"; key="$(probe_key)"
  if [ -n "$key" ]; then curl -sf -o /dev/null -H "Authorization: Bearer $key" "$url" 2>/dev/null
  else curl -sf -o /dev/null "$url" 2>/dev/null; fi
}

check_image(){
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "image $IMAGE missing: docker build -f $REPO_DIR/recipe/Dockerfile.glm53-kt -t $IMAGE $REPO_DIR/recipe"
}

check_runtime(){
  check_image
  log "checking kt-kernel + sglang-kt inside $IMAGE (CPU-only)"
  local out rc
  out="$(docker run --rm -i -v "$MODEL":/model:ro --entrypoint python3 "$IMAGE" - <<'PYCHECK' 2>&1
import ctypes
ctypes.CDLL("libnuma.so.1")
print("libnuma.so.1: OK (R-030 gate: layerwise-prefill NUMA buffers)")
import kt_kernel, importlib.metadata as md, json, pathlib
v = kt_kernel.__cpu_variant__
print('kt-kernel', kt_kernel.__version__, 'cpu-variant:', v)
assert v in ('avx2', 'avx512', 'amx'), 'unexpected CPU variant: %s' % v
import sglang
print('sglang-kt', md.version('sglang-kt'))
# Glm5Next model module exists in sglang-kt (full import needs GPU for sgl_kernel)
models_dir = pathlib.Path(sglang.__file__).parent / 'srt' / 'models'
glm_files = sorted(p.name for p in models_dir.glob('glm5_next*.py'))
assert glm_files, 'no glm5_next*.py model files in sglang-kt'
print('model files:', glm_files)
# config.json parse (CPU-only)
cfg = json.load(open('/model/config.json'))
archs = ' '.join(cfg.get('architectures', []))
assert 'Glm5Next' in archs, 'arch mismatch: %s' % archs
print('config OK:', cfg.get('model_type'), archs)
print('NOTE: full model-class import requires GPU (sgl_kernel); verified at --smoke')
print('OK')
PYCHECK
)"; rc=$?
  echo "$out" | sed 's/^/[glm53kt]   /'
  [ "$rc" -eq 0 ] || die "runtime check FAILED (docker exit $rc)"

  log "host ISA check:"
  if grep -q 'avx512f' /proc/cpuinfo; then
    log "  AVX-512: YES"
  else
    log "  AVX-512: NO (AVX2-only; FP8 CPU kernel may be limited)"
  fi
  if grep -q 'amx_tile' /proc/cpuinfo; then
    log "  AMX: YES"
  else
    log "  AMX: NO"
  fi

  local avail_gb
  avail_gb=$(( $(free -b | awk 'NR==2{print $7}') / 1000000000 ))
  log "RAM available: ${avail_gb} GB (need >= 350 GB)"
  [ "$avail_gb" -lt 350 ] && die "insufficient RAM: ${avail_gb} GB < 350 GB required"

  log "model dir check:"
  [ -d "$MODEL" ] || die "model dir missing: $MODEL"
  [ -f "$MODEL/config.json" ] || die "config.json missing"
  local shards; shards=$(ls "$MODEL"/*.safetensors 2>/dev/null | wc -l)
  log "  shards: $shards / $EXPECTED_SHARDS"
  [ "$shards" -lt "$EXPECTED_SHARDS" ] && die "incomplete model: $shards/$EXPECTED_SHARDS shards"
}

check_model_present(){
  [ -d "$MODEL" ] || die "model dir missing: $MODEL"
  [ -f "$MODEL/config.json" ] || die "config.json missing in $MODEL"
  if [ "$1" = "all" ]; then
    local n; n="$(ls "$MODEL"/*.safetensors 2>/dev/null | wc -l)"
    [ "$n" -lt "$EXPECTED_SHARDS" ] && log "WARNING: only $n/$EXPECTED_SHARDS shards"
  fi
}

require_gpus_free(){
  local s
  for s in $OTHER_SERVES; do
    if docker ps --filter name="$s" --format '{{.Names}}' | grep -q "$s"; then
      die "$s is running (holds GPU). Stop it first: docker rm -f $s"
    fi
  done
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{if($1>1024) found=1} END{exit !found}')
  [ $? -eq 0 ] && die "GPU memory in use (>1 GB). Wait for it to free or stop the process."
}

check_numa(){
  log "NUMA topology (informational; KT is NUMA-aware internally):"
  numactl -H 2>/dev/null | grep -E 'node [0-9]+ size' | sed 's/^/  /'
}

check_capture_alive(){
  local capdir="/buffer/powertrip"
  local latest; latest="$(ls -t "$capdir"/klog-*.log 2>/dev/null | head -1)"
  if [ -z "$latest" ]; then
    log "WARNING: no klog-*.log under $capdir — capture may not be writing"
    return 0
  fi
  local age; age=$(( $(date +%s) - $(stat -c %Y "$latest" 2>/dev/null || echo 0) ))
  if [ "$age" -gt 60 ]; then
    log "WARNING: $latest last wrote ${age}s ago — capture may have gone quiet"
  else
    log "capture alive: $latest updated ${age}s ago"
  fi
}

arm_safety(){
  local cap="powertrip-capture"
  if docker ps --filter name="$cap" --format '{{.Names}}' | grep -q "$cap"; then
    log "capture already running"
  else
    bash "$REPO_DIR/run-powertrip-capture.sh" start 2>/dev/null || log "warn: capture start failed (continuing)"
  fi
  if pgrep -f "edac-ce-watch.sh" >/dev/null 2>&1; then
    log "EDAC CE watch already running"
  else
    nohup bash "$REPO_DIR/recipe/edac-ce-watch.sh" >/var/tmp/edac-ce-watch.out 2>&1 &
    log "armed EDAC CE watch"
  fi
  log "GPU power cap target: ${GPU_POWER_CAP} W (needs sudo; will warn if not applied)"
  nvidia-smi -pl "$GPU_POWER_CAP" 2>/dev/null
  local mismatch=0
  while IFS=, read -r i limit; do
    i="$(echo "$i" | tr -d ' ')"; limit="$(echo "$limit" | tr -d ' ')"
    local limit_int="${limit%%.*}"
    if [ -n "$limit_int" ] && [ "$limit_int" -gt "$GPU_POWER_CAP" ] 2>/dev/null; then
      log "*** WARNING: gpu $i power limit is ${limit} W, NOT ${GPU_POWER_CAP} W — cap did not apply ***"
      mismatch=1
    fi
  done < <(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits 2>/dev/null)
  [ "$mismatch" = "1" ] && log "*** cap mismatch is non-fatal (no sudo); continuing with less margin ***"
  log "CE baseline:"
  ras-mc-ctl --summary 2>/dev/null | grep -E "channel" | sed 's/^/[glm53kt]   /' || true
  check_capture_alive
}

drop_caches(){
  log "dropping page caches before 305 GB load"
  sync && (echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || log "warn: drop_caches needs root (continuing)")
}

launch(){
  local ctx="$1"
  local api_args=()
  [ -n "$API_KEY" ] && api_args+=(--api-key "$API_KEY")

  local publish="$PORT:30000"
  [ -n "$BIND_HOST" ] && publish="$BIND_HOST:$PORT:30000"

  local cg_args=()
  if [ -n "$CUDAGRAPH_BS" ]; then
    for bs in $CUDAGRAPH_BS; do cg_args+=(--cuda-graph-bs "$bs"); done
  fi

  docker rm -f "$NAME" >/dev/null 2>&1
  log "launching $NAME: ctx=$ctx gpu=$GPU method=$KT_METHOD cpuinfer=$KT_CPUINFER threadpool=$KT_THREADPOOL gpu_experts=$KT_NUM_GPU_EXPERTS memfrac=$MEM_FRACTION seqs=$MAX_RUNNING_REQUESTS"
  nohup docker run -d --name "$NAME" \
    --init --gpus "device=$GPU" --shm-size 16g --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$MODEL":/model:ro \
    -v "$CPU_WEIGHTS":/cpu_weights:ro \
    -p "$publish" \
    -e OMP_NUM_THREADS="$KT_CPUINFER" \
    "$IMAGE" \
    python3 -m sglang.launch_server \
    --model-path /model \
    --kt-weight-path /cpu_weights \
    --served-model-name "$SERVED_MODEL" \
    --host 0.0.0.0 --port 30000 \
    --tp-size 1 \
    --context-length "$ctx" \
    --mem-fraction-static "$MEM_FRACTION" \
    --chunked-prefill-size "$CHUNKED_PREFILL" \
    --kt-method "$KT_METHOD" \
    --kt-cpuinfer "$KT_CPUINFER" \
    --kt-threadpool-count "$KT_THREADPOOL" \
    --kt-num-gpu-experts "$KT_NUM_GPU_EXPERTS" \
    --kt-gpu-prefill-token-threshold "$KT_GPU_PREFILL_THRESHOLD" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --trust-remote-code \
    "${api_args[@]}" \
    "${cg_args[@]}" \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    > "$SERVE_LOG" 2>&1 &
  log "launched (305 GB ZFS load into RAM: expect 30–60 min)"
  log "  progress:  docker logs -f $NAME"
  log "  wait:      bash $0 --wait"
  log "  launch err: $SERVE_LOG (holds container ID only if run started)"
  log "  REMINDER: log this attempt to RUN-LOG.md before next step (P9)"
}

wait_ready(){
  local timeout="${WAIT_TIMEOUT:-3600}" t0 elapsed st
  t0=$(date +%s)
  log "waiting for $NAME (timeout ${timeout}s); ^C is safe, it does not stop the container"
  while :; do
    elapsed=$(( $(date +%s) - t0 ))
    st="$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null)"
    if [ -z "$st" ]; then log "container $NAME is gone"; return 1; fi
    if [ "$st" != "running" ]; then
      log "FAILED after ${elapsed}s — container $st (exit $(docker inspect "$NAME" --format '{{.State.ExitCode}}' 2>/dev/null))"
      docker logs "$NAME" 2>&1 | grep -aE "RuntimeError|ValueError|AssertionError|CUDA error|No valid|not supported|Error" \
        | grep -avE "otel.py|please check the stack trace" | tail -5 | sed 's/^/[glm53kt]   /'
      log "next: snapshot before cleanup -> bash $REPO_DIR/recipe/run-log.sh --container $NAME --stage <stage>"
      return 1
    fi
    if http_probe; then
      log "READY after ${elapsed}s — $(probe_url)"; return 0
    fi
    if [ "$elapsed" -ge "$timeout" ]; then log "timed out after ${elapsed}s (still running; keep watching docker logs -f $NAME)"; return 2; fi
    sleep 15
  done
}

status(){
  docker ps -a --filter name="$NAME" --format 'serve: {{.Names}} {{.Status}}'
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
  free -g | awk 'NR==2{print "  ram used/avail GB: "$3"/"$7}'
  http_probe \
    && echo "  http: READY on :$PORT" || echo "  http: not serving on :$PORT"
  echo "  --- docker logs (last 8) ---"
  docker logs "$NAME" 2>&1 | tail -8 | sed 's/^/  /'
}

case "${1:---help}" in
  --check)  check_runtime; check_model_present config ;;
  --smoke)  check_image; check_model_present all; require_gpus_free; arm_safety; check_numa; drop_caches; launch "$SMOKE_CTX" ;;
  --dummy)  log "KT has no --load-format dummy; --dummy maps to --smoke (real weights, ctx $SMOKE_CTX)"; check_image; check_model_present all; require_gpus_free; arm_safety; check_numa; drop_caches; launch "$SMOKE_CTX" ;;
  --serve)  check_image; check_model_present all; require_gpus_free; arm_safety; check_numa; drop_caches; launch "$CTX" ;;
  --wait)   wait_ready ;;
  --logs)   docker logs -f "$NAME" ;;
  --stop)   docker rm -f "$NAME" >/dev/null 2>&1 && log "stopped $NAME" || log "no $NAME running" ;;
  --status) status ;;
  *) cat <<EOF
usage: $0 --check | --smoke | --serve | --wait | --logs | --stop | --status
  --check   CPU-only: image, kt-kernel, sglang-kt, model dir, RAM, ISA audit
  --smoke   Stage 1: real weights, ctx $SMOKE_CTX, GPU$GPU (first high-risk event)
  --serve   Stage 2: real weights, ctx $CTX (tutorial-validated)
  --wait    poll until bind/die/timeout (WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}s)
  --logs    docker logs -f $NAME
env: MODEL CPU_WEIGHTS IMAGE NAME PORT GPU KT_METHOD KT_CPUINFER KT_THREADPOOL
      KT_NUM_GPU_EXPERTS KT_GPU_PREFILL_THRESHOLD MEM_FRACTION CHUNKED_PREFILL
      SMOKE_CTX CTX CUDAGRAPH_BS MAX_RUNNING_REQUESTS GPU_POWER_CAP
      BIND_HOST API_KEY SERVE_LOG WAIT_TIMEOUT
see recipe/GLM-53-FLASH-KT-RECIPE.md for staged plan and risk gates
EOF
  ;;
esac
