#!/bin/bash
# serve-glm-53-flash-nvfp4.sh — launcher for nvidia/GLM-5.3-Flash-NVFP4 on this box (2×72 GB, TP2).
# ModelOpt NVFP4 MoE (320B total / 18B active), 190.4 GiB on disk, native 1M ctx.
# Recipe + rationale: recipe/GLM-53-FLASH-NVFP4-RECIPE.md (read §3 gates before --serve).
#
# 2026-09-17 sm_120 NoPE port (R-014/R-015 block lifted in software): default IMAGE is now the
# pensive-derived build `pensive/glm53-flash:nope-sm120-617d0cc` = official vendor-fork base
# + glm53_sparse_mla (Apache-2.0, Libertai/glm53-flash-vllm-gb10 @ 617d0cc; hand-written NoPE
# sparse-MLA CUDA kernel installed as vllm.general_plugins, no vLLM file patched; provenance
# + port notes in recipe §0 and §5). Dockerfile: recipe/Dockerfile.glm53-flash-nope-sm120.
# The plugin overrides the FLASHINFER_MLA_SPARSE_SM120 slot only when
# VLLM_GLM53_CUDA_SPARSE_MLA=1 (this launcher sets it); kernel is FIXED to 32 heads/rank =
# TP2, bf16 KV per shipped backend.py, CUDA graphs per shipped code = NEVER (their README's
# fp8-KV/UNIFORM_BATCH claims are unconfirmed — verify before re-enabling either).
# MoE input-scale fix (VLLM_GLM53_MOE_INPUT_SCALE) stays UNSET: nvidia's checkpoint is
# fully-serialised (verified: 36,297 inline input_scale tensors; base image's loader maps them,
# routed_experts.py ModelOpt NVFP4 branch). If first serve shows repeated-token garbage, THAT
# is the fault-2 signature — but expect it impossible here; suspect something else first.
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
#   GPU_MEM_UTIL=0.90  GPU_POWER_CAP=250  ENFORCE_EAGER=1 (default since the port: shipped plugin
#                         declares AttentionCGSupport.NEVER; try 0 only after measuring what vLLM does)
#   KV_CACHE_DTYPE=bfloat16 (NOT auto — see below. The ported kernel supports bf16 only in the
#                         shipped build; fp8 unverified — their README claims it, code denies it.)
#   SPARSE_MLA=1         the plugin gate (VLLM_GLM53_CUDA_SPARSE_MLA). 0 = A/B back to stock
#                         sm_120 selection — which fails per R-014/R-015; keep 1.
#   MOE_INPUT_SCALE=""   VLLM_GLM53_MOE_INPUT_SCALE; leave unset for this checkpoint (see header).
#                         Never 1.0 — upstream retracted it (632x median; block-scale underflow).
#   SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":2}'   # stage-3 only
#   EP=1  adds --enable-expert-parallel --enable-ep-weight-filter (stage-3; untested at TP2)
#   NCCL_CUMEM_HOST_ENABLE=auto|0|1   auto probes per-GPU NUMA node memory (memory-less node => force 0)
#   SERVE_LOG=/var/tmp/serve-glm53-nvfp4.log
#   BIND_HOST=""    where to PUBLISH the port on the host: "" = all interfaces (legacy default);
#                   100.70.5.43 = tailscale0 only; 127.0.0.1 = localhost only. The engine still binds
#                   0.0.0.0 INSIDE the container (docker-proxy NAT requires it) — never move --host.
#   API_KEY=""      "" = unauthenticated (legacy). If set, passed to vLLM as --api-key; clients need
#                   "Authorization: Bearer $API_KEY". 2026-09-16: clients are remote on the tailnet,
#                   not just localhost — set this before exposing the port. Visible in docker inspect/logs.
# Hard gates (channel G / MM4 history — see power-trip-instances.md):
#   - refuses while qwen38-flash-serve or glm53-flash-serve holds VRAM
#   - arms powertrip-capture + edac-ce-watch, GPU power cap 250 W (verified, not just issued)
#   - serialized load (--max-parallel-loading-workers 1), spawn, NCCL_P2P_DISABLE=1
#   - NUMA/NCCL memory-less-node guard (ported from the qwen38 launcher)
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/trunk/ai/huggingface/models/nvidia/GLM-5.3-Flash-NVFP4}"
IMAGE="${IMAGE:-pensive/glm53-flash:nope-sm120-617d0cc}"   # fallback: vllm/vllm-openai:glm53-flash (stock; blocked per R-014/R-015)
NAME="${NAME:-glm53-nvfp4-serve}"
PORT="${PORT:-8092}"
SERVED_MODEL="${SERVED_MODEL:-nvidia/GLM-5.3-Flash-NVFP4}"
GPU_POWER_CAP="${GPU_POWER_CAP:-250}"
TP="${TP:-2}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-}"              # per worker; auto: TP2->32, TP1->80
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# KV DTYPE + BLOCK SIZE (history: R-014/R-015 — stock sm_120 selection had NO NoPE sparse-MLA
# path: fp8 died in concat_and_cache_mla "pe_dim must be 64"; auto died at backend
# construction). The ported glm53_sparse_mla plugin (header) lifts that, with two hard
# constraints now measured on this box (R-017/R-018):
#   1. KV dtype: the shipped backend advertises only auto|bfloat16, and `auto` does NOT
#      resolve to the model dtype here — it canonicalized to fp8_e4m3, which the selector
#      rejected (R-017). fp8 KV is impossible until the .so's fp8 path is verified (their
#      README claims 6/6 verified; the shipped backend.py denies it). -> bfloat16.
#   2. Block size: the kpool indexer's DeepGEMM logits kernel (sm_120 non-fp4) asserts the
#      STORAGE block == 64, and storage = block_size // index_kpool(4) -> block_size 256.
#      The vLLM default (16) inflates to 2176 via the hybrid "attention page >= mamba page"
#      rule, and 2176//4 = 544 -> assert (R-018). Their sm_120 deployment pins exactly this:
#      "Glm5NextIndexerCache asserts block_size % (index_kpool*32)==0 ... 256 -> storage 64,
#      satisfying both. Do not 'fix' this to 64."
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"        # never auto (see 1); fp8 blocked until .so verified
BLOCK_SIZE="${BLOCK_SIZE:-256}"                     # see (2); 128 breaks the kpool guard, 512+ breaks DeepGEMM
SPARSE_MLA="${SPARSE_MLA:-1}"                        # plugin gate VLLM_GLM53_CUDA_SPARSE_MLA (1=ported NoPE kernel)
MOE_INPUT_SCALE="${MOE_INPUT_SCALE:-}"               # VLLM_GLM53_MOE_INPUT_SCALE; unset = off (correct here)
ENGINE_READY_TIMEOUT_S="${ENGINE_READY_TIMEOUT_S:-3600}"  # engine-ready timeout; default 600s is tight for a 190 GiB ZFS load (adopted from the GB200 recipe)
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"                 # shipped plugin declares CG support NEVER -> eager default; test 0 only after measuring
SPEC_CONFIG="${SPEC_CONFIG:-}"                    # e.g. '{"method":"mtp","num_speculative_tokens":2}'
EP="${EP:-0}"                                     # 1 = --enable-expert-parallel --enable-ep-weight-filter
NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-auto}"
SERVE_LOG="${SERVE_LOG:-/var/tmp/serve-glm53-nvfp4.log}"
BIND_HOST="${BIND_HOST:-}"                        # "" = publish on all interfaces; IP = tailscale0/LAN bind only
API_KEY="${API_KEY:-}"                            # "" = no auth; else vLLM --api-key (Bearer token)
OTHER_SERVES="qwen38-flash-serve glm53-flash-serve"
EXPECTED_SHARDS=33

log(){ echo "[glm53nv] $*"; }
die(){ log "ERROR: $*"; exit 1; }

# ---- remote-access readiness (2026-09-16: clients are on the tailnet, not just localhost) ----
# Derived from the LIVE container so a fresh --wait/--status shell without BIND_HOST/API_KEY
# still probes the right address with the right token.
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
  [ "$IMAGE" = "${IMAGE#pensive/}" ] && { log "image is not a pensive build; skipping plugin assertions"; return 0; }
  log "checking glm53_sparse_mla plugin inside $IMAGE (CPU-only)"
  docker run --rm --entrypoint python3 "$IMAGE" -u -c "
import importlib.metadata as md, os
eps = {e.name: e.value for e in md.entry_points(group='vllm.general_plugins')}
assert 'glm53_sparse_mla' in eps, 'sparse-MLA plugin entry point missing (image lacks the port)'
import glm53_sparse_mla, glob
p = os.path.dirname(glm53_sparse_mla.__file__)
assert glob.glob(p + '/_C*.so'), 'AOT kernel .so missing from ' + p
from vllm.v1.attention.backends.registry import AttentionBackendEnum as E
assert E.FLASHINFER_MLA_SPARSE_SM120.get_path().startswith('vllm.'), 'plugin must be inert without the env gate'
os.environ['VLLM_GLM53_CUDA_SPARSE_MLA'] = '1'
import glm53_sparse_mla.backend as B
assert B.register() is True and E.FLASHINFER_MLA_SPARSE_SM120.get_path().startswith('glm53_sparse_mla'), 'override failed'
print('OK: glm53_sparse_mla installed, inert-until-gated, override verified')
" || die "plugin check FAILED: rebuild from recipe/Dockerfile.glm53-flash-nope-sm120"
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
  # Reasoning parser is deepseek_r1, NOT glm45, despite the model card's SGLang recipe naming it.
  # LibertAI trap (sm_120-verified): the chat template puts the OPENING think-open token as the
  # last PROMPT token, so output only ever carries the closer; glm45's state machine never opens
  # a span and silently returns empty content with nonzero completion_tokens. deepseek_r1
  # terminates on the bare closer. Provenance: recipe §0 port note (LibertAI traps).
  local api_args=()
  [ -n "$API_KEY" ] && api_args+=(--api-key "$API_KEY")
  local publish="$PORT:8000"
  [ -n "$BIND_HOST" ] && publish="$BIND_HOST:$PORT:8000"
  local cumem_env=()
  if [ -n "${RESOLVED_CUMEM_HOST_ENABLE:-}" ]; then
    cumem_env+=(-e "NCCL_CUMEM_HOST_ENABLE=${RESOLVED_CUMEM_HOST_ENABLE}")
  fi
  local plugin_env=(-e "VLLM_GLM53_CUDA_SPARSE_MLA=$SPARSE_MLA")
  [ -n "$MOE_INPUT_SCALE" ] && plugin_env+=(-e "VLLM_GLM53_MOE_INPUT_SCALE=$MOE_INPUT_SCALE") \
    && log "WARNING: MOE_INPUT_SCALE=$MOE_INPUT_SCALE set — only correct for weight-only NVFP4 checkpoints; this one is fully serialised"
  docker rm -f "$NAME" >/dev/null 2>&1
  log "launching $NAME mode=$mode TP=$TP offload=${CPU_OFFLOAD_GB}GB/wkr ctx=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS mem_util=$GPU_MEM_UTIL kv=$KV_CACHE_DTYPE block=$BLOCK_SIZE eager=$ENFORCE_EAGER sparse_mla=$SPARSE_MLA NCCL_CUMEM_HOST_ENABLE=${RESOLVED_CUMEM_HOST_ENABLE:-<default>} publish=${BIND_HOST:-<all-ifaces>}:${PORT} auth=$([ -n "$API_KEY" ] && echo on || echo off)"
  nohup docker run -d --name "$NAME" \
    --gpus all --shm-size 16g --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$MODEL":/model:ro -p "$publish" \
    -e NCCL_P2P_DISABLE=1 -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_ENGINE_READY_TIMEOUT_S="$ENGINE_READY_TIMEOUT_S" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${plugin_env[@]}" \
    "${cumem_env[@]}" \
    "$IMAGE" \
    /model --tensor-parallel-size "$TP" --quantization modelopt \
    --served-model-name "$SERVED_MODEL" \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --kv-cache-dtype "$KV_CACHE_DTYPE" --block-size "$BLOCK_SIZE" \
    --disable-custom-all-reduce --max-parallel-loading-workers 1 \
    --host 0.0.0.0 \
    "${api_args[@]}" \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser deepseek_r1 \
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
       GPU_POWER_CAP KV_CACHE_DTYPE BLOCK_SIZE ENGINE_READY_TIMEOUT_S ENFORCE_EAGER SPEC_CONFIG EP
       SPARSE_MLA MOE_INPUT_SCALE NCCL_CUMEM_HOST_ENABLE SERVE_LOG WAIT_TIMEOUT BIND_HOST API_KEY
see recipe/GLM-53-FLASH-NVFP4-RECIPE.md for the staged plan and risk gates
EOF
  ;;
esac
