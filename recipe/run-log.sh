#!/bin/bash
# run-log.sh — snapshot one serving attempt into a pre-filled RUN-LOG.md entry.
#
# Why: attempts that don't crash the host leave no durable trace (raw logs in /var/tmp are
# overwritten by the next launch). This captures the machine-readable half of an entry so the
# only thing left to write by hand is the judgement half: Outcome, Lesson, Next.
#
# Usage:
#   bash recipe/run-log.sh --container glm53-nvfp4-serve --stage dummy
#   bash recipe/run-log.sh --container qwen38-flash-fp8-serve --stage tune --note "SPEC_TOKENS=4"
#   bash recipe/run-log.sh --stage check --model nvidia/GLM-5.3-Flash-NVFP4 --note "arch assertions"
#
# Prints a markdown entry to stdout. Review it, fill the TODO lines, prepend to RUN-LOG.md.
# New launchers should call this on exit; see AGENTS.md §7 (launcher contract) and §9.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER=""; STAGE="?"; MODEL=""; NOTE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --container) CONTAINER="${2:-}"; shift 2 ;;
    --stage)     STAGE="${2:-}";     shift 2 ;;
    --model)     MODEL="${2:-}";     shift 2 ;;
    --note)      NOTE="${2:-}";      shift 2 ;;
    -h|--help)   sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

q(){ "$@" 2>/dev/null || true; }   # never let a probe abort the snapshot

# --- container facts -------------------------------------------------------
EXIT_CODE="n/a"; OOM="n/a"; STARTED=""; FINISHED=""; DURATION="n/a"; ARGS=""; IMAGE=""
if [ -n "$CONTAINER" ] && docker inspect "$CONTAINER" >/dev/null 2>&1; then
  EXIT_CODE="$(q docker inspect "$CONTAINER" --format '{{.State.ExitCode}}')"
  OOM="$(q docker inspect "$CONTAINER" --format '{{.State.OOMKilled}}')"
  STARTED="$(q docker inspect "$CONTAINER" --format '{{.State.StartedAt}}')"
  FINISHED="$(q docker inspect "$CONTAINER" --format '{{.State.FinishedAt}}')"
  IMAGE="$(q docker inspect "$CONTAINER" --format '{{.Config.Image}}')"
  ARGS="$(q docker inspect "$CONTAINER" --format '{{join .Args " "}}')"
  if [ -n "$STARTED" ] && [ -n "$FINISHED" ]; then
    s="$(q date -d "$STARTED" +%s)"; f="$(q date -d "$FINISHED" +%s)"
    if [ -n "$s" ] && [ -n "$f" ] && [ "$f" -gt "$s" ] 2>/dev/null; then
      DURATION="$(( (f-s)/60 ))m$(( (f-s)%60 ))s"
    else
      DURATION="still running"
    fi
  fi
  [ -z "$MODEL" ] && MODEL="$(echo "$ARGS" | grep -oE '\-\-served-model-name [^ ]+' | awk '{print $2}')"
fi

# --- host state ------------------------------------------------------------
DRV_LOADED="$(q grep -oE '[0-9]{3}\.[0-9]+(\.[0-9]+)?' /proc/driver/nvidia/version | head -1)"
DRV_PKG="$(q modinfo nvidia | awk '/^version:/{print $2}')"
DRV_STATE="$DRV_LOADED (pkg $DRV_PKG)"
[ -n "$DRV_LOADED" ] && [ "$DRV_LOADED" = "$DRV_PKG" ] && DRV_STATE="$DRV_LOADED matched"
RAM="$(q free -g | awk 'NR==2{print $2" GB total, "$7" GB avail"}')"
NUMA="$(q numactl -H | awk '/node [0-9]+ size:/{printf "%d/", $4/1024}' | sed 's:/$: GB:')"
BOOT="$(q uptime -s)"
GITREV="$(q git -C "$REPO_DIR" rev-parse --short HEAD)"
q git -C "$REPO_DIR" diff --quiet || GITREV="$GITREV+dirty"
GPUS="$(q nvidia-smi --query-gpu=index,memory.used,power.limit --format=csv,noheader | tr '\n' ';')"

# --- capture liveness (P4: verify the effect, not the invocation) ----------
CAPTURE="not running"
if q docker ps --format '{{.Names}}' | grep -q powertrip-capture; then
  latest="$(q ls -t /buffer/powertrip/klog-*.log | head -1)"
  if [ -n "$latest" ]; then
    age=$(( $(date +%s) - $(q stat -c %Y "$latest" || echo 0) ))
    CAPTURE="armed, klog last wrote ${age}s ago"
    [ "$age" -gt 60 ] && CAPTURE="$CAPTURE — STALE, crash window may be blind"
  else
    CAPTURE="container up, no klog found"
  fi
fi

# --- the failure signature -------------------------------------------------
ERRLINE="(no container logs)"
if [ -n "$CONTAINER" ]; then
  ERRLINE="$(q docker logs "$CONTAINER" 2>&1 \
    | grep -aE "RuntimeError|ValueError|AssertionError|CUDA error|not supported|No valid|OutOfMemory|Killed" \
    | grep -avE "otel.py|sync_wrapper|please check the stack trace" \
    | tail -3)"
  [ -z "$ERRLINE" ] && ERRLINE="$(q docker logs "$CONTAINER" 2>&1 | tail -2)"
fi

# --- emit ------------------------------------------------------------------
cat <<EOF
### R-NNN · $(date '+%Y-%m-%d %H:%M') local — ${MODEL:-<model>} · stage \`${STAGE}\`

| | |
|---|---|
| **Outcome** | TODO: passed / failed-at-<stage> / trip / OOM |
| **Duration** | ${DURATION} (exit ${EXIT_CODE}, OOMKilled=${OOM}) |
| **Image** | ${IMAGE:-n/a} |
| **Host** | driver ${DRV_STATE}; RAM ${RAM}; NUMA nodes ${NUMA}; booted ${BOOT}; repo ${GITREV} |
| **GPUs** | ${GPUS:-n/a} |
| **Capture** | ${CAPTURE} |
| **Changed vs. last attempt** | TODO: the ONE variable (P2) |

**Config**
\`\`\`
${ARGS:-<command or flags>}
\`\`\`

**Signature**
\`\`\`
${ERRLINE}
\`\`\`

**Read:** TODO — how far did it get? Name the last stage reached (NCCL init / backend select /
weight load / KV init / graph capture / serving) and what that rules in or out.

**Lesson:** TODO — one line, transferable.

**Next:** TODO — the single next variable to change.
${NOTE:+
**Note:** ${NOTE}}
EOF
