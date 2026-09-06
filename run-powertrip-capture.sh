#!/bin/bash
# run-powertrip-capture.sh
# Standard launcher for the crash-capture telemetry container.
# - Runs the capture container in a PRIVILEGED container so it can read
#   RAPL package power, dmesg, and EDAC, plus host GPU via mounted nvidia-smi.
# - Writes directly to /buffer/powertrip (/capture inside) which SURVIVES the
#   power-trip reset (unlike /tmp).
# - Restart policy: unless-stopped (keeps sampling across accidental stops).
#
# Usage:
#   bash run-powertrip-capture.sh start   # launch capture (name: powertrip-capture)
#   bash run-powertrip-capture.sh stop
#   bash run-powertrip-capture.sh status
#   bash run-powertrip-capture.sh restart

set -u
NAME=powertrip-capture
INTERVAL="${INTERVAL:-1}"
OUTDIR_HOST=/buffer/powertrip

NV="-v"
nvidia_mounts=(
  "$NV" "/usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro"
  "$NV" "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro"
  "$NV" "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02:ro"
  "$NV" "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro"
  "$NV" "/usr/lib/x86_64-linux-gnu/libcuda.so.580.173.02:/usr/lib/x86_64-linux-gnu/libcuda.so.580.173.02:ro"
)

case "${1:-start}" in
  start)
    docker rm -f "$NAME" >/dev/null 2>&1
    docker run -d --name "$NAME" \
      --privileged --pid=host --net=host \
      --restart unless-stopped \
      -e OUTDIR=/capture -e INTERVAL="$INTERVAL" \
      "${nvidia_mounts[@]}" \
      -v /var/lib/rasdaemon:/var/lib/rasdaemon:ro \
      -v /dev:/dev:rw \
      -v "$OUTDIR_HOST:/capture" \
      powertrip-capture:local \
      /usr/local/bin/powertrip-capture
    echo "started $NAME (interval=${INTERVAL}s, out=/buffer/powertrip)"
    ;;
  stop)
    docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME"
    ;;
  restart)
    docker rm -f "$NAME" >/dev/null 2>&1; sleep 1; exec bash "$0" start
    ;;
  status)
    docker ps -a --filter name="$NAME" --format '{{.Names}} {{.Status}}'
    ;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 1;;
esac
