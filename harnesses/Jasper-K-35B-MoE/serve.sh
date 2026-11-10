#!/usr/bin/env bash
# Start a local vLLM completion service and wait for readiness.
# harness.py passes checkpoint, device, port, and capacity settings from harness.toml.
# For a standalone invocation, provide MODEL and the required serving settings:
#   MODEL=/path/to/checkpoint TP=2 PORT=8111 JASPER_K_GPUS=0,1 ./serve.sh
#   PORT=8111 ./serve.sh --stop
# An existing endpoint is reused only when it serves the expected checkpoint.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${PORT:-8111}"
PIDFILE="$HERE/vllm.pid"
LOG="${LOG:-$HERE/vllm.log}"

if [ "${1:-}" = "--stop" ]; then
  [ -f "$PIDFILE" ] && kill -- -"$(cat "$PIDFILE")" 2>/dev/null && echo "stopped $(cat "$PIDFILE")"
  rm -f "$PIDFILE"; exit 0
fi

# Confirm that an existing service uses the requested checkpoint before reuse.
serving_model() {
  curl -sf --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys
try: d = json.load(sys.stdin)["data"][0]
except Exception: sys.exit(1)
print(d.get("root") or d.get("id") or "")' 2>/dev/null
}

ready() {
  local got
  got="$(serving_model)" || return 1
  [ -n "$got" ] || return 1
  # No MODEL to compare against (manual two-step run): anything answering will do.
  [ -z "${MODEL:-}" ] && return 0
  [ "$got" = "$MODEL" ]
}

MODEL="${MODEL:-}"

if ready; then
  echo "[serve] http://127.0.0.1:$PORT/v1 already serving ${MODEL:-that model}; not starting another"
  exit 0
fi
OTHER="$(serving_model)" || true
if [ -n "$OTHER" ]; then
  echo "[serve] port $PORT is serving $OTHER, not ${MODEL:-<unset>}."
  echo "[serve] Choose an unused port or stop the existing service before continuing."
  exit 1
fi
[ -n "$MODEL" ] || { echo "[serve] no MODEL= (model directory) and nothing already serving on $PORT"; exit 1; }
TP="${TP:-1}"
UTIL="${UTIL:-0.85}"
NAME="${NAME:-Jasper-K-35B-MoE}"
GPUS="${GPUS:-${JASPER_K_GPUS:-}}"
WAIT="${WAIT:-900}"

[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"

# Derive serving options and validate tensor parallelism against the model configuration.
EXTRA="$(python3 "$HERE/vllm_args.py" "$MODEL" --tp "$TP" --window "${WINDOW:-16384}")" || {
  echo "[serve] vllm_args.py rejected these settings; not starting"; exit 1; }
[ -n "$EXTRA" ] && echo "[serve] extra flags: $EXTRA"

echo "[serve] starting vLLM: $MODEL  port=$PORT tp=$TP gpus=${GPUS:-<inherited>} -> $LOG"
# shellcheck disable=SC2086  EXTRA is our own generated flag string; split on whitespace
setsid vllm serve "$MODEL" \
  --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" --gpu-memory-utilization "$UTIL" \
  --dtype bfloat16 --disable-log-stats \
  $(eval echo $EXTRA) \
  --trust-remote-code > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

# Wait until model loading and service initialization complete.
t0=$SECONDS
while [ $((SECONDS - t0)) -lt "$WAIT" ]; do
  if ready; then echo "[serve] ready in $((SECONDS - t0))s"; exit 0; fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "[serve] the vLLM process is gone; tail of $LOG:"; tail -20 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[serve] still not ready after ${WAIT}s; see $LOG"; exit 1
