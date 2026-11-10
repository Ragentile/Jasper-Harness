#!/usr/bin/env bash
# Generate candidates for the example tasks and re-evaluate them with your backend.
# Requires a configured model service and a working evaluator adapter.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES=1; TIMEOUT=1800; PROFILE=""; GPUS="${JASPER_K_GPUS:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --samples) SAMPLES="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --eval-profiling) PROFILE="$2"; shift 2;;
    -h|--help)
      cat <<'HELP'
Usage: ./selftest.sh [--gpus 0,1] [--samples 1] [--timeout 1800]
                     [--eval-profiling true|false]

Configure the model in harness.toml and set JASPER_K_EVALUATOR_ENDPOINT.
This command recreates selftest_out, runs the example tasks, and re-evaluates
generated candidates. Results are local integration checks, not benchmark claims.
HELP
      exit 0;;
    *) echo "Unknown option: $1"; exit 2;;
  esac
done
PROBLEMS="$HERE/sample_problems"
OUT="$HERE/selftest_out"
[ -d "$PROBLEMS" ] || { echo "Missing example task directory: $PROBLEMS"; exit 1; }
LEVELS="$(ls -d "$PROBLEMS"/level[0-9]* 2>/dev/null | sed 's#.*/level##' | sort -n | paste -sd, -)"
[ -n "$LEVELS" ] || { echo "No level directories found in $PROBLEMS"; exit 1; }
EP="${JASPER_K_EVALUATOR_ENDPOINT:-}"
[ -n "$EP" ] || { echo "Set JASPER_K_EVALUATOR_ENDPOINT to your evaluator adapter"; exit 1; }
curl -sf --noproxy '*' --max-time 8 "$EP/health" >/dev/null 2>&1 \
  || { echo "Cannot reach the evaluator adapter at $EP"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPUS" JASPER_K_GPUS="$GPUS"
export JASPER_K_BACKEND=cuda JASPER_K_EVALUATOR_ENDPOINT="$EP"
export JASPER_K_EVAL_TIMEOUT="${JASPER_K_EVAL_TIMEOUT:-240}"
export JASPER_K_EVAL_CLIENT_MARGIN="${JASPER_K_EVAL_CLIENT_MARGIN:-1800}"
echo "Jasper-K self-test: $SAMPLES trajectories per task; GPUs $GPUS; timeout ${TIMEOUT}s"
cd "$HERE"
if [ -n "$PROFILE" ]; then set -- --eval-profiling "$PROFILE"; else set --; fi
python3 "$HERE/harness.py" --problems "$PROBLEMS" --out "$OUT" --backend cuda \
  --timeout "$TIMEOUT" --levels "$LEVELS" --samples "$SAMPLES" "$@"
RC=$?
echo "Re-evaluating saved candidates (harness exit $RC)"
exec python3 "$HERE/selftest_judge.py" "$PROBLEMS" "$OUT" "$SAMPLES"
