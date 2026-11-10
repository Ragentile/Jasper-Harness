# Connecting your evaluator

**Provide an evaluation backend for kernel compilation, execution, correctness checks, performance measurement, and optional profiling. The repository includes an adapter and implementation template.**

Provide your own evaluation implementation. The `evaluator_adapter.py` module translates between HTTP requests and a Python function; the evaluation operations are implemented by your backend. `example_evaluator.py` is a template that raises an explicit error until implemented.

## Setup

See the [evaluator template and feedback walkthrough](examples/README.md) for copyable requests, complete response examples, a synthetic HTTP demo, and rendered next-turn feedback.

1. Copy `example_evaluator.py` to `my_evaluator.py` and implement `run_backend(request)` using your evaluator; the template provides `evaluate(request)`.
2. Start the adapter:

```bash
python3 evaluator_adapter.py --evaluator my_evaluator:evaluate --host 127.0.0.1 --port 9000
```

In another terminal, point the Jasper-K harness to it:

```bash
export JASPER_K_EVALUATOR_ENDPOINT=http://127.0.0.1:9000
python3 harness.py --problems /path/to/problems --out ./run_out --samples 16 --timeout 7200
```

You can also implement the HTTP protocol below in your own service without using the adapter. Set `JASPER_K_EVALUATOR_ENDPOINT` to your evaluator service URL.

## Optional harness settings

The main README command is an example: choose your own device IDs, model-serving configuration, task directory, and evaluator URL. `JASPER_K_GPUS` selects devices for model serving. `JASPER_K_EVALUATOR_ENDPOINT` specifies your evaluator service URL.

`JASPER_K_EVAL_QUOTA` is an optional per-task budget across trajectories. If unset, the harness uses `loop.max_rounds`. The 27B and 35B harnesses use an explicit quota to cap turns; the 32B harness honors it only when `loop.honor_eval_quota=true`. It is not needed to connect an evaluator.

`JASPER_K_ALLOW_LIBRARY_CALLS` controls the library-call rule in prompts and feedback: unset or `1` allows library calls alongside a real custom kernel; `0` requires custom kernels for all computation. Match this choice to your evaluator's rules. This variable does not implement or enable backend rule checking.

## Public configuration interface

| Setting | Purpose |
|---|---|
| `JASPER_K_GPUS` | Comma-separated device IDs for local model serving. |
| `JASPER_K_PORT` | Optional override for the primary model-serving port. |
| `JASPER_K_EVALUATOR_ENDPOINT` | URL of your evaluator adapter. |
| `JASPER_K_EVAL_QUOTA` | Optional task budget across trajectories; see model-specific handling above. |
| `JASPER_K_ALLOW_LIBRARY_CALLS` | Library-call policy used in prompts and feedback. |

For 27B and 35B, `JASPER_K_EVAL_TIMEOUT`, `JASPER_K_EVAL_CLIENT_MARGIN`, and
`JASPER_K_EVAL_MODE` configure evaluator requests. The 32B optimization loop uses
the corresponding fields in `[feedback]` in `harness.toml`. Evaluation modes are
optional labels defined by your backend; no mode is selected by default.
`selftest_judge.py` reads the timeout and client-margin environment settings for
the final recheck. `GENERATION_TIMEOUT_SECONDS` optionally bounds generation HTTP
requests, including time spent waiting in the model server's queue.

## Request

Send `POST /evaluate` with the following JSON body. `timeout` is measured in seconds. The adapter passes the source strings to your Python function without invoking an outer shell.

```json
{
  "ref_code": "Complete Python source for the reference model",
  "kernel_code": "Complete Python source for the candidate ModelNew",
  "timeout": 240,
  "cfg": {
    "backend": "cuda",
    "num_correct_trials": 5,
    "measure_performance": true,
    "num_perf_trials": 100,
    "enable_profiling": true,
    "return_profiling_events": true
  }
}
```

The 32B configuration defaults to a 300-second evaluation timeout. Your evaluation backend defines the test inputs, numerical tolerances, and timing procedure. Record these settings when reporting results.

If a request includes `cfg.eval_mode`, it is only a backend-specific configuration label passed through by the harness. The adapter does not interpret it or implement an evaluation algorithm. Your evaluator can map this label to its own settings or ignore it if it has no evaluation modes.

The adapter may call `evaluate` concurrently. Your backend is responsible for device scheduling, process isolation, and enforcing evaluation timeouts.

## Response

Return `HTTP 200` with the following structure. The numbers below are illustrative; populate them with actual measurements.

```json
{
  "compiled": true,
  "correctness": true,
  "cheated": false,
  "infra_error": false,
  "tests_passed": 5,
  "tests_total": 5,
  "runtime_ms": 0.5,
  "baseline_runtime_ms": 1.0,
  "speedup": 2.0,
  "error_message": "",
  "metadata": {}
}
```

| Field | Meaning |
|---|---|
| `compiled` | Required boolean: whether the candidate loaded or compiled successfully. |
| `correctness` | Required boolean: whether the candidate passed actual correctness checks. |
| `speedup` | A correct candidate must have a positive, finite value equal to reference time divided by candidate time. |
| `runtime_ms` / `baseline_runtime_ms` | Measured candidate / reference execution time in milliseconds. |
| `tests_passed` / `tests_total` | Correctness test counts. |
| `error_message` | Specific compilation, runtime, incorrect-output, or timeout feedback. |
| `cheated` | Optional: whether the candidate violated rules enforced by your evaluator. |
| `infra_error` | Optional: whether the service or device failed independently of the candidate code. |
| `metadata` | Optional extended data, such as profiler summaries. |

Candidate failures, including compilation errors and incorrect outputs, should return HTTP 200 with `correctness=false`, `speedup=null`, and an error message so the harness can use the feedback in the next turn. Service failures may return HTTP 5xx or a valid response with `infra_error=true`; these trigger infrastructure-failure handling. Candidate timeouts should return a failed result whose error message includes `timeout`.

The adapter rejects incorrectly typed results and results with `correctness=true` but no valid speedup.

## Profiling

Profiling is enabled by default. If you do not have a profiler, use `--eval-profiling false` to exercise the generation/evaluation loop. This changes the feedback and evaluation conditions, so it does not establish a reproduction of the original benchmark results.

To supply profiler feedback, return the optional summary below. This example represents two measurements with four kernel launches in total:

```json
{
  "metadata": {
    "profiling": {
      "events_summary": {
        "num_trials": 2,
        "gpu": {
          "kernel_launches": 4,
          "busy_ratio": 0.8,
          "kernel_time_us": 100,
          "per_trial": {"kernel_launches": 2}
        },
        "kernels": {
          "top": [{"name": "my_kernel", "share": 1.0, "count": 4, "ops": []}],
          "others": {"total_us": 0}
        }
      }
    }
  }
}
```

If no profiler summary is provided, the harness omits that feedback. Your backend must still implement correctness checks and rule-violation detection.

## Running without an evaluator

Multiple rollouts are still available, but each trajectory generates only one turn. Multi-turn optimization requires actual execution feedback; model weights and an inference endpoint alone do not complete the generation/evaluation/revision loop.

`GET /health` confirms only that the adapter process is reachable. It does not establish that an evaluation backend has been implemented or is operational.
