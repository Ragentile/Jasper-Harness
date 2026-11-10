# Evaluator template and feedback examples

These examples show the complete path from an evaluator response to the feedback sent to the model on its next turn. **All example measurements and verdicts are synthetic.** The demo does not compile or execute kernels and must not be used to report model performance.

Run the commands below from the parent harness directory, which contains `harness.py`.

## 1. Try the HTTP contract without a model or GPU

Start the included fixture-based evaluator:

```bash
python3 evaluator_adapter.py --evaluator demo_evaluator:evaluate --host 127.0.0.1 --port 9000
```

In another terminal:

```bash
curl --fail-with-body http://127.0.0.1:9000/evaluate \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

The response has `compiled=true`, `correctness=true`, a 1.0 ms reference time, a 0.5 ms candidate time, and `speedup=2.0`. These values come from a fixture. Set `cfg.example_case` in the request to any case in [feedback_cases.json](feedback_cases.json) to inspect other outcomes: compilation failure, incorrect output, a correct but slower candidate, or success with profiling. `example_case` is understood only by the demo, not required by the evaluator protocol.

## 2. See exactly what the model receives

```bash
python3 preview_feedback.py --case success
python3 preview_feedback.py --case correctness_error
python3 preview_feedback.py --case success_with_profile
```

This command calls the selected harness's existing feedback renderer. It combines the previous candidate code, the model's short summary, the evaluator verdict, and optional profiling into a `Previous Attempt` block, followed by instructions for the next turn. The evaluator returns structured JSON; it does not need to write the model-facing feedback text itself.

The renderer's success feedback includes:

```text
Your previous answer was correct but can be made faster. Here is the speedup you achieved relative to the baseline: 2.00x
```

See [feedback_success.txt](feedback_success.txt) for the complete rendered example. [feedback_cases.json](feedback_cases.json) contains the request-side candidate and full response for each outcome. Diagnostic messages for incorrect outputs should state the mismatch clearly; keep service failures separate from candidate failures.

## 3. Connect real measurements

```bash
cp example_evaluator.py my_evaluator.py
# Implement run_backend(request) in my_evaluator.py.
python3 evaluator_adapter.py --evaluator my_evaluator:evaluate --host 127.0.0.1 --port 9000
```

[example_evaluator.py](../example_evaluator.py) defines a `BackendResult` template and converts measured reference/candidate times to speedup. Implement `run_backend` to call your execution backend and return actual compilation status, test counts, timings, errors, and optional profiling. The hook raises `NotImplementedError` until you implement it.

Your backend supplies compilation, reference inputs, numerical checks, warm-up, synchronized timing, rule checks, worker isolation, timeouts, and device scheduling. The HTTP adapter and template do not provide those operations. See [EVALUATOR_API.md](../EVALUATOR_API.md) for all request and response fields.

Then follow the [example harness configuration](../README.md#running-the-harness), using your own device IDs and evaluator URL. Use `selftest.sh` with your configured model service and implemented evaluator for a live integration check on the included sample problems. The protocol demo above runs without model weights or a GPU.
