"""Template for adapting your execution backend to the Jasper-K evaluator API.

Implement run_backend() with your compiler, correctness tests, timing procedure,
device scheduling, worker isolation, and timeout enforcement. This template never
executes candidate code in the HTTP adapter process.
"""
from dataclasses import dataclass, field
import math

from evaluator_adapter import validate_result


@dataclass
class BackendResult:
    compiled: bool
    correctness: bool
    tests_passed: int
    tests_total: int
    runtime_ms: float | None = None
    baseline_runtime_ms: float | None = None
    error_message: str = ''
    cheated: bool = False
    infra_error: bool = False
    metadata: dict = field(default_factory=dict)


def run_backend(request: dict) -> BackendResult:
    """Replace this hook with a call to your isolated worker or execution service.

    Inputs:
      request['ref_code']: complete reference Model source and input constructors.
      request['kernel_code']: complete candidate ModelNew source.
      request['timeout']: execution deadline in seconds; enforce in your worker.
      request['cfg']: backend, trial counts, profiling flags, and optional settings.

    Your worker should:
      1. Load/compile the reference and candidate; report compiler diagnostics.
      2. Construct the actual test inputs and compare outputs using your tolerances.
      3. If every test passes, warm up and time both implementations with device
         synchronization, matching inputs, and a consistent measurement protocol.
      4. Optionally attach metadata.profiling.events_summary, as in EVALUATOR_API.md.

    Return actual trial counts and milliseconds. On a candidate failure, return
    correctness=False with the relevant diagnostic and no performance claim.
    Report service failures with infra_error=True; do not treat them as bad kernels.
    """
    raise NotImplementedError('Implement run_backend() using your own execution backend.')


def evaluate(request: dict) -> dict:
    """Convert backend measurements to the JSON response consumed by the harness."""
    result = run_backend(request)
    if not isinstance(result, BackendResult):
        raise TypeError('run_backend() must return BackendResult')
    if (type(result.tests_passed) is not int or type(result.tests_total) is not int
            or not 0 <= result.tests_passed <= result.tests_total):
        raise ValueError('Return valid actual correctness-test counts')
    speedup = None
    if result.correctness:
        if (result.cheated or result.infra_error or result.tests_total == 0
                or result.tests_passed != result.tests_total):
            raise ValueError('A correct result requires all tests to pass without rule or service failures')
        for timing in (result.runtime_ms, result.baseline_runtime_ms):
            if type(timing) not in (int, float) or not math.isfinite(timing) or timing <= 0:
                raise ValueError('Correct results require positive, finite measured runtimes')
        speedup = result.baseline_runtime_ms / result.runtime_ms
    return validate_result({
        'compiled': result.compiled,
        'correctness': result.correctness,
        'cheated': result.cheated,
        'infra_error': result.infra_error,
        'tests_passed': result.tests_passed,
        'tests_total': result.tests_total,
        'runtime_ms': result.runtime_ms,
        'baseline_runtime_ms': result.baseline_runtime_ms,
        'speedup': speedup,
        'error_message': result.error_message,
        'metadata': result.metadata,
    })
