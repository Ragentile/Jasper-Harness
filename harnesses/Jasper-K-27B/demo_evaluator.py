"""Synthetic protocol demo: returns fixtures, never compiles or executes kernels.

Use for adapter integration only. Values are illustrative, not benchmark results.
Start with: python3 evaluator_adapter.py --evaluator demo_evaluator:evaluate
"""
import copy
import json
from pathlib import Path

CASES = json.loads((Path(__file__).parent/'examples/feedback_cases.json').read_text())


def evaluate(request: dict) -> dict:
    case = request.get('cfg', {}).get('example_case', 'success')
    if case not in CASES:
        raise ValueError('Unknown example_case; choose one of: ' + ', '.join(CASES))
    return copy.deepcopy(CASES[case]['response'])
