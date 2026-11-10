"""Render illustrative evaluator responses using this model's actual feedback code."""
import argparse
import json
from pathlib import Path
import tomllib

from evaluator_adapter import validate_result
import solve

HERE = Path(__file__).resolve().parent


def render(case: str) -> str:
    cases = json.loads((HERE/'examples/feedback_cases.json').read_text())
    example = cases[case]
    response = validate_result(example['response'])
    cfg = tomllib.loads((HERE/'harness.toml').read_text())
    history = [{'turn':0, 'kernel':example['kernel_code'], 'summary':example['summary'],
                'eval':solve._eval_record(response, format_ok=True)}]
    return solve.refinement_text(history, 'cuda', solve.jasper_version(cfg), profiling=True)


if __name__ == '__main__':
    cases = json.loads((HERE/'examples/feedback_cases.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=list(cases), default='success')
    args = parser.parse_args()
    print('ILLUSTRATIVE DATA: exact harness feedback formatting, not measured model performance.\n')
    print(render(args.case))
