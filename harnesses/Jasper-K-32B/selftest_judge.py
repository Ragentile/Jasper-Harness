#!/usr/bin/env python3
"""Re-evaluate self-test outputs through the configured evaluator.

Called by selftest.sh. For each task, report the fastest verified-correct candidate
and aggregate the results. Profiling is disabled for this final recheck; correctness
and timing are supplied by the user's evaluator.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve  # noqa: E402  same directory

NAME = re.compile(r"^L(\d+)P(\d+)(?:_s(\d+))?\.py$")


def main() -> int:
    problems_dir, out_dir, n_samples = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    endpoint = os.environ.get("JASPER_K_EVALUATOR_ENDPOINT", "")
    backend = os.environ.get("JASPER_K_BACKEND", "cuda")
    timeout = float(os.environ.get("JASPER_K_EVAL_TIMEOUT", 240))
    margin = float(os.environ.get("JASPER_K_EVAL_CLIENT_MARGIN", 1800))

    refs = {}
    for f in glob.glob(os.path.join(problems_dir, "level*", "[0-9]*_*.py")):
        lvl = int(os.path.basename(os.path.dirname(f))[5:])
        refs[(lvl, int(os.path.basename(f).split("_", 1)[0]))] = Path(f).read_text()

    jobs = []
    for f in sorted(out_dir.glob("*.py")):
        m = NAME.match(f.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if key in refs and int(m.group(3) or 0) < n_samples:
            jobs.append((key, f))

    print(f"  collected {len(jobs)} kernels, covering "
          f"{len({k for k, _ in jobs})}/{len(refs)} problems\n")
    if not jobs:
        print("  nothing collected -- check the filenames are L{level}P{pid}[_s{i}].py")
        return 1

    def judge(item):
        (lvl, pid), f = item
        r = solve.self_evaluate(endpoint=endpoint, ref_code=refs[(lvl, pid)],
                                kernel=f.read_text(), backend=backend,
                                timeout=timeout, margin=margin, log=lambda _m: None) or {}
        return (lvl, pid), f.name, r

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        done = list(pool.map(judge, jobs))

    best: dict[tuple[int, int], float] = {k: 0.0 for k in refs}
    n_ok = n_compiled = 0
    for (lvl, pid), name, r in sorted(done):
        correct = bool(r.get("correctness")) and not r.get("cheated") and not r.get("infra_error")
        sp = r.get("speedup") or 0.0
        n_ok += correct
        n_compiled += bool(r.get("compiled"))
        if correct and sp:
            best[(lvl, pid)] = max(best[(lvl, pid)], float(sp))
        state = "correct" if correct else "compiled" if r.get("compiled") else "failed"
        extra = f", {sp:.2f}x" if correct and sp else ""
        err = (r.get("error_message") or "").strip().splitlines()
        print(f"  {name:<18} {state}{extra}"
              + (f"   {err[0][:70]}" if err and not correct else ""))

    vals = [v if v > 0 else 1.0 for v in best.values()]   # unsolved counts as 1.0 (= no speedup)
    geo = math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
    solved = sum(1 for v in best.values() if v > 0)
    print(f"\n  compiled {n_compiled}/{len(done)}   correct {n_ok}/{len(done)}   "
          f"solved {solved}/{len(refs)}   speedup {geo:.3f}x")
    print("  per problem: " + ", ".join(f"L{l}/P{p} {v:g}x"
                                        for (l, p), v in zip(sorted(best), vals)))
    print("\n  NOTE checks are provided by your evaluator; profiling is off in this final recheck. "
          "Record evaluator settings when comparing results.")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
