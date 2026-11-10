#!/usr/bin/env python3
"""Run Jasper-K kernel optimization across a directory of tasks.

The harness connects to an OpenAI-compatible completion service, optionally starts
local vLLM serving, and runs independent optimization trajectories. Each trajectory
writes its best available kernel as it progresses. Configuration is read from
harness.toml; per-task generation and evaluation are implemented in solve.py.

    python3 harness.py --problems /path/to/tasks --out ./run_out --samples 16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import solve                                  # same folder: per-problem generation + self-eval

HERE = Path(__file__).resolve().parent
T0 = time.time()
_PID = re.compile(r"^(\d+)_")


_LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    """Write a timestamped log line without interleaving concurrent output."""
    with _LOG_LOCK:
        print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def ensure_model_service(cfg: dict, log) -> bool:
    """Step (1): do nothing if the endpoint answers; otherwise, with autostart=true,
    call serve.sh to bring one up.

    Returns whether *we* were the ones who started it -- that decides whether to stop
    it when we finish.
    """
    model = cfg.get("model", {})
    serve_cfg = model.get("serve", {}) or {}
    # Apply the configured port override consistently to serving and generation.
    port = os.environ.get("JASPER_K_PORT", "").strip() or str(serve_cfg.get("port", 8111))
    base = str(model.get("base_url", "")).rstrip("/")
    if os.environ.get("JASPER_K_PORT", "").strip():
        base = re.sub(r"://([^/:]+)(:\d+)?", rf"://\1:{port}", base) or f"http://127.0.0.1:{port}/v1"
        base = base.rstrip("/")
        # Write it back into cfg: solve.py builds its client from cfg["model"]["base_url"],
        # and if the two disagree the harness talks to the new port while solving talks to
        # the old one.
        cfg.setdefault("model", {})["base_url"] = base
        serve_cfg["port"] = int(port)
        log(f"service assigned port {port}; serving and connecting there")
    try:
        solve.get_json(f"{base}/models", timeout=5)
        log(f"model service already up: {base}")
        return False
    except Exception:  # noqa: BLE001  not reachable is normal; decide below whether to start
        pass
    if not model.get("autostart"):
        log(f"cannot reach {base} and autostart=false -- every problem will fail to sample")
        return False

    serve = serve_cfg
    env = dict(os.environ)
    # Resolve checkpoint paths relative to this harness directory.
    model_path = str(serve.get("path", ""))
    if model_path and not os.path.isabs(model_path):
        model_path = str((HERE / model_path).resolve())
    env |= {"MODEL": model_path, "PORT": port,
            "TP": str(serve.get("tp", 1)), "UTIL": str(serve.get("gpu_memory_utilization", 0.85)),
            "WAIT": str(serve.get("wait_seconds", 900)),
            # Window for prompt + generation: problem text, feedback and code go back
            # and forth for several rounds, so allow 2x max_tokens. vllm_args.py clamps
            # it to the model max_position_embeddings.
            "WINDOW": str(2 * int(cfg.get("sampling", {}).get("max_tokens", 8192)))}
    if os.environ.get("JASPER_K_GPUS"):
        env["GPUS"] = os.environ["JASPER_K_GPUS"]     # devices selected for this process
    log(f"starting model service: {env['MODEL']} port={env['PORT']} tp={env['TP']} "
        f"gpus={env.get('GPUS', '<inherited>')} (waiting up to {env['WAIT']}s)")
    t0 = time.time()
    rc = subprocess.run([str(HERE / "serve.sh")], env=env, check=False).returncode
    if rc != 0:
        log(f"model service did not come up (serve.sh exit {rc}) -- every problem will fail to sample")
        return False
    log(f"model service ready in {time.time() - t0:.0f}s")
    return True


def discover(problems_dir: Path, levels: list[int]) -> list[tuple[int, int, Path]]:
    """Benchmark -> [(level, pid, file path)]. The id comes from the filename prefix."""
    out = []
    for lvl in levels:
        for f in sorted((problems_dir / f"level{lvl}").glob("[0-9]*_*.py")):
            m = _PID.match(f.name)
            if m:
                out.append((lvl, int(m.group(1)), f))
    return sorted(out, key=lambda t: (-t[0], t[1]))   # highest level first


SUPPORTS_SAMPLE_SELECTION = True


def selected_jobs(tasks, samples, manifest):
    jobs = [(lvl, pid, path, i) for (lvl, pid, path) in tasks for i in range(samples)]
    if not manifest:
        return jobs
    selected = {tuple(map(int, key)) for key in json.loads(Path(manifest).read_text())}
    available = {(lvl, pid, i) for lvl, pid, _, i in jobs}
    if not selected or not selected <= available:
        raise ValueError("sample selection is empty or references unknown samples")
    return [job for job in jobs if (job[0], job[1], job[3]) in selected]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)   # benchmark dir (level{n}/{pid}_*.py)
    ap.add_argument("--out", required=True)        # write L{level}P{pid}.py here
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--timeout", type=float, default=3600)   # wall clock for the whole call
    ap.add_argument("--levels", default="")        # "1,2"; empty = whatever is in the dir
    ap.add_argument("--samples", type=int, default=1)        # independent trajectories per task
    ap.add_argument("--eval-profiling", choices=["true", "false"], default=None)
    ap.add_argument("--only-samples", help="JSON list of [level, problem_id, sample] to regenerate")
    args = ap.parse_args()

    problems_dir = Path(args.problems)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = ([int(x) for x in args.levels.split(",") if x.strip()]
              or sorted(int(p.name[5:]) for p in problems_dir.glob("level[0-9]*")))
    tasks = discover(problems_dir, levels)
    jobs = selected_jobs(tasks, args.samples, args.only_samples)

    cfg = tomllib.loads((HERE / "harness.toml").read_text())
    if args.eval_profiling is not None:
        cfg.setdefault("loop", {})["eval_profiling"] = args.eval_profiling == "true"
    # Read once, up front: a typo in the key should stop the run here, not blow up every
    # trajectory one by one.
    profiling = solve.eval_profiling(cfg)

    # -- step (1): make sure the model service is up --------------------------
    # Done once, before the walk: a 27B takes minutes to come up, and redoing that per
    # problem would burn the entire budget on loading weights.
    started = ensure_model_service(cfg, log)

    # 0 (the default) means no cap: every trajectory starts at once. vLLM batches
    # requests continuously, so holding them back here mostly leaves the GPU idle --
    # and a capped pool splits the wall clock into per-job slices, which makes an early
    # trajectory that could have used the spare time stop instead. Set a number only if
    # your model service cannot take the load.
    workers = int(cfg.get("loop", {}).get("workers", 0) or 0)
    reserve = float(cfg.get("loop", {}).get("reserve_seconds", 120))
    deadline = T0 + args.timeout - reserve       # leave slack; do not finish on the deadline

    endpoint = os.environ.get("JASPER_K_EVALUATOR_ENDPOINT", "")

    # Divide an optional task quota across its independent trajectories.
    quota = int(os.environ.get("JASPER_K_EVAL_QUOTA", "0") or 0)
    rounds_cap = max(1, quota // max(1, args.samples)) if quota else 0
    # Use deterministic seed offsets to make repeated runs comparable.
    base_seed = cfg.get("sampling", {}).get("seed")
    prefix = (f"retry: {len(jobs)} trajectories; " if args.only_samples else
              f"{len(tasks)} problems x {args.samples} trajectories = {len(jobs)}; ")
    log(prefix +
        f"self-eval quota {quota or 'unset'} per problem, "
        f"{rounds_cap or 'per config'} rounds per trajectory "
        f"(each usable candidate is evaluated), "
        f"seed {base_seed if base_seed is not None else 'unset'}")

    pool_size = workers if workers > 0 else max(1, len(jobs))
    log(f"levels {levels}, wall clock {args.timeout:.0f}s, "
        f"{'no worker cap' if workers <= 0 else f'{workers} workers'}, "
        f"self-eval {'on' if endpoint else 'off'}, "
        f"eval profiling {'on (profiler feedback requested)' if profiling else 'off'}")

    remaining = {"n": len(jobs)}
    results: list[dict] = []

    def one(job) -> dict:
        lvl, pid, path, si = job
        tag = f"L{lvl}P{pid}" + (f"_s{si}" if si else "")
        seed = None if base_seed is None else int(base_seed) + si + 1000 * (lvl * 1000 + pid)
        try:
            r = solve.solve_one(
                ref_code=path.read_text(), level=lvl, pid=pid, backend=args.backend,
                out_path=out_dir / f"{tag}.py", deadline=deadline, cfg=cfg,
                endpoint=endpoint, rounds_cap=rounds_cap, seed=seed, profiling=profiling,
                log=lambda m: log(f"{tag}: {m}"),
            )
        except Exception as e:  # noqa: BLE001  isolate failures to the affected trajectory
            log(f"{tag}: failed {type(e).__name__}: {e}")
            r = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        remaining["n"] -= 1
        r |= {"level": lvl, "problem_id": pid, "sample": si}
        log(f"{tag}: {'saved' if r.get('ok') else 'nothing (' + str(r.get('reason')) + ')'}"
            + (f", {r.get('rounds', 0)} rounds" if r.get("rounds") else "")
            + (f", self-eval {r['self_speedup']:.2f}x" if r.get("self_speedup") else "")
            + f"  [{remaining['n']} left, {(deadline-time.time())/60:.0f} min to go]")
        return r

    # -- step (2): solve ------------------------------------------------------
    try:
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            results = list(pool.map(one, jobs))
    finally:
        if started and cfg.get("model", {}).get("serve", {}).get("stop_when_done"):
            # Only stop what we started. On a timeout the whole process group is
            # killed, and vLLM is a descendant of this process, so it goes too; on a
            # normal exit nothing reaps it for us, hence the explicit stop (otherwise it
            # keeps sitting on the cards).
            log("done; stopping the model service we started")
            subprocess.run([str(HERE / "serve.sh"), "--stop"], check=False)

    # Write a machine-readable summary for this run.
    (out_dir / "_harness_results.json").write_text(
        json.dumps({"seconds": round(time.time() - T0, 1), "eval_profiling": profiling,
                    "results": results},
                   ensure_ascii=False, indent=2))
    ok = sum(1 for r in results if r.get("ok"))
    # Report generation limits separately from evaluator failures.
    turns = sum(r.get("rounds", 0) for r in results)
    cut = sum(r.get("truncated_turns", 0) for r in results)
    slow = sum(r.get("timeout_turns", 0) for r in results)
    log(f"finished: {ok}/{len(jobs)} trajectories saved a candidate "
        f"({len({(r['level'], r['problem_id']) for r in results if r.get('ok')})}/{len(tasks)} problems), "
        f"in {(time.time()-T0)/60:.1f} min")
    if turns:
        log(f"turns: {turns} total, {cut} cut off by max_tokens ({cut / turns:.1%}), "
            f"{slow} past the time budget ({slow / turns:.1%}) -- both counted as failed rounds")
        forced = sum(r.get("think_forced_turns", 0) for r in results)
        if forced:
            log(f"turns: {forced} had their thinking force-closed by think_budget ({forced / turns:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
