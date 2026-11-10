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
import copy
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
    port = str(serve_cfg.get("port", 8111))
    base = str(model.get("base_url", "")).rstrip("/")
    model_path = str(serve_cfg.get("path", ""))
    if model_path and not os.path.isabs(model_path):
        model_path = str((HERE / model_path).resolve())
    try:
        available = solve.get_json(f"{base}/models", timeout=5)["data"][0]
    except Exception:  # noqa: BLE001  not reachable is normal; decide below whether to start
        available = None
    if available is not None:
        actual = available.get("root") or available.get("id")
        if model_path and actual != model_path:
            raise RuntimeError(f"{base} serves {actual!r}, expected {model_path!r}")
        model["name"] = available["id"]
        log(f"model service already up: {base}")
        return False
    if not model.get("autostart"):
        raise RuntimeError(f"cannot reach {base} and autostart=false")

    serve = serve_cfg
    env = dict(os.environ)
    env["PIDFILE"] = str(HERE / f"vllm-{port}.pid")
    env["LOG"] = str(HERE / f"vllm-{port}.log")
    # Resolve checkpoint paths relative to this harness directory.
    env |= {"MODEL": model_path, "PORT": port,
            "TP": str(serve.get("tp", 1)), "UTIL": str(serve.get("gpu_memory_utilization", 0.85)),
            "WAIT": str(serve.get("wait_seconds", 900)),
            # Window for prompt + generation: problem text, feedback and code go back
            # and forth for several rounds, so allow 2x max_tokens. vllm_args.py clamps
            # it to the model max_position_embeddings.
            "WINDOW": str(2 * int(cfg.get("sampling", {}).get("max_tokens", 8192)))}
    if serve.get("gpus") or os.environ.get("JASPER_K_GPUS"):
        env["GPUS"] = serve.get("gpus") or os.environ["JASPER_K_GPUS"]     # devices selected for this process
    log(f"starting model service: {env['MODEL']} port={env['PORT']} tp={env['TP']} "
        f"gpus={env.get('GPUS', '<inherited>')} (waiting up to {env['WAIT']}s)")
    t0 = time.time()
    rc = subprocess.run([str(HERE / "serve.sh")], env=env, check=False).returncode
    if rc != 0:
        log(f"model service did not come up (serve.sh exit {rc})")
        raise RuntimeError("model service failed to start; see its log")
    log(f"model service ready in {time.time() - t0:.0f}s")
    return True


def replica_configs(cfg: dict) -> list[dict]:
    serve = cfg.get("model", {}).get("serve", {})
    replicas, tp = serve.get("replicas", 1), serve.get("tp", 1)
    limit = cfg.get("loop", {}).get("generation_concurrency_per_replica", 64)
    for key, value in (("replicas", replicas), ("tp", tp),
                       ("generation_concurrency_per_replica", limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be an integer >= 1")
    cards = [x.strip() for x in os.environ.get("JASPER_K_GPUS", "").split(",") if x.strip()]
    if (cards or replicas > 1) and (len(cards) != replicas * tp or len(set(cards)) != len(cards)):
        raise RuntimeError(f"need {replicas * tp} assigned GPUs for {replicas} replicas; got {cards}")
    primary_port = int(os.environ.get("JASPER_K_PORT", "").strip() or serve.get("port", 8111))
    configs = []
    for i in range(replicas):
        item = copy.deepcopy(cfg)
        sc = item.setdefault("model", {}).setdefault("serve", {})
        if cards:
            sc["gpus"] = ",".join(cards[i * tp:(i + 1) * tp])
        port = primary_port + 10000 * i
        if not 1 <= port <= 65535:
            raise ValueError(f"replica {i} port {port} is out of range")
        sc["port"] = port
        item["model"]["base_url"] = f"http://127.0.0.1:{port}/v1"
        configs.append(item)
    return configs


def stop_replicas(configs: list[dict], log) -> None:
    for item in configs:
        if not item.get("_started"):
            continue
        port = item["model"]["serve"]["port"]
        log(f"stopping model service on port {port}")
        subprocess.run([str(HERE / "serve.sh"), "--stop"],
                       env=dict(os.environ, PIDFILE=str(HERE / f"vllm-{port}.pid")), check=False)


def start_replicas(cfg: dict, log) -> list[dict]:
    configs = replica_configs(cfg)
    try:
        for i, item in enumerate(configs):
            item["_started"] = ensure_model_service(item, log)
            limit = int(item.get("loop", {}).get("generation_concurrency_per_replica", 64))
            item["_generation_semaphore"] = threading.BoundedSemaphore(limit)
            log(f"replica {i}: {item['model']['base_url']}, GPUs={item['model']['serve'].get('gpus')}, generation concurrency={limit}")
    except BaseException:
        stop_replicas(configs, log)
        raise
    return configs


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
    solve.jasper_version(cfg)
    solve.truncation_feedback(cfg)

    # -- step (1): make sure the model service is up --------------------------
    # Done once, before the walk: a 27B takes minutes to come up, and redoing that per
    # problem would burn the entire budget on loading weights.
    model_configs = start_replicas(cfg, log)

    # workers controls trajectory threads; a separate semaphore per replica limits
    # generation HTTP requests. Waiting for that semaphore does not use the HTTP
    # hang-guard budget. Self-evaluation does not hold a generation slot.
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

    def one(indexed_job) -> dict:
        index, job = indexed_job
        job_cfg = model_configs[index % len(model_configs)]
        lvl, pid, path, si = job
        tag = f"L{lvl}P{pid}" + (f"_s{si}" if si else "")
        seed = None if base_seed is None else int(base_seed) + si + 1000 * (lvl * 1000 + pid)
        try:
            r = solve.solve_one(
                ref_code=path.read_text(), level=lvl, pid=pid, backend=args.backend,
                out_path=out_dir / f"{tag}.py", deadline=deadline, cfg=job_cfg,
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
            results = list(pool.map(one, enumerate(jobs)))

    finally:
        if cfg.get("model", {}).get("serve", {}).get("stop_when_done"):
            # Stop only model services started by this harness.
            stop_replicas(model_configs, log)

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
