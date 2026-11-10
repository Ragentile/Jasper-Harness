#!/usr/bin/env python3
"""Generate and refine a kernel implementation for one task.

The model receives the reference implementation and feedback from the configured
evaluator. Correct candidates take precedence over unverified candidates; among
correct candidates, the fastest measured implementation is retained. Outputs are
written after each improvement so completed work remains available if interrupted.

    python3 solve.py --problem /path/to/task.py --out ./kernel.py --timeout 900
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import threading
import time
import tomllib
import urllib.request
from pathlib import Path

import kernel_protocol as protocol

HERE = Path(__file__).resolve().parent

# Generation HTTP timeout in seconds; unset/0 means unlimited (including server queueing).
# Set GENERATION_TIMEOUT_SECONDS to a positive value to opt into a timeout.
_generation_timeout = float(os.environ.get("GENERATION_TIMEOUT_SECONDS", "0"))
if not 0 <= _generation_timeout < float("inf"):
    raise ValueError("GENERATION_TIMEOUT_SECONDS must be finite and non-negative")
GENERATION_HANG_GUARD = _generation_timeout or None

# Close the thinking stage before continuing answer generation.
THINK_FORCE_TEXT = ("\n\nConsidering the limited time by the user, I have to give the solution "
                    "based on the thinking directly now.\n</think>\n\n")


# -- small HTTP helpers ------------------------------------------------------

def post_json(url: str, payload: dict, timeout: float | None, headers: dict | None = None) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no proxy on the LAN
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_json(url: str, timeout: float = 30) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as resp:
        return json.loads(resp.read())


# -- the model ---------------------------------------------------------------

_TOKENIZERS: dict[str, object] = {}
_TOKENIZER_LOCK = threading.Lock()


def _tokenizer(path: str):
    """One tokenizer per model directory, shared by every trajectory in the process."""
    with _TOKENIZER_LOCK:
        if path not in _TOKENIZERS:
            from transformers import AutoTokenizer
            _TOKENIZERS[path] = AutoTokenizer.from_pretrained(path)
        return _TOKENIZERS[path]


class Model:
    """Client for an OpenAI-compatible /v1/completions service.

    Prompts are rendered with the tokenizer in model.serve.path. ChatML is used when
    no tokenizer directory is configured. model.reasoning_effort is passed to the
    template when set. Thinking-budget continuation requires server-returned token IDs.
    """

    def __init__(self, cfg: dict):
        m = cfg.get("model", {})
        self.base = str(m.get("base_url", "")).rstrip("/")
        self.key = str(m.get("api_key") or "EMPTY")
        self.name = m.get("name") or get_json(f"{self.base}/models")["data"][0]["id"]
        self.sampling = cfg.get("sampling", {})
        self.thinking = bool(m.get("enable_thinking", True))
        # Pass an explicit reasoning effort to the tokenizer template when configured.
        self.effort = str(m.get("reasoning_effort") or "")
        path = str(m.get("serve", {}).get("path") or "")
        self.tok_path = str((HERE / path).resolve()) if path else ""
        self.max_tokens = int(self.sampling.get("max_tokens", 22528))
        self.think_budget = int(self.sampling.get("think_budget", 0) or 0)
        self.think_forced = 0                   # turns whose thinking was force-closed

    def _params(self, seed: int | None) -> dict:
        p = {"model": self.name, "n": 1,
             "temperature": float(self.sampling.get("temperature", 0.9)),
             "top_p": float(self.sampling.get("top_p", 0.95)),
             "max_tokens": self.max_tokens}
        if int(self.sampling.get("top_k", 0) or 0) > 0:
            p["top_k"] = int(self.sampling["top_k"])      # vLLM/sglang honour it; OpenAI ignores
        if seed is not None:
            # Use deterministic seed offsets to make repeated runs comparable.
            p["seed"] = int(seed)
        return p

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}

    def _chatml(self, messages: list[dict]) -> str:
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        out += "<|im_start|>assistant\n"
        return out + "<think>\n" if self.thinking else out

    def _render(self, messages: list[dict]) -> str:
        if not self.tok_path:
            return self._chatml(messages)
        extra = {"reasoning_effort": self.effort} if self.effort else {}
        return _tokenizer(self.tok_path).apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.thinking, **extra)

    def chat(self, messages: list[dict], budget: float | None, seed: int | None = None,
             log=None) -> tuple[str, str]:
        """Generate one response, optionally splitting thinking and answer generation.

        When configured, think_budget limits the first stage. If necessary, a closing
        instruction is appended before continuing within the remaining total token budget.
        """
        prompt = self._render(messages)
        params = self._params(seed) | {"stop": ["<|im_end|>"]}
        budget_t = self.think_budget
        if not 0 < budget_t < self.max_tokens:
            d = post_json(f"{self.base}/completions", params | {"prompt": prompt},
                          timeout=budget, headers=self._auth())
            ch = d["choices"][0]
            return ch.get("text") or "", ch.get("finish_reason", "")

        t0 = time.time()
        d = post_json(f"{self.base}/completions",
                      params | {"prompt": prompt, "max_tokens": budget_t, "return_token_ids": True},
                      timeout=budget, headers=self._auth())
        ch = d["choices"][0]
        text1, fin1 = ch.get("text") or "", ch.get("finish_reason", "")
        if fin1 != "length":
            return text1, fin1
        ids1 = list(ch["token_ids"])
        if "</think>" in text1:
            inject, inject_ids = "", []
        else:
            if not self.tok_path:
                raise RuntimeError("think_budget needs model.serve.path (its tokenizer)")
            inject = THINK_FORCE_TEXT
            inject_ids = _tokenizer(self.tok_path).encode(inject, add_special_tokens=False)
        rest = self.max_tokens - len(ids1) - len(inject_ids)
        if rest <= 0:
            return text1, "length"
        if inject:
            self.think_forced += 1
            if log:
                log(f"think budget {budget_t} reached without </think> -> forced close")
        d = post_json(f"{self.base}/completions",
                      params | {"prompt": list(ch["prompt_token_ids"]) + ids1 + inject_ids,
                                "max_tokens": rest},
                      timeout=None if budget is None else max(1.0, budget - (time.time() - t0)),
                      headers=self._auth())
        ch2 = d["choices"][0]
        return text1 + inject + (ch2.get("text") or ""), ch2.get("finish_reason", "")


# Request measured feedback from the configured evaluator.

def self_evaluate(endpoint: str, ref_code: str, kernel: str, backend: str,
                  timeout: float, log, margin: float = 1800,
                  profiling: bool = False) -> dict | None:
    """Send one candidate to the configured evaluator.

    The request includes reference and candidate source, measurement settings, and an
    optional profiler request. The backend implements compilation, correctness checks,
    timing, rule enforcement, and execution isolation. The HTTP timeout includes an
    additional margin for queueing and response delivery.
    """
    cfg = {"backend": backend, "num_correct_trials": 5,
           "measure_performance": True, "num_perf_trials": 100}
    mode = os.environ.get("JASPER_K_EVAL_MODE", "")
    if mode:
        cfg["eval_mode"] = mode
    if profiling:
        cfg["enable_profiling"] = True
        cfg["return_profiling_events"] = True
    payload = {"ref_code": ref_code, "kernel_code": kernel, "timeout": int(timeout),
               "cfg": cfg}
    # Allow a client-side margin for evaluator queueing and response delivery.
    try:
        return post_json(f"{endpoint.rstrip('/')}/evaluate", payload, timeout=timeout + margin)
    except Exception as e:  # noqa: BLE001
        log(f"self-eval failed (service/network): {type(e).__name__}: {e}")
        return None


def _eval_record(res: dict | None, format_ok: bool) -> dict:
    """Normalize an evaluator response for prompt construction.

    format_ok records whether a candidate was parsed from the model's response.
    An unparseable response is recorded locally without an evaluator request.
    """
    res = dict(res or {})
    return {"format_ok": format_ok,
            "compiled": bool(res.get("compiled")),
            "correctness": bool(res.get("correctness")),
            "cheated": bool(res.get("cheated")),
            "infra_error": bool(res.get("infra_error")),
            # Use default test counts when the evaluator omits them.
            "tests_passed": res.get("tests_passed", 0),
            "tests_total": res.get("tests_total", 5),
            "speedup": res.get("speedup"),
            "runtime_ms": res.get("runtime_ms"),
            "baseline_runtime_ms": res.get("baseline_runtime_ms"),
            "error_message": res.get("error_message") or "",
            # Preserve evaluator metadata for optional profiler feedback.
            "metadata": res.get("metadata") or {}}


def eval_profiling(cfg: dict) -> bool:
    """Read loop.eval_profiling from harness.toml.

    Boolean values and conventional true/false strings are accepted. Invalid values
    raise an error rather than silently changing the requested evaluation behavior.
    """
    v = cfg.get("loop", {}).get("eval_profiling", False)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    raise ValueError(f"loop.eval_profiling accepts only true / false, got {v!r}")


def truncation_feedback(cfg: dict) -> bool:
    """Read whether truncated responses receive dedicated feedback.

    Truncated responses are not evaluated. When enabled, the next turn receives a
    dedicated truncation message; otherwise the response is recorded as a format error.
    """
    v = cfg.get("loop", {}).get("truncation_feedback", False)
    if isinstance(v, bool):
        return v
    raise ValueError(f"loop.truncation_feedback accepts only true / false, got {v!r}")


def _status(er: dict) -> str:
    """Return a concise diagnostic label for a normalized evaluation result."""
    return {"infra_error": "EVAL_SERVICE_ERROR", "truncated": "CUT_OFF_AT_MAX_TOKENS", "cheating": "REJECTED_NOT_YOUR_KERNEL",
            "timeout": "EVAL_TIMEOUT", "format_error": "NO_KERNEL_PARSED",
            "compilation_error": "COMPILE_ERROR", "runtime_error": "RUNTIME_ERROR",
            "correctness_error": "WRONG_RESULT", "performance_warning": "CORRECT_BUT_SLOW",
            "success": "CORRECT"}[protocol._categorize_error(er)]


# Construct the Jasper-K task, system, and refinement prompts.

def jasper_version(cfg: dict) -> "protocol.JasperKVersion":
    """Return the configured Jasper-K prompt and feedback profile."""
    style = str(cfg.get("loop", {}).get("feedback_style") or protocol.DEFAULT_FEEDBACK_STYLE)
    if style not in protocol.JASPER_K_VERSIONS:
        raise ValueError(f"loop.feedback_style accepts only {' / '.join(protocol.JASPER_K_VERSIONS)} "
                         f"; got {style!r}")
    return protocol.JASPER_K_VERSIONS[style]


def profile_section(er: dict, profiling: bool) -> str:
    """Render optional profiler feedback from an evaluator result.

    No profiler text is included when profiling is disabled or usable profiling data
    is absent from the response.
    """
    if not profiling:
        return ""
    text = protocol._profile_feedback(er)
    return "\n\n" + text if text else ""


def refinement_text(history: list[dict], backend: str, kv: "protocol.JasperKVersion",
                    profiling: bool = False) -> str:
    """Construct refinement feedback from every attempt so far, in order.

    Not just the last one. Carrying the whole history is what the Jasper-K style is, and it
    stays bounded without truncating the conversation: each block is a summary (<= 800
    chars) plus the kernel (a wrong one cut at 8000 chars, a correct one whole) plus one
    feedback sentence. The model's raw reasoning is never carried -- that is what the
    ``<SUMMARY>`` block replaces, and why dropping the tags would force the choice
    between "last turn only" and a context blow-up.
    """
    parts = []
    for entry in history:
        er = entry["eval"]
        category = protocol._categorize_error(er)
        summary = protocol._truncate_summary(
            entry.get("summary") or protocol._fallback_summary(er, category))
        parts.append(protocol.JASPER_K_ATTEMPT_TEMPLATE.format(
            turn=entry["turn"] + 1,
            summary=summary,
            kernel=protocol._truncate_kernel(
                entry["kernel"],
                max_len=None if er["correctness"] else protocol.MAX_KERNEL_HISTORY_LEN),
            feedback=protocol._JASPER_K_feedback_line(er, category, backend,
                                              slow_note=kv.slow_note, cheat=kv.cheat)
            + profile_section(er, profiling),
        ))
    parts.append("\n## Instructions\n\n" + kv.closing + "\n")
    text = "\n".join(parts)
    return text + "\nRemember: respond using <KERNEL>...</KERNEL> followed by <SUMMARY>...</SUMMARY>."


def build_messages(problem_prompt: str, system_prompt: str, history: list[dict],
                   backend: str, kv: "protocol.JasperKVersion", profiling: bool = False) -> list[dict]:
    """Build model messages, including prior attempts on refinement turns."""
    messages = [{"role": "system", "content": system_prompt}]
    if not history:
        messages.append({"role": "user", "content": problem_prompt})
    else:
        messages.append({"role": "user", "content": "\n".join(
            [problem_prompt, refinement_text(history, backend, kv, profiling)])})
    return messages


def prompts(ref_code: str, backend: str, kv: "protocol.JasperKVersion") -> tuple[str, str]:
    """Construct the initial task and system prompts.

    The Jasper-K profile adds kernel rules and numerical guidance, and removes task
    instructions that conflict with the structured KERNEL/SUMMARY response format.
    The library-call setting must match the rules enforced by the evaluator.
    """
    problem_prompt = string.Template((HERE / "prompt.txt").read_text()).safe_substitute(
        ref_code=ref_code)
    system_prompt = protocol.MULTITURN_SYSTEM_PROMPT_NO_THINK.format(backend=backend.upper())
    if kv.rules_addendum:
        system_prompt = (protocol.slim_system_prompt(system_prompt, backend)
                         + protocol.build_JASPER_K_system_addendum(backend))
    if kv.strip_kb_conflicts:
        problem_prompt = protocol.strip_kb_output_conflicts(problem_prompt)
    return problem_prompt, system_prompt


# -- one problem --------------------------------------------------------------

def solve_one(ref_code: str, level: int, pid: int, backend: str, out_path: Path,
              deadline: float, cfg: dict, endpoint: str, log,
              rounds_cap: int = 0, seed: int | None = None,
              profiling: bool | None = None) -> dict:
    """Generate, evaluate, and refine a candidate while retaining the best result.

    Every usable candidate is evaluated, including the final turn. Without an evaluator
    endpoint, each trajectory generates a single candidate. Profiling defaults to the
    loop.eval_profiling setting when no explicit override is supplied.
    """
    model = Model(cfg)
    if profiling is None:
        profiling = eval_profiling(cfg)
    kv = jasper_version(cfg)
    trunc_fb = truncation_feedback(cfg)
    problem_prompt, system_prompt = prompts(ref_code, backend, kv)

    max_rounds = int(cfg.get("loop", {}).get("max_rounds", 4)) if endpoint else 1
    if rounds_cap:
        # The quota is **shared across the problem**; the harness works out the
        # per-trajectory cap and passes it in.
        max_rounds = min(max_rounds, rounds_cap)

    history: list[dict] = []
    best, best_speed, rounds, truncated_turns, timeout_turns = "", -1.0, 0, 0, 0
    infra_error = False
    eval_timeout_turns = 0
    for rnd in range(max_rounds):
        left = deadline - time.time()
        if left < 60:
            log(f"{left:.0f}s left, not enough for another round")
            break
        try:
            text, finish = model.chat(
                build_messages(problem_prompt, system_prompt, history, backend, kv, profiling),
                # Default to unlimited HTTP waiting so server queueing cannot consume
                # a generation turn. An explicit GENERATION_TIMEOUT_SECONDS opts in.
                # The trajectory deadline and round/token limits remain separate.
                budget=GENERATION_HANG_GUARD, seed=seed, log=log)
        except TimeoutError:
            text, finish = "", "timeout"
        except Exception as e:  # noqa: BLE001
            log(f"sampling failed: {type(e).__name__}: {e}")
            break
        rounds = rnd + 1
        # Track incomplete generations separately from evaluated candidates.
        if finish in ("length", "timeout"):
            truncated_turns += int(finish == "length")
            timeout_turns += int(finish == "timeout")
            log(f"round {rnd}: hit the {'max_tokens' if finish == 'length' else 'time'} "
                f"limit -> failed round, not evaluated")
            if finish == "length" and trunc_fb:
                history.append({"turn": rnd, "kernel": protocol.TRUNCATED_KERNEL_PLACEHOLDER,
                                "summary": "", "eval": protocol._truncated_eval_result()})
            else:
                history.append({"turn": rnd, "kernel": "", "summary": "",
                                "eval": _eval_record(None, format_ok=False)})
            continue
        parsed = protocol.parse_structured_response(text)
        summary = parsed.thought_summary or ""

        if not parsed.format_ok:
            log(f"round {rnd}: no usable ModelNew in the reply (finish={finish})")
            history.append({"turn": rnd, "kernel": parsed.kernel or "", "summary": summary,
                            "eval": _eval_record(None, format_ok=False)})
            continue

        kernel = parsed.kernel
        if best_speed < 0:                      # nothing verified yet -> newest is the best guess
            out_path.write_text(kernel)
            best = kernel
            log(f"round {rnd}: wrote {len(kernel)} chars")

        if not endpoint:
            break
        res = self_evaluate(
            endpoint, ref_code, kernel, backend,
            timeout=float(os.environ.get("JASPER_K_EVAL_TIMEOUT", "240")),
            margin=float(os.environ.get("JASPER_K_EVAL_CLIENT_MARGIN", "1800")), log=log,
            profiling=profiling)
        if res is None:
            infra_error = True
            break
        er = _eval_record(res, format_ok=True)
        eval_timeout_turns += int(_status(er) == "EVAL_TIMEOUT")
        sp = float(er.get("speedup") or 0)
        log(f"round {rnd} self-eval: {_status(er)}" + (f", {sp:.2f}x" if sp else ""))
        # Retain correct candidates first, then prefer higher measured speedup.
        if er["correctness"] and not er["cheated"] and sp > best_speed:
            best, best_speed = kernel, sp
            out_path.write_text(kernel)
        if er["infra_error"]:
            infra_error = True
            break                               # stop when the evaluator reports an infrastructure failure
        history.append({"turn": rnd, "kernel": kernel, "summary": summary, "eval": er})

    return {"ok": bool(best), "reason": "" if best else "never produced a usable ModelNew",
            "rounds": rounds, "chars": len(best), "truncated_turns": truncated_turns,
            "timeout_turns": timeout_turns, "infra_error": infra_error, "eval_timeout_turns": eval_timeout_turns, "think_forced_turns": model.think_forced,
            "self_speedup": best_speed if best_speed > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--problem-id", type=int, default=0)
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval-profiling", choices=["true", "false"], default=None,
                    help="override loop.eval_profiling for this run")
    args = ap.parse_args()

    t0 = time.time()
    cfg = tomllib.loads((HERE / "harness.toml").read_text())
    r = solve_one(
        ref_code=Path(args.problem).read_text(), level=args.level, pid=args.problem_id,
        backend=args.backend, out_path=Path(args.out), deadline=t0 + args.timeout,
        cfg=cfg, endpoint=os.environ.get("JASPER_K_EVALUATOR_ENDPOINT", ""), seed=args.seed,
        profiling=None if args.eval_profiling is None else args.eval_profiling == "true",
        log=lambda m: print(f"[{time.time()-t0:7.1f}s] {m}", flush=True),
    )
    print(json.dumps(r, ensure_ascii=False))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
