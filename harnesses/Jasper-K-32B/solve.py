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
import math
import os
import re
import string
import sys
import threading
import time
import tomllib
import urllib.request
from pathlib import Path

from thinking_budget import (ThinkingBudget, ThinkingBudgetExhausted,
                             validate_config as validate_thinking_config)

import kernel_protocol as protocol

HERE = Path(__file__).resolve().parent

def generation_hang_guard(cfg: dict) -> float | None:
    """Return the generation HTTP timeout; zero permits unlimited waiting."""
    override = os.environ.get("GENERATION_TIMEOUT_SECONDS", "").strip()
    value = float(override or cfg.get("loop", {}).get("generation_timeout", 0))
    if value < 0 or not math.isfinite(value):
        raise ValueError("generation timeout must be finite and non-negative")
    return value or None


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


class ContextWindowExhausted(ValueError):
    """The conversation leaves no room for a response in the context window."""


class Model:
    """Client for an OpenAI-compatible /v1/completions service.

    Prompts are rendered with the tokenizer in model.serve.path. ChatML is used when
    no tokenizer directory is configured. model.reasoning_effort is passed to the
    template when set. Thinking-budget continuation requires server-returned token IDs.
    """

    def __init__(self, cfg: dict):
        think_budget = validate_thinking_config(cfg)
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
        self.context_tokens = int(self.sampling.get("context_tokens", 0) or 0)
        if self.context_tokens < 0:
            raise ValueError("sampling.context_tokens must be non-negative")
        serve_window = int(m.get("serve", {}).get("max_model_len", 0) or 0)
        if self.context_tokens and serve_window and self.context_tokens > serve_window:
            raise ValueError("sampling.context_tokens exceeds model.serve.max_model_len")
        if self.context_tokens and not self.tok_path:
            raise ValueError("dynamic response budgeting requires model.serve.path for exact token counts")
        self.think_policy = ThinkingBudget(cfg, _tokenizer(self.tok_path)) if think_budget else None
        self.thinking_rounds: list[dict] = []

    def _params(self, seed: int | None, prompt_tokens: int = 0) -> dict:
        max_tokens = int(self.sampling.get("max_tokens", 22528))
        if max_tokens < 1:
            raise ValueError("sampling.max_tokens must be positive")
        if self.context_tokens:
            remaining = self.context_tokens - prompt_tokens
            if remaining < 1:
                raise ContextWindowExhausted(
                    f"prompt uses {prompt_tokens} tokens, leaving no response room "
                    f"in the {self.context_tokens}-token context"
                )
            max_tokens = min(max_tokens, remaining)
        p = {"model": self.name, "n": 1,
             "temperature": float(self.sampling.get("temperature", 0.9)),
             "top_p": float(self.sampling.get("top_p", 0.95)),
             "max_tokens": max_tokens}
        # Send top_k explicitly so checkpoint defaults do not override this setting.
        p["top_k"] = int(self.sampling.get("top_k", -1))
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
        rendered = self._render(messages)
        # Count the complete rendered prompt before assigning the response budget.
        prompt_tokens = (len(_tokenizer(self.tok_path)(
            rendered, add_special_tokens=False)["input_ids"]) if self.context_tokens else 0)
        params = self._params(seed, prompt_tokens=prompt_tokens)
        if log and self.context_tokens:
            log(f"model context: {prompt_tokens} prompt + up to {params['max_tokens']} "
                f"response tokens (window {self.context_tokens})")
        if self.think_policy is not None:
            request_at = time.monotonic()

            def request(payload):
                timeout = None if budget is None else budget - (time.monotonic() - request_at)
                if timeout is not None and timeout <= 0:
                    raise TimeoutError("generation timeout reached before thinking continuation")
                return post_json(f"{self.base}/completions", payload,
                                 timeout=timeout, headers=self._auth())

            record: dict = {"prompt_tokens": prompt_tokens}
            self.thinking_rounds.append(record)
            try:
                return self.think_policy.generate(rendered, params, request, record, log)
            except ThinkingBudgetExhausted as exc:
                record.update(status="context_exhausted", error=str(exc))
                raise ContextWindowExhausted(str(exc)) from exc
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise

        d = post_json(f"{self.base}/completions",
                      params | {"prompt": rendered, "stop": ["<|im_end|>"]},
                      timeout=budget, headers=self._auth())
        ch = d["choices"][0]
        return ch.get("text") or "", ch.get("finish_reason", "")


# Request measured feedback from the configured evaluator.

def self_evaluate(endpoint: str, ref_code: str, kernel: str, backend: str,
                  timeout: float, log, margin: float = 1800,
                  profiling: bool = False, eval_mode: str = "",
                  num_correct_trials: int = 5, num_perf_trials: int = 100) -> dict | None:
    """Send one candidate to the configured evaluator.

    The request includes reference and candidate source, measurement settings, and an
    optional profiler request. The backend implements compilation, correctness checks,
    timing, rule enforcement, and execution isolation. The HTTP timeout includes an
    additional margin for queueing and response delivery.
    """
    cfg = {"backend": backend, "num_correct_trials": int(num_correct_trials),
           "measure_performance": True, "num_perf_trials": int(num_perf_trials)}
    if str(eval_mode).strip():
        cfg["eval_mode"] = str(eval_mode).strip()
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

    When disabled, a length-limited response is parsed and evaluated before the
    trajectory stops. When enabled, it is recorded as truncated and refinement continues.
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
    generation_timeout = generation_hang_guard(cfg)
    feedback_cfg = cfg.get("feedback", {})
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
    stopped_reason = "max_rounds"
    for rnd in range(max_rounds):
        left = deadline - time.time()
        if left < 60:
            log(f"{left:.0f}s left, not enough for another round")
            stopped_reason = "deadline"
            break
        try:
            text, finish = model.chat(
                build_messages(problem_prompt, system_prompt, history, backend, kv, profiling),
                # Default to unlimited HTTP waiting so server queueing cannot consume
                # a generation turn. An explicit GENERATION_TIMEOUT_SECONDS opts in.
                # The trajectory deadline and round/token limits remain separate.
                budget=generation_timeout, seed=seed, log=log)
        except ContextWindowExhausted as e:
            stopped_reason = "context_exhausted"
            log(f"{e}; ending trajectory and retaining the best previous kernel")
            break
        except TimeoutError:
            timeout_turns += 1
            infra_error = True
            stopped_reason = "generation_timeout"
            log("sampling timed out; ending trajectory without inventing compile feedback")
            break
        except Exception as e:  # noqa: BLE001
            log(f"sampling failed: {type(e).__name__}: {e}")
            infra_error = True
            stopped_reason = "sampling_error"
            break
        rounds = rnd + 1
        length_limited = finish == "length"
        truncated_turns += int(length_limited)
        if length_limited and trunc_fb:
            log(f"round {rnd}: max_tokens reached; truncated feedback enabled, not evaluated")
            history.append({"turn": rnd, "kernel": protocol.TRUNCATED_KERNEL_PLACEHOLDER,
                            "summary": "", "eval": protocol._truncated_eval_result()})
            continue
        parsed = protocol.parse_structured_response(text)
        summary = parsed.thought_summary or ""
        if length_limited and not parsed.thought:
            summary = ""

        if not parsed.format_ok:
            log(f"round {rnd}: no usable ModelNew in the reply (finish={finish})")
            history.append({"turn": rnd, "kernel": parsed.kernel or "", "summary": summary,
                            "eval": _eval_record(None, format_ok=False)})
            if length_limited:
                stopped_reason = "length"
                break
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
            timeout=float(feedback_cfg.get("timeout", 300)),
            margin=float(feedback_cfg.get("client_margin", 1800)), log=log,
            profiling=profiling,
            eval_mode=str(feedback_cfg.get("eval_mode", "")),
            num_correct_trials=int(feedback_cfg.get("num_correct_trials", 5)),
            num_perf_trials=int(feedback_cfg.get("num_perf_trials", 100)))
        if res is None:
            infra_error = True
            stopped_reason = "evaluation_transport_error"
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
            stopped_reason = "evaluation_infra_error"
            break                               # stop when the evaluator reports an infrastructure failure
        history.append({"turn": rnd, "kernel": kernel, "summary": summary, "eval": er})
        if length_limited:
            # Track incomplete generations separately from evaluated candidates.
            stopped_reason = "length"
            log(f"round {rnd}: length-limited response processed; ending trajectory")
            break

    return {"ok": bool(best), "reason": "" if best else "never produced a usable ModelNew",
            "rounds": rounds, "chars": len(best), "truncated_turns": truncated_turns,
            "timeout_turns": timeout_turns, "infra_error": infra_error, "eval_timeout_turns": eval_timeout_turns,
            "stopped_reason": stopped_reason,
            "think_forced": sum(r.get("forced", False) for r in getattr(model, "thinking_rounds", [])),
            "thinking_budget_rounds": getattr(model, "thinking_rounds", []),
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
