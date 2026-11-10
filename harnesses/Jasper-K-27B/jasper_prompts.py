"""Shared prompt construction for the Jasper-K inference harness."""
from pathlib import Path
import string
import kernel_protocol as protocol
HERE = Path(__file__).resolve().parent

def truncation_feedback(cfg: dict) -> bool:
    """Read whether truncated responses receive dedicated feedback.

    Truncated responses are not evaluated. When enabled, the next turn receives a
    dedicated truncation message; otherwise the response is recorded as a format error.
    """
    v = cfg.get("loop", {}).get("truncation_feedback", False)
    if isinstance(v, bool):
        return v
    raise ValueError(f"loop.truncation_feedback accepts only true / false, got {v!r}")


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


def refinement_text(history, backend, kv, profiling=False):
    return "\n".join(refinement_parts(history, backend, kv, profiling))


def refinement_parts(history: list[dict], backend: str, kv: "protocol.JasperKVersion",
                    profiling: bool = False) -> list[str]:
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
    parts[-1] += "\nRemember: respond using <KERNEL>...</KERNEL> followed by <SUMMARY>...</SUMMARY>."
    return parts


def build_messages(problem_prompt, system_prompt, history, backend, kv, profiling=False):
    messages = [{"role": "system", "content": system_prompt}]
    parts = [problem_prompt]
    if history:
        parts.extend("\n" + part for part in refinement_parts(history, backend, kv, profiling))
    messages.append({"role": "user", "content": "".join(parts), "parts": parts})
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


