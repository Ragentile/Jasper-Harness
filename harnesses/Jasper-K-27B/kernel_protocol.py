"""Prompt templates, response parsing, and evaluator feedback for Jasper-K.

These utilities define the model-facing protocol used by the inference harness.
Evaluator responses are normalized before being rendered as refinement feedback.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)   # library_calls_allowed warns through it


# Strip compiler command echoes before extracting diagnostics. Long ninja commands
# contain no diagnostic information and would exhaust MAX_ERROR_LEN.
_CMDLINE_NOISE = re.compile(
    r"^\s*(?:\[\d+/\d+\]\s*)?(?:c\+\+|cc1plus|gcc|g\+\+|nvcc|ninja|ld)\s"
    r"|^\s*-{1,2}[\w-]"
    r"|-isystem\s|-I\s?/|--compiler-options"
)


# Remove randomized temporary build-directory prefixes from compiler diagnostics.
_TMP_PATH = re.compile(r"^\S*?/(?=[\w.-]+:\d+:\d+:)")


# Diagnostic tiers, ordered from most specific to most general. Use only the first
# matching tier so generic Python errors do not hide useful compiler diagnostics.
_DIAG_TIERS: list[re.Pattern] = [
    # 0. Structured evaluator verdicts have already been filtered.
    re.compile(r"^.*?\b(?:Correctness FAIL|Shape mismatch|Output mismatch|"
               r"Evaluation timed out|detected as cheating|remote eval unreachable)\b.*$"),
    # 1. gcc/nvcc/clang file:line:column diagnostics identify the actual source error.
    re.compile(r"^.*?:\d+:\d+:\s*(?:fatal\s+)?error:.*$"),
    # 2. Triton compilation and runtime errors.
    re.compile(r"^.*?\b(?:triton\.\w+\.errors\.\w+|CompilationError|"
               r"module 'triton\.language' has no attribute)\b.*$"),
    # 3. Other error lines, after command echoes have been removed.
    re.compile(r"^.*?\berror\s*:\s*\S.*$", re.IGNORECASE),
    # 4. General Python exceptions as a fallback. Preserve evaluator prefixes such as
    #    "Compilation failed:" and "Model init failed:" because they identify the stage.
    re.compile(r"\b\w+(?:Error|Exception):\s*\S"),
]


# Limit diagnostics per tier; repeated errors from one typo should not consume
# the entire feedback budget.
MAX_DIAG_LINES = 3


# Maximum lines of error context to include
MAX_ERROR_LINES = 10


# Maximum error message length for feedback
MAX_ERROR_LEN = 800


# Maximum characters of a failed kernel retained in the refinement prompt.
# The previous 2,000-character cap truncated most kernels. An 8,000-character
# cap preserves nearly all observed kernels with a modest context increase.
# This limit applies to input history, not generated output or format scoring.
MAX_KERNEL_HISTORY_LEN = 8000


# Maximum summary length carried across turns (context control)
MAX_SUMMARY_CHARS = 800


def _extract_key_error(error_message: str | None) -> str:
    """Extract the most useful diagnostic lines from evaluator errors.

    Remove compiler command echoes, search diagnostic tiers from most specific
    to most general, and keep up to MAX_DIAG_LINES unique lines from the first
    matching tier. This prevents long build commands and generic wrapper errors
    from hiding the actual compiler diagnostic.
    """
    if not error_message:
        return ""

    lines = [ln.rstrip() for ln in error_message.split("\n")
             if ln.strip() and not _CMDLINE_NOISE.search(ln)]

    for tier in _DIAG_TIERS:
        picked: list[str] = []
        seen: set[str] = set()
        n_hit = 0
        for ln in lines:
            if not tier.search(ln):
                continue
            n_hit += 1
            # Keep the source location and error, without a randomized build-directory prefix.
            text = _TMP_PATH.sub("", ln.strip())
            if text in seen:
                continue
            seen.add(text)
            if len(picked) < MAX_DIAG_LINES:
                picked.append(text)
        if picked:
            out = "\n".join(picked)
            extra = n_hit - len(picked)
            if extra > 0:
                out += f"\n... (+{extra} more)"
            return out[:MAX_ERROR_LEN]

    # If no tier matches, retain meaningful lines and omit traceback boilerplate.
    meaningful_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("Traceback"):
            continue
        if line.startswith('File "') and ", line " in line:
            continue
        if line.startswith("During handling of"):
            continue
        meaningful_lines.append(line)
        if len(meaningful_lines) >= MAX_ERROR_LINES:
            break

    if meaningful_lines:
        return "\n".join(meaningful_lines)[:MAX_ERROR_LEN]

    return error_message[:MAX_ERROR_LEN]


# Recognize evaluator timeouts from the error text. A candidate that never finishes
# is a candidate failure, not an infrastructure error. Keep this expression
# aligned with the evaluator message.
_TIMEOUT_MSG = re.compile(r"\bEvaluation timed out\b", re.IGNORECASE)


# A truncated turn produced no complete kernel. Reusing a partial reasoning draft
# as if it were submitted code would mislead the next turn.
TRUNCATED_KERNEL_PLACEHOLDER = (
    "# (no code: this response was cut off at the length limit before any code was written)"
)


def _truncated_eval_result() -> "KernelEvalResult":
    """Represent a truncated response that was not submitted for evaluation.

    The extra truncated=True flag lets _categorize_error distinguish this case.
    """
    return {
        "format_ok": False, "compiled": False, "correctness": False,
        "tests_passed": 0, "tests_total": 0, "speedup": None, "runtime_ms": None,
        "baseline_runtime_ms": None, "cheated": False, "error_message": None,
        "code_length": 0, "metadata": {}, "truncated": True,  # type: ignore[typeddict-unknown-key]
    }


def _categorize_error(eval_result: "KernelEvalResult") -> str:
    """Categorize a result for feedback.

    Infrastructure failures, generation truncation, rule violations, and evaluation
    timeouts are checked before compilation status. Those cases can leave compiled
    false without indicating a compiler error. Remaining categories distinguish
    formatting, compilation, runtime, correctness, performance, and success.
    """
    if eval_result.get("infra_error"):
        return "infra_error"
    if eval_result.get("truncated"):
        return "truncated"
    if eval_result.get("cheated"):
        return "cheating"
    if _TIMEOUT_MSG.search(eval_result.get("error_message") or ""):
        return "timeout"
    if not eval_result["format_ok"]:
        return "format_error"
    if not eval_result["compiled"]:
        return "compilation_error"
    if not eval_result["correctness"]:
        # Check if it's a runtime crash vs wrong output
        error_msg = eval_result.get("error_message", "") or ""
        if "Error" in error_msg or "Exception" in error_msg:
            return "runtime_error"
        return "correctness_error"
    # Correct - check performance
    if eval_result.get("speedup") is not None and eval_result["speedup"] < 1.0:
        return "performance_warning"
    return "success"


# Shared feedback for a response truncated at the generation limit.
TRUNCATED_FEEDBACK = (
    "Your previous answer was cut off at the maximum generation length before it "
    "produced any complete code: your reasoning ran too long. Reason more concisely, "
    "commit to one approach instead of going back and forth, and make sure to write out "
    "the complete code within the length limit."
)


# The rules already appear in the system prompt. Refer back to them and let the
# error message identify the violation instead of repeating the full rules
# in every history entry. The no-library variant adds a concrete repair step.
_CHEAT_RULES_REF = "Your previous attempt scored ZERO: it broke one of the Rules in the system prompt."


_CHEAT_RULES_REF_FORBIDDEN = " Replace each operator named in the error message with your own kernel code."


def _get_error_guidance(error_category: str, backend: str, cheat: str = "full") -> str:
    """Provide guidance consistent with the evaluator's rules.

    Library calls are allowed only alongside a real custom kernel unless
    JASPER_K_ALLOW_LIBRARY_CALLS=0, which requires custom kernels for all computation.
    Keep this text aligned with the system rules and backend checks.

    cheat selects full guidance or a reference to the system rules ("rules_ref").
    """
    libcall = ("Library calls (PyTorch, cuBLAS/cuDNN) are allowed alongside it but do not "
               "count as your kernel. ")
    stream = "(a plain `kernel<<<grid, block>>>` launch is fine): "
    if error_category == "cheating" and cheat == "rules_ref":
        return _CHEAT_RULES_REF + ("" if library_calls_allowed() else _CHEAT_RULES_REF_FORBIDDEN)
    guidance = {
        "truncated": TRUNCATED_FEEDBACK,
        "infra_error": (
            "The previous attempt could NOT be evaluated (evaluation service failure, "
            "not a problem with your code). Keep your previous approach and continue "
            "improving it."
        ),
        # The full-feedback variant must state the rules explicitly. Library calls are
        # allowed only alongside a real custom kernel; bypasses remain violations.
        # Describe the rule without prescribing a particular optimization strategy.
        "cheating": (
            f"Your previous attempt scored ZERO. You must ship a real {backend.upper()} "
            f"kernel and let it do work: shipping no {backend.upper()} kernel at all, "
            f"using `try`/`except` or `pass` to avoid implementing something, patching the "
            f"timing, or returning a cached result instead of computing one all score zero. "
            f"A `__global__` kernel that is never launched does not count. {libcall}"
            f"`nn.Module`, `nn.Parameter` and `torch.nn.init.*` are fine, and reading "
            f"`self.conv.weight` to feed your own kernel is the intended way to use them. "
            f"Run everything on the current CUDA stream {stream}"
            f"creating or switching to another stream -- `torch.cuda.Stream`, "
            f"`with torch.cuda.stream(...)`, `cudaStreamCreate`, `getStreamFromPool`, ... -- "
            f"also scores zero. "
            f"The error line above names exactly what fired."
        ) if library_calls_allowed() else (
            # When JASPER_K_ALLOW_LIBRARY_CALLS=0, every computation must use a custom kernel.
            f"Your previous attempt scored ZERO. The rule: EVERY computation in "
            f"`ModelNew` must be performed by your own {backend.upper()} kernel. Calling "
            f"a PyTorch operator to do the work instead -- `torch.matmul`, `torch.softmax`, "
            f"`torch.clamp`, `F.relu`, `F.pad`, ... -- scores zero even when the output is "
            f"numerically correct and fast, and so does using `try`/`except` or `pass` to "
            f"avoid implementing something, shipping no {backend.upper()} kernel at all, or "
            f"returning a cached result instead of computing one. The same applies to "
            f"`nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d` and every other `torch.nn` compute "
            f"layer: holding one to store weights is fine, but calling it -- `self.conv(x)`, "
            f"including through an `nn.Sequential` or an `nn.ModuleList` you iterate over -- "
            f"is cuDNN/cuBLAS doing the work, not you. Same for handing it to a library from "
            f"C++: `cublasSgemm`, `cudnnConvolutionForward`, `at::conv2d`, `at::matmul`, ... "
            f"-- an unused `__global__` kernel next to one of those still scores zero. "
            f"`nn.Module`, `nn.Parameter` and `torch.nn.init.*` are allowed -- they only hold "
            f"and initialise weights, and reading `self.conv.weight` to feed your own kernel "
            f"is the intended way to use them. "
            f"Run everything on the current CUDA stream {stream}"
            f"creating or switching to another stream -- `torch.cuda.Stream`, "
            f"`with torch.cuda.stream(...)`, `cudaStreamCreate`, `getStreamFromPool`, ... -- "
            f"also scores zero. "
            f"The error line above names exactly what fired: replace each listed operator "
            f"with your own kernel code."
        ),
        "timeout": (
            f"The kernel did not finish within the evaluation time limit. Look for a "
            f"loop whose bound depends on data (or never decreases), a grid/block "
            f"configuration that launches far too much work, or a synchronization that "
            f"never completes. Simplify to a correct, straightforward {backend.upper()} "
            f"kernel first, then optimize."
        ),
        "format_error": (
            "Ensure your output is a valid Python class named `ModelNew` "
            "wrapped in a code block."
        ),
        "compilation_error": (
            f"Fix the {backend.upper()} syntax/API errors. Check that all "
            f"kernel functions are correctly decorated and all imports are valid."
        ),
        "runtime_error": (
            "Fix the runtime error. Common issues: shape mismatches, "
            "incorrect tensor indexing, or invalid kernel launch configurations."
        ),
        "correctness_error": (
            "The kernel runs but produces incorrect output. Check your algorithm "
            "implementation, boundary conditions, and reduction operations."
        ),
        "performance_warning": (
            "The kernel is correct but slower than PyTorch. Consider optimizing "
            "memory access patterns, tiling, or parallelization strategy. Modify the "
            "working kernel incrementally instead of rewriting it -- verify each change "
            "preserves correctness."
        ),
        # Successful candidates need optimization guidance, not an instruction to fix
        # nonexistent errors. Preserve the option to resubmit a correct implementation
        # when no reliable improvement is available.
        "success": (
            "Your kernel is CORRECT with the speedup shown above. Do NOT rewrite it -- "
            "make it faster with small, targeted changes: improve memory coalescing, "
            "use shared memory tiling, fuse remaining operations, reduce synchronization, "
            "or tune the launch configuration. Change one thing at a time and verify it "
            "preserves correctness; a correct kernel is worth far more than a fast broken "
            "one. If you have no confident optimization left, resubmit the previous "
            "kernel unchanged."
        ),
    }
    return guidance.get(error_category, "")


# Profiler summaries become compact feedback when eval_profiling=True. Report
# kernel time shares, launches per forward pass, and GPU utilization.
# Use neutral measurements: ownership labels can misclassify custom kernels,
# and allocator or harness overhead may be outside the candidate's control.
# Avoid absolute profiler timings that use a different basis from runtime_ms.
PROFILE_TOP_KERNELS = 5


PROFILE_KERNEL_NAME_CHARS = 60


def _short_kernel_name(name: str) -> str:
    """Shorten a device kernel name for readable feedback.

    Remove the return type, template arguments, parameter list, and long
    namespaces. Names may already be truncated inside a template, so split at
    the first '<' or '(' instead of trying to balance delimiters.
    """
    s = name.strip().replace("(anonymous namespace)::", "")
    if s.startswith("void "):
        s = s[5:]
    cut = min((i for i in (s.find("<"), s.find("(")) if i >= 0), default=len(s))
    parts = [p for p in s[:cut].strip().split("::") if p]
    if not parts:
        return name[:PROFILE_KERNEL_NAME_CHARS]
    # Include the preceding namespace for generic names such as cutlass::Kernel.
    short = "::".join(parts[-2:]) if len(parts) > 1 and parts[-1].lower().startswith("kernel") else parts[-1]
    if len(short) > PROFILE_KERNEL_NAME_CHARS:
        short = short[:PROFILE_KERNEL_NAME_CHARS] + "..."
    return short


def _profile_feedback(eval_result) -> str:
    """Render a compact profiler summary, or an empty string if unavailable."""
    md = eval_result.get("metadata") or {}
    summ = (md.get("profiling") or {}).get("events_summary") or {}
    gpu = summ.get("gpu") or {}
    if not gpu or summ.get("summary_error") or not gpu.get("kernel_launches"):
        return ""
    n = max(1, int(summ.get("num_trials") or 1))
    per = gpu.get("per_trial") or {}

    head = [f"{per.get('kernel_launches', gpu['kernel_launches'] / n):g} GPU kernel launches"]
    if gpu.get("busy_ratio") is not None:
        head.append(f"GPU busy {gpu['busy_ratio']:.0%} of wall time")
    lines = [
        "Profile of this kernel (torch.profiler, per forward pass):",
        "- " + " | ".join(head),
        "- Top GPU kernels by share of GPU kernel time:",
    ]
    top = (summ.get("kernels") or {}).get("top") or []
    shown = 0
    for k in top[:PROFILE_TOP_KERNELS]:
        share = k.get("share") or 0.0
        if shown and share < 0.01:
            break
        op = next((o["name"] for o in k.get("ops") or [] if o.get("name")), "")
        src = f" (from {op})" if op else ""
        lines.append(f"  - {_short_kernel_name(k.get('name', ''))}{src}: {share:.0%}, "
                     f"{k.get('count', 0) / n:g} launch(es)")
        shown += 1
    rest = sum(k.get("share") or 0.0 for k in top[shown:])
    others = (summ.get("kernels") or {}).get("others") or {}
    if gpu.get("kernel_time_us"):
        rest += (others.get("total_us") or 0.0) / gpu["kernel_time_us"]
    if rest >= 0.01:
        lines.append(f"  - (other kernels): {rest:.0%}")
    lines.append("(For reference only.)")
    return "\n".join(lines)


def _JASPER_K_feedback_line(eval_result, error_category: str, backend: str,
                         slow_note: bool = False, cheat: str = "full") -> str:
    """Render the frozen multi-turn feedback for an evaluation result.

    slow_note explicitly identifies speedups below 1x as slower than the reference.
    Historical versions omit that note. See _get_error_guidance for cheat modes.
    """
    def err_suffix() -> str:
        text = _extract_key_error(eval_result.get("error_message") or "")
        return f" Here is the error message: {text}" if text else ""

    if error_category in ("success", "performance_warning"):
        sp = eval_result.get("speedup")
        sp_text = f"{sp:.2f}x" if sp is not None else "(not measured)"
        line = (
            "Your previous answer was correct but can be made faster. "
            f"Here is the speedup you achieved relative to the baseline: {sp_text}"
        )
        if slow_note and sp is not None and sp < 1.0:
            line += " (below 1.00x: slower than the PyTorch reference)"
        return line
    if error_category == "format_error":
        return (
            "Your previous answer failed to be parsed due to not adhering to the "
            "desired formatting." + err_suffix()
        )
    if error_category == "compilation_error":
        return "Your previous answer failed to compile." + err_suffix()
    if error_category == "runtime_error":
        return (
            "Your previous answer compiled successfully but had runtime errors."
            + err_suffix()
        )
    if error_category == "correctness_error":
        return "Your previous answer was incorrect." + err_suffix()
    if error_category == "truncated":
        return TRUNCATED_FEEDBACK
    # Use category-specific guidance for cheating, timeout, and infrastructure errors.
    # Include relevant candidate diagnostics; omit infrastructure noise.
    line = _get_error_guidance(error_category, backend, cheat=cheat)
    if error_category in ("cheating", "timeout"):
        line += err_suffix()
    return line


def _truncate_kernel(
    kernel_code: str, max_len: int | None = MAX_KERNEL_HISTORY_LEN
) -> str:
    """Truncate failed-kernel history while preserving its beginning and signature.

    max_len=None retains the full code, as used for verified-correct kernels.
    """
    if max_len is None or len(kernel_code) <= max_len:
        return kernel_code

    # Try to keep the class signature and forward method
    truncated = kernel_code[:max_len]
    return truncated + "\n# ... (truncated)"


def _truncate_summary(summary: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    """Cap summary length to keep refinement context compact."""
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rstrip() + "..."


def _fallback_summary(eval_result: "KernelEvalResult", error_category: str) -> str:
    """Build a minimal summary when model did not emit <SUMMARY>."""
    if error_category == "truncated":
        return "Previous attempt ran out of length while still reasoning; no code was produced."
    if error_category == "success":
        speedup = eval_result.get("speedup")
        if speedup is not None:
            return f"Previous attempt compiled and passed all tests with speedup {speedup:.2f}x."
        return "Previous attempt compiled and passed all tests."
    if not eval_result["compiled"]:
        return "Previous attempt failed to compile; next turn should focus on fixing syntax/API issues."
    if not eval_result["correctness"]:
        return (
            f"Previous attempt compiled but only passed {eval_result['tests_passed']}/"
            f"{eval_result['tests_total']} correctness tests; next turn should fix output correctness."
        )
    return "Previous attempt needs refinement based on the reported evaluation feedback."


# Multi-turn feedback carries all attempts in chronological order, including
# code, reasoning summaries, and evaluation feedback. Correct kernels are
# retained in full; failed kernels are bounded by MAX_KERNEL_HISTORY_LEN.
# The current closing instruction permits either refinement or a new approach.
JASPER_K_ATTEMPT_TEMPLATE = """
## Previous Attempt (Turn {turn})

Your reasoning summary: {summary}

```python
{kernel}
```

Feedback: {feedback}
"""


# The current closing instruction permits both refinement and exploration.
# Best-of-turn scoring retains earlier correct kernels. Every new attempt
# must still provide a complete ModelNew implementation.
JASPER_K_REFINE_INSTRUCTION = (
    "Restart your reasoning process: build on your fastest correct kernel or try a "
    "different approach. Generate new, complete code."
)


# Prompt profile settings. Verify generated messages when changing these values.
@dataclass(frozen=True)
class JasperKVersion:
    rules_addendum: bool       # Append system rules and numerical guidance;
                               # remove duplicate instructions from the system prompt.
    strip_kb_conflicts: bool   # Remove task instructions that conflict with the rules or summary format.
    closing: str               # Closing instruction for refinement turns.
    slow_note: bool            # Explicitly identify correct implementations slower than the reference.
    cheat: str                 # Detailed guidance or a reference to the system rules.


JASPER_K_VERSIONS: dict[str, JasperKVersion] = {
    "Jasper-K": JasperKVersion(rules_addendum=True, strip_kb_conflicts=True,
                               closing=JASPER_K_REFINE_INSTRUCTION, slow_note=True,
                               cheat="rules_ref"),
}


DEFAULT_FEEDBACK_STYLE = "Jasper-K"


# Jasper-K system prompt without thinking tokens
MULTITURN_SYSTEM_PROMPT_NO_THINK = """You are an expert GPU kernel developer. Your task is to optimize PyTorch operations by writing efficient custom {backend} kernels.

When given a PyTorch model and optimization examples, write an optimized kernel implementation. If a previous attempt failed, fix the errors based on the feedback provided.

Your solution must:
- Be a drop-in replacement as a class named `ModelNew`
- Use custom {backend} kernels, not just PyTorch operations
- Be correct and produce the same results as the reference

You MUST respond in exactly this format:

<KERNEL>
```python
# Your complete optimized implementation here
class ModelNew(nn.Module):
    ...
```
</KERNEL>

<SUMMARY>
2-4 sentences describing:
- What optimization strategy you used this turn
- Key implementation decisions and tradeoffs
- What you would improve next if needed

Keep this section concise (<= 120 tokens). It will be provided in the next refinement turn.
</SUMMARY>"""


# Keep these rules consistent with the cheating feedback and the evaluator's
# library-call policy.
JASPER_K_RULES_LIBCALL_ALLOWED = """## Rules (violations score zero)
- Launch at least one real {backend} kernel of your own. Library calls (PyTorch, cuBLAS/cuDNN) are allowed but do not count as your kernel.
- No `try`/`except` or `pass` to skip work, no cached results, no timing patches, no Tensor subclasses or lazy evaluation, no threads or processes.
- Do not create or switch CUDA streams; launch kernels the plain way (`kernel<<<grid, block>>>(...)`).
{numerics}"""


JASPER_K_RULES_LIBCALL_FORBIDDEN = """## Rules (violations score zero)
- EVERY computation in `ModelNew` must be done by your own {backend} kernel: no PyTorch compute operators (`torch.matmul`, `F.relu`, ...), no calling `torch.nn` compute layers (`self.conv(x)`, also through `nn.Sequential`), no cuBLAS/cuDNN/`at::` library calls from C++. `nn.Module`, `nn.Parameter` and `torch.nn.init.*` are fine for holding weights.
- No `try`/`except` or `pass` to skip work, no cached results, no timing patches, no Tensor subclasses or lazy evaluation, no threads or processes.
- Do not create or switch CUDA streams; launch kernels the plain way (`kernel<<<grid, block>>>(...)`).
{numerics}"""


# Numerical guidance for the default Jasper-K prompt profile.
# Configure evaluator checks to match the precision requirements in the prompt.
JASPER_K_NUMERICS_FP32 = "- Correctness is checked tightly against the FP32 reference (tolerance ≈ the reference's own rounding error); lowering precision usually fails."


JASPER_K_NUMERICS_LOWP = "- Inputs, weights and the reference all run in {dtype}; correctness is checked at that precision."


def build_JASPER_K_system_addendum(backend: str, eval_dtype: str = "") -> str:
    """Append system rules, ending with numerical guidance after a blank line."""
    dtype_norm = (eval_dtype or "fp32").strip().lower()
    numerics = (JASPER_K_NUMERICS_FP32 if dtype_norm == "fp32"
                else JASPER_K_NUMERICS_LOWP.format(dtype=dtype_norm.upper()))
    rules = (JASPER_K_RULES_LIBCALL_ALLOWED if library_calls_allowed()
             else JASPER_K_RULES_LIBCALL_FORBIDDEN)
    return "\n\n" + rules.format(backend=backend.upper(), numerics=numerics)


# Remove KernelBench's output-only instructions that conflict with the required
# <KERNEL>/<SUMMARY> format. Preserve the remainder of the task statement.
_KB_OUTPUT_CONFLICTS = (
    (" Output the new code in codeblocks.", ""),
    (" Just output the new model code, no other text, and NO testing code!",
     " Do not include any testing code."),
)


# When library calls are disabled, remove the task text allowing operators to
# remain unchanged, because it conflicts with the custom-kernel rule.
_KB_FREEDOM_CONFLICT = re.compile(
    r"You have complete freedom to choose the set of operators you want to replace\. "
    r"You may make the decision to replace some operators with custom [^.]*? and leave "
    r"others unchanged\. ")


def strip_kb_output_conflicts(prompt: str) -> str:
    for old, new in _KB_OUTPUT_CONFLICTS:
        prompt = prompt.replace(old, new)
    if not library_calls_allowed():
        prompt = _KB_FREEDOM_CONFLICT.sub("", prompt, count=1)
    return prompt


# When the rules addendum is present, remove duplicate custom-kernel and
# correctness instructions from the system template.
_SYSTEM_MUST_OLD = """Your solution must:
- Be a drop-in replacement as a class named `ModelNew`
- Use custom {backend} kernels, not just PyTorch operations
- Be correct and produce the same results as the reference
"""


_SYSTEM_MUST_NEW = "Your solution must be a drop-in replacement as a class named `ModelNew`.\n"


def slim_system_prompt(system_prompt: str, backend: str) -> str:
    old = _SYSTEM_MUST_OLD.format(backend=backend.upper())
    assert old in system_prompt, "system template changed: update _SYSTEM_MUST_OLD"
    return system_prompt.replace(old, _SYSTEM_MUST_NEW)


# Control the library-call rule in prompts and feedback.
_LIBCALL_ENV = "JASPER_K_ALLOW_LIBRARY_CALLS"


def library_calls_allowed() -> bool:
    """Return whether library calls are allowed by JASPER_K_ALLOW_LIBRARY_CALLS."""
    raw = (os.environ.get(_LIBCALL_ENV) or "").strip().lower()
    if raw in ("0", "false", "no"):
        return False
    if raw not in ("", "1", "true", "yes"):
        logger.warning(
            f"[kernel-rules] {_LIBCALL_ENV}={raw!r} is not 0/1; treating it as 1 (library calls allowed)"
        )
    return True


# Regex patterns for parsing structured output (Jasper-K/Qwen3 style)
# Thinking block patterns - supports <think>, <thinking>, <THOUGHT> variants
THINKING_PATTERN = re.compile(
    r"<(?:think|thinking|THOUGHT)>(.*?)</(?:think|thinking|THOUGHT)>",
    re.DOTALL | re.IGNORECASE
)


# Closing tag alone -- for completions whose opening tag lives in the chat
# template (see parse_structured_response).
THINK_CLOSE_PATTERN = re.compile(r"</(?:think|thinking|THOUGHT)>", re.IGNORECASE)


# Kernel block pattern - code inside <KERNEL>...</KERNEL>
KERNEL_BLOCK_PATTERN = re.compile(
    r"<KERNEL>\s*```(?:cuda|python|cpp)?\s*\n?(.*?)```\s*</KERNEL>",
    re.DOTALL | re.IGNORECASE
)


# Fallback: just <KERNEL>...</KERNEL> without fenced code block
KERNEL_BLOCK_SIMPLE_PATTERN = re.compile(
    r"<KERNEL>(.*?)</KERNEL>",
    re.DOTALL | re.IGNORECASE
)


# Summary block pattern - concise reasoning carried across refinement turns
SUMMARY_BLOCK_PATTERN = re.compile(
    r"<SUMMARY>(.*?)</SUMMARY>",
    re.DOTALL | re.IGNORECASE
)


_FENCED_BLOCK_PATTERN = re.compile(r"```[\w+-]*[ \t]*\n(.*?)```", re.DOTALL)


def _find_modelnew_block(text: str) -> tuple[str, int] | None:
    """Last fenced code block containing ``class ModelNew`` -> (code, end offset)."""
    hit = None
    for m in _FENCED_BLOCK_PATTERN.finditer(text):
        if "class ModelNew" in m.group(1):
            hit = m
    if hit is None:
        return None
    return hit.group(1).strip(), hit.end()


# In a bare-code answer, whether a column-0 line "looks like Python" (lines inside
# triple-quoted strings are handled separately):
_CODE_LINE = re.compile(
    r"^(?:import\s|from\s|class\s|def\s|async\s+def\s|@|#|if\s|for\s|while\s|try\b|with\s|"
    r"return\b|raise\b|assert\s|del\s|global\s|print\(|pass\b|"
    r"[A-Za-z_][\w.]*\s*(?:[,=(\[.]|:\s*[A-Za-z_\[])|[)\]}]|\"\"\"|\'\'\')"
)


def _split_unfenced_answer(text: str) -> tuple[str, str]:
    """Bare-code answer -> (kernel, trailing summary).

    Code starts at the first import/from/class/def/@ line; triple-quoted strings (the
    CUDA source for load_inline) are skipped whole; the first column-0 line after that
    which does not look like Python ends the code, everything after it is the summary.
    A run of comment-only lines at the end of the code (QwQ writes "# Summary of
    changes: ...") is also peeled off into the summary with the leading '#' removed.
    """
    lines = text.split("\n")
    start = next(
        (i for i, l in enumerate(lines) if re.match(r"^(import |from |class |def |@)", l)),
        None,
    )
    if start is None:
        return text.strip(), ""
    in_str = False
    end = len(lines)
    for i in range(start, len(lines)):
        l = lines[i]
        n_triple = l.count('"""') + l.count("\'\'\'")
        if in_str:
            if n_triple % 2 == 1:
                in_str = False
            continue
        if n_triple % 2 == 1:
            in_str = True
            continue
        if not l.strip() or l[0] in " \t":
            continue
        if not _CODE_LINE.match(l):
            end = i
            break
    code_lines = lines[start:end]
    # trailing comment-only lines -> summary
    k = len(code_lines)
    while k > 0 and (not code_lines[k - 1].strip() or code_lines[k - 1].lstrip().startswith("#")):
        k -= 1
    tail_comments = [l.lstrip().lstrip("#").strip() for l in code_lines[k:] if l.strip()]
    prose = "\n".join(lines[end:]).strip()
    summary = "\n".join([*tail_comments, prose] if prose else tail_comments).strip()
    return "\n".join(code_lines[:k]).strip(), summary


def _clean_summary(text: str) -> str:
    """Clean a free-form summary: drop stray think tags (QwQ sometimes emits a second
    </think> in the body) and surrounding whitespace.
    """
    return THINK_CLOSE_PATTERN.sub("", text).replace("<think>", "").strip()


def _strip_fences(text: str) -> str:
    """Remove leftover fenced code blocks (a summary should not carry code)."""
    return _FENCED_BLOCK_PATTERN.sub("", text)


def extract_code_block(text: str, languages: list[str] | None = None) -> str | None:
    """Extract the first code block from text.

    Args:
        text: The text containing code blocks
        languages: Optional list of language tags to look for (e.g., ["python", "cuda"])

    Returns:
        The extracted code or None if no valid code block found
    """
    if languages is None:
        languages = ["python", "cuda", "cpp", ""]

    # Try to find fenced code blocks
    for lang in languages:
        if lang:
            pattern = rf"```{lang}\s*\n(.*?)```"
        else:
            pattern = r"```\s*\n(.*?)```"

        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

    # If no fenced block, try to find code that looks like a Python module
    # (contains class definitions, imports, etc.)
    if "class ModelNew" in text or "def forward" in text:
        # Try to extract just the code portion
        lines = text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith(("import ", "from ", "class ", "def ", "@")):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines)

    return None


@dataclass
class ParsedResponse:
    """Parsed model response with thinking and kernel blocks."""
    thought: str  # Content from <think>/<THOUGHT> block (may be empty)
    thought_summary: str  # Content from <SUMMARY> block (may be empty)
    kernel: str   # Kernel code (from <KERNEL> block or extracted code block)
    raw: str      # Original raw response
    format_ok: bool  # Whether we successfully extracted kernel code


def parse_structured_response(text: str) -> ParsedResponse:
    """Parse model response with Jasper-K/Qwen3 structured format.

    Expected format:
        <think>
        Brief reasoning about optimization approach...
        </think>

        <KERNEL>
        ```cuda
        // CUDA kernel code
        ```
        </KERNEL>

        <SUMMARY>
        Brief summary for next refinement turn...
        </SUMMARY>

    Also handles:
    - <thinking>...</thinking> and <THOUGHT>...</THOUGHT> variants
    - Missing thinking block (thought will be empty string)
    - Missing <KERNEL> tags (falls back to extract_code_block)
    - Plain code without any tags

    Args:
        text: Raw model output

    Returns:
        ParsedResponse with thought, kernel, raw text, and format_ok flag
    """
    raw = text
    thought = ""
    thought_summary = ""
    kernel = ""

    # Extract thinking block (optional)
    think_match = THINKING_PATTERN.search(text)
    if think_match:
        thought = think_match.group(1).strip()
        # Remove thinking block from text for kernel extraction
        text = THINKING_PATTERN.sub("", text).strip()
    else:
        # Reasoning models (Qwen3.x/QwQ) open the think block inside the chat
        # template, so the completion carries ONLY a closing </think> -- the
        # paired pattern above never matches and the reasoning stays in `text`.
        # That reasoning routinely REHEARSES the required format (literally
        # writing "<SUMMARY>\n...\n</SUMMARY>"), and a first-match search then
        # returns the rehearsed placeholder instead of the real block after the
        # kernel (Qwen3.6-35B-A3B, 0827: 39% of eval turns got summary="...",
        # 16% even got kernel="..."). Treat everything before the first closing
        # tag as thought and cut it away before any tag extraction.
        close_match = THINK_CLOSE_PATTERN.search(text)
        if close_match:
            thought = text[: close_match.start()].strip()
            text = text[close_match.end() :].strip()

    # Extract summary block (optional)
    summary_match = SUMMARY_BLOCK_PATTERN.search(text)
    if summary_match:
        thought_summary = summary_match.group(1).strip()
        # Remove summary block from text for kernel extraction
        text = SUMMARY_BLOCK_PATTERN.sub("", text).strip()

    # Try to extract kernel from <KERNEL> block
    kernel_match = KERNEL_BLOCK_PATTERN.search(text)
    if kernel_match:
        kernel = kernel_match.group(1).strip()
    else:
        # Try simple <KERNEL>...</KERNEL> without fenced code
        kernel_match = KERNEL_BLOCK_SIMPLE_PATTERN.search(text)
        if kernel_match:
            kernel = kernel_match.group(1).strip()
            # Try to extract code from within if it has fences
            inner_code = extract_code_block(kernel)
            if inner_code:
                kernel = inner_code

    # Fallback: no <KERNEL> tags. Free-form answer ("code block + a few sentences"):
    # the kernel is the **last** fenced block containing class ModelNew (earlier blocks
    # are usually drafts), the summary is the text after that block (when there is no
    # <SUMMARY> tag). Bare-code answers are split structurally; anything else falls back
    # to the old "first code block" extraction.
    if not kernel:
        found = _find_modelnew_block(text)
        if found is not None:
            kernel, block_end = found
            if not thought_summary:
                thought_summary = _clean_summary(_strip_fences(text[block_end:]))
        elif "```" not in text and "class ModelNew" in text:
            # Bare-code answer (Jasper-K's example has no fences and QwQ copies that): split
            # by Python structure, prose after the code / a trailing comment block is the
            # summary. The old extract_code_block would stuff the prose into the kernel
            # -> a SyntaxError blamed on the model.
            kernel, trailing = _split_unfenced_answer(text)
            if not thought_summary:
                thought_summary = _clean_summary(trailing)
        else:
            kernel = extract_code_block(text) or ""

    # No fallback - malformed responses should fail with format_ok=False
    # This incentivizes models to use proper <KERNEL>```...```</KERNEL> format

    # Check if we got valid kernel code
    format_ok = bool(kernel) and ("class ModelNew" in kernel or "def forward" in kernel)

    return ParsedResponse(
        thought=thought,
        thought_summary=thought_summary,
        kernel=kernel,
        raw=raw,
        format_ok=format_ok,
    )

