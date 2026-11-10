"""Two-stage thinking budgets for an OpenAI-compatible completion service.

Continuation uses server-returned token IDs. The closing instruction counts toward
the configured total generation budget.
"""

FORCE_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)


def validate_config(cfg):
    sampling = cfg.get("sampling", {})
    value = sampling.get("think_budget", 0)
    if type(value) is not int or value < 0:
        raise ValueError("sampling.think_budget must be a non-negative integer")
    if not value:
        return 0
    total = sampling.get("max_tokens")
    if type(total) is not int or not 0 < value < total:
        raise ValueError("think_budget must be smaller than sampling.max_tokens")
    model = cfg.get("model", {})
    if model.get("enable_thinking", True) is not True:
        raise ValueError("think_budget requires model.enable_thinking = true")
    if not model.get("serve", {}).get("path"):
        raise ValueError("think_budget requires the model's tokenizer directory")
    return value


def contains(tokens, sequence):
    return any(tokens[i:i + len(sequence)] == sequence
               for i in range(len(tokens) - len(sequence) + 1))


class ThinkingBudgetExhausted(ValueError):
    """Too little response space for thinking, its closing hint, and an answer."""


class ThinkingBudget:
    def __init__(self, cfg, tokenizer):
        self.budget = validate_config(cfg)
        if not self.budget:
            raise ValueError("ThinkingBudget requires an enabled budget")
        self.total = cfg["sampling"]["max_tokens"]
        self.tokenizer = tokenizer
        self.force_ids = tokenizer.encode(FORCE_TEXT, add_special_tokens=False)
        self.end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if not self.end_ids or not contains(self.force_ids, self.end_ids):
            raise ValueError("tokenizer does not preserve the thinking end marker")
        if self.budget + len(self.force_ids) >= self.total:
            raise ValueError("max_tokens must leave room for the hint and an answer")

    @staticmethod
    def unpack(response, limit):
        choices = response.get("choices", [])
        if len(choices) != 1:
            raise RuntimeError("thinking budget requires exactly one completion")
        choice = choices[0]
        ids = choice.get("token_ids")
        if (not isinstance(ids, list)
                or any(type(t) is not int or t < 0 for t in ids)):
            raise RuntimeError("vLLM must return actual completion token_ids")
        if len(ids) > limit:
            raise RuntimeError("vLLM exceeded the requested token budget")
        count = response.get("usage", {}).get("completion_tokens")
        if count is not None and count != len(ids):
            raise RuntimeError("completion token_ids disagree with usage")
        finish = choice.get("finish_reason")
        if finish not in ("stop", "length"):
            raise RuntimeError(f"unexpected completion finish_reason: {finish!r}")
        if finish == "length" and len(ids) != limit:
            raise RuntimeError("completion reached a context limit before its token budget")
        return choice, ids, finish

    def generate(self, prompt, params, request, record, log=None):
        """request(payload) owns HTTP timeout/deadline checks; record survives errors."""
        total = params.get("max_tokens")
        if (type(total) is not int or not 0 < total <= self.total
                or params.get("n", 1) != 1):
            raise ValueError("request parameters do not match the thinking budget")
        # Tony counts the complete rendered prompt against its context window.
        # Reserve the forced closing hint and at least one answer token, even when
        # history reduces this turn's remaining response allowance.
        first_limit = min(self.budget, total - len(self.force_ids) - 1)
        if first_limit < 1:
            raise ThinkingBudgetExhausted(
                f"only {total} response tokens remain; no room for thinking, "
                "the closing hint and an answer")
        record.update(think_budget=self.budget, effective_think_budget=first_limit,
                      max_tokens=total,
                      stage1_tokens=0, injected_tokens=0, stage2_tokens=0,
                      total_tokens=0, forced=False, continued=False, status="stage1")
        first = request(params | {"prompt": prompt, "max_tokens": first_limit,
                                  "return_token_ids": True, "stop": ["<|im_end|>"]})
        ch1, ids1, finish1 = self.unpack(first, first_limit)
        record.update(stage1_tokens=len(ids1), total_tokens=len(ids1))
        if finish1 != "length":
            record.update(status="complete", finish_reason=finish1)
            if log:
                log(f"think_budget: natural finish, {len(ids1)} tokens")
            return ch1.get("text") or "", finish1

        prompt_ids = ch1.get("prompt_token_ids")
        if prompt_ids is None:
            prompt_ids = first.get("prompt_token_ids")
        if (not isinstance(prompt_ids, list) or not prompt_ids
                or any(type(t) is not int or t < 0 for t in prompt_ids)):
            raise RuntimeError("vLLM must return actual prompt_token_ids for continuation")
        injected = [] if contains(ids1, self.end_ids) else self.force_ids
        remaining = total - len(ids1) - len(injected)
        record.update(injected_tokens=len(injected), forced=bool(injected),
                      continued=True, status="stage2", total_tokens=len(ids1) + len(injected))
        if log:
            log(f"think_budget: first={len(ids1)}, forced={bool(injected)}, "
                f"injected={len(injected)}, answer allowance={remaining}")
        second = request(params | {"prompt": prompt_ids + ids1 + injected,
                                   "max_tokens": remaining, "return_token_ids": True,
                                   "add_special_tokens": False, "stop": ["<|im_end|>"]})
        _, ids2, finish2 = self.unpack(second, remaining)
        all_ids = ids1 + injected + ids2
        record.update(stage2_tokens=len(ids2), total_tokens=len(all_ids),
                      status="complete", finish_reason=finish2)
        if log:
            log(f"think_budget: second={len(ids2)}, total={len(all_ids)}/{total}, "
                f"finish={finish2}")
        # Decode together so a boundary splitting a UTF-8 character remains lossless.
        # This checkpoint's </think> is retained by skip_special_tokens=True, as in vLLM.
        return self.tokenizer.decode(all_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False), finish2
