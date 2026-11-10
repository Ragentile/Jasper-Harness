#!/usr/bin/env python3
"""Derive vLLM serving arguments from a model configuration.

The helper selects text-only serving for multimodal models, uses PIECEWISE CUDA
graphs for hybrid linear-attention models, bounds the requested context window,
and validates tensor-parallel head divisibility. Check these settings against the
vLLM version and hardware used for deployment.

    python3 vllm_args.py ./model --tp 2 --window 32768

Arguments are written to stdout; explanatory notes are written to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_config(model_dir: str) -> dict:
    f = Path(model_dir) / "config.json"
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def has_field(name: str) -> bool:
    """Does this vLLM version know this engine argument?

    (Older ones have no language_model_only.)
    """
    try:
        import dataclasses

        from vllm import AsyncEngineArgs

        return name in {f.name for f in dataclasses.fields(AsyncEngineArgs)}
    except Exception:  # noqa: BLE001  vllm missing/broken: treat as absent, let serve complain
        return False


def extra_args(cfg: dict, tp: int, want_len: int = 0) -> tuple[list[str], list[str]]:
    """Return (flags to add, human-readable notes)."""
    args: list[str] = []
    notes: list[str] = []
    text_cfg = cfg.get("text_config") or {}

    # (1) multimodal -> language tower only
    if any(k in cfg for k in ("vision_config", "audio_config", "image_token_id")):
        if has_field("language_model_only"):
            args += ["--language-model-only"]
        else:
            args += ["--limit-mm-per-prompt", '{"image":0,"video":0}']
        notes.append("multimodal model: enable text-only serving")

    # (2) hybrid linear attention -> PIECEWISE
    layer_types = cfg.get("layer_types") or text_cfg.get("layer_types") or []
    if any("linear_attention" in str(t) or "mamba" in str(t) for t in layer_types):
        args += ["--compilation-config", '{"cudagraph_mode":"PIECEWISE"}']
        notes.append("hybrid linear attention: use cudagraph_mode=PIECEWISE")

    # (3) window: must fit prompt+generation, must not exceed the model own ceiling
    if want_len:
        limit = (text_cfg.get("max_position_embeddings")
                 or cfg.get("max_position_embeddings") or 0)
        use = min(want_len, limit) if limit else want_len
        args += ["--max-model-len", str(use)]
        if limit and want_len > limit:
            notes.append(f"window {want_len} exceeds the model ceiling {limit}; clamped to {use}")
        else:
            notes.append(f"window --max-model-len {use}")

    # (4) TP sanity
    heads = text_cfg.get("num_attention_heads") or cfg.get("num_attention_heads")
    if heads and tp > 1 and heads % tp:
        print(f"[vllm_args] TP={tp} does not divide {heads} attention heads -- pick a TP that does",
              file=sys.stderr)
        raise SystemExit(2)
    for key in ("linear_num_key_heads", "linear_num_value_heads"):
        n = text_cfg.get(key) or cfg.get(key)
        if n and tp > 1 and n % tp:
            print(f"[vllm_args] TP={tp} does not divide {key}={n} (GDN layer) -- pick a TP that does",
                  file=sys.stderr)
            raise SystemExit(2)
    return args, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--window", type=int, default=0,
                    help="window for prompt+generation (usually 2x max_tokens); clamped to the model ceiling")
    a = ap.parse_args()
    args, notes = extra_args(load_config(a.model), a.tp, a.window)
    for n in notes:
        print(f"[vllm_args] {n}", file=sys.stderr)
    print(" ".join(f"'{x}'" if x.startswith("{") else x for x in args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
