"""verdict-shell-safety: local, offline shell-command safety classification.

Parse-then-classify pipeline: every command is parsed with bashlex into a
flat serialized AST, then scored by a 151M-parameter ModernBERT binary
classifier (ONNX). Fail-closed on parse errors and empty input.

Weights are hosted on the Hugging Face Hub
(aialchemist-dev/verdict-shell-safety) and downloaded on first use.
"""

import os
from pathlib import Path

THRESHOLD = 0.50
MAX_LENGTH = 128
HF_REPO = "aialchemist-dev/verdict-shell-safety"
MODEL_SHA256 = "01cf7a82f811aec46d5c264b0d21f9ccc2ad65b3ee4943714884e3aca061e81d"

_cache_dir = Path(os.environ.get("VERDICT_CACHE_DIR", Path.home() / ".cache" / "verdict-shell-safety"))
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from .predictor import VerdictPredictor

        _cache_dir.mkdir(parents=True, exist_ok=True)
        _predictor = VerdictPredictor(cache_dir=_cache_dir)
    return _predictor


def classify(command: str) -> dict:
    """Score a shell command. Returns dict with command, parsed, score,
    decision ("BLOCK"/"PASS"), and fail_closed (bool)."""
    return _get_predictor().predict(command)


def is_dangerous(command: str) -> bool:
    """True if the command should be blocked."""
    return classify(command)["decision"] == "BLOCK"


__all__ = ["classify", "is_dangerous", "THRESHOLD", "HF_REPO"]
