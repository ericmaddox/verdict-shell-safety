"""Local predictor: downloads artifacts from the Hub once, then runs the exact
inference contract:

1. parse_and_serialize via bashlex -> UNPARSED:/EMPTY fail closed (score 1.0)
2. tokenize serialized form (max_length=128, truncation, padding)
3. single ONNX forward pass -> logit
4. sigmoid(logit); score >= 0.50 -> BLOCK else PASS
"""

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from . import HF_REPO, MAX_LENGTH, MODEL_SHA256, THRESHOLD
from .parser import parse_and_serialize


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class VerdictPredictor:
    def __init__(self, cache_dir, model_file: str = "model.onnx"):
        model_path = hf_hub_download(HF_REPO, model_file, cache_dir=str(cache_dir))
        if model_file == "model.onnx" and _sha256(model_path) != MODEL_SHA256:
            raise RuntimeError(f"model.onnx SHA-256 mismatch: {model_path}")
        # both files land in the same snapshot dir; load the tokenizer from it
        snapshot_dir = Path(model_path).parent
        hf_hub_download(HF_REPO, "tokenizer.json", cache_dir=str(cache_dir))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))

    def _score_parsed(self, parsed: str) -> float:
        # Tokenize without padding (the file carries no pad-token metadata),
        # then pad manually: pad positions get attention_mask 0, so the pad
        # id cannot affect the masked mean pool.
        enc = self.tokenizer(
            parsed,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )
        ids = enc["input_ids"][:MAX_LENGTH]
        mask = enc["attention_mask"][:MAX_LENGTH]
        pad_len = MAX_LENGTH - len(ids)
        ort_inputs = {
            "input_ids": np.array([ids + [0] * pad_len], dtype=np.int64),
            "attention_mask": np.array([mask + [0] * pad_len], dtype=np.int64),
        }
        logit = float(self.session.run(["logits"], ort_inputs)[0][0])
        return 1.0 / (1.0 + np.exp(-logit))

    def predict(self, raw_command: str) -> dict:
        parsed = parse_and_serialize(raw_command)
        if parsed.startswith("UNPARSED:") or parsed == "EMPTY":
            return {
                "command": raw_command,
                "parsed": parsed,
                "score": 1.0,
                "decision": "BLOCK",
                "fail_closed": True,
            }
        prob = self._score_parsed(parsed)
        return {
            "command": raw_command,
            "parsed": parsed,
            "score": float(round(prob, 6)),
            "decision": "BLOCK" if prob >= THRESHOLD else "PASS",
            "fail_closed": False,
        }
