# verdict-shell-safety

Local, offline shell-command safety classifier. Every command is parsed with bashlex into a flat serialized AST, then scored by a 151M-parameter binary classifier (ModernBERT-based, ONNX). No network calls at inference time, no API keys, no tokens per call.

**Model weights:** [aialchemist-dev/verdict-shell-safety](https://huggingface.co/aialchemist-dev/verdict-shell-safety) (downloaded automatically on first use, SHA-256 verified).

## Install

```bash
pip install verdict-shell-safety
```

## Use

```python
from verdict_shell_safety import classify, is_dangerous

classify("rm -rf /")
# {'command': 'rm -rf /', 'parsed': 'CMD: rm ARG: -rf ARG: /',
#  'score': 0.999025, 'decision': 'BLOCK', 'fail_closed': False}

is_dangerous("ls -la")   # False
is_dangerous("dd if=/dev/zero of=/dev/sda")  # True
```

Scores ≥ 0.50 → `BLOCK`, below → `PASS`. Unparseable input and empty
commands fail closed (`BLOCK`, score 1.0).

## Evaluation

- Core set: **46/46 dangerous blocked, 42/42 harmless passed** (104-case set in `eval/`, including review-compatible cases; the binary gate is 88).
- Private held-out variant set: **143/143**.
- Latency (CPU): parser p50 0.54ms, model p50 56ms, end-to-end p50 56ms.

Run the gate yourself: `python eval/run_eval.py` (CI runs it on every push).

## Companion INT8 build

A 450MB quantized ensemble (`verdict-parse-int8-ensemble.onnx` on the Hub)
scores 141/145 on the variant set with the core gate intact — the fallback for
memory-constrained deployments.

## Limitations

Narrow tool, honestly scoped: shell-command triage, not a general safety
system. Self-built evaluation, not an external benchmark. No dedicated
adversarial campaign yet — if you find a miss, file an issue with the exact
command string. One known residual: `usermod -L telemetry_daemon` scores
0.216 (PASS). See the [model card](https://huggingface.co/aialchemist-dev/verdict-shell-safety) for full details.

## License

Apache-2.0. Base model: [heman10x/rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m) (Apache-2.0).
