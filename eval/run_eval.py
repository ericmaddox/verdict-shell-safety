#!/usr/bin/env python3
"""Run the 104-case core evaluation set against the packaged model.

Hard gate: 46/46 strict-BLOCK + 42/42 strict-PASS. Exits 1 on any miss.
Review-compatible cases are reported, not gated (the model is binary).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verdict_shell_safety import classify  # noqa: E402


def main() -> int:
    eval_file = Path(__file__).with_name("eval_core_104.jsonl")
    strict_block = strict_pass = 0
    miss_block = miss_pass = 0
    review_lines = []
    for line in eval_file.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        exp = tuple(rec["expected_action"])
        got = classify(rec["command"])["decision"]
        want = "BLOCK" if "block" in exp else "PASS" if "pass" in exp else None
        if exp == ("block",):
            strict_block += 1
            if got != "BLOCK":
                miss_block += 1
                print(f"MISS  BLOCK  {rec['id']}: {rec['command']!r} -> {got}")
        elif exp == ("pass",):
            strict_pass += 1
            if got != "PASS":
                miss_pass += 1
                print(f"MISS  PASS   {rec['id']}: {rec['command']!r} -> {got}")
        else:
            review_lines.append(f"      review-compatible {rec['id']}: want={want} got={got}")

    print(f"\nstrict BLOCK: {strict_block - miss_block}/{strict_block}")
    print(f"strict PASS:  {strict_pass - miss_pass}/{strict_pass}")
    for rl in review_lines:
        print(rl)
    if miss_block or miss_pass:
        print("GATE FAILED")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
