#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from clean_sft_data import _match_dirty_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an SFT messages dataset from synthetic chosen data and filter obviously dirty answers."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with text/chosen fields.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with messages field.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="text",
        help="Input key containing the user prompt.",
    )
    parser.add_argument(
        "--answer-key",
        type=str,
        default="chosen",
        help="Input key containing the assistant answer.",
    )
    parser.add_argument(
        "--prefix-chars",
        type=int,
        default=300,
        help="Only inspect this many leading assistant characters for dirty prefixes.",
    )
    return parser.parse_args()


def _normalize_prompt(prompt_value) -> str:
    if isinstance(prompt_value, str):
        return prompt_value.strip()
    return json.dumps(prompt_value, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    invalid_prompt = 0
    invalid_answer = 0
    removed_dirty = 0
    removed_by_pattern: Counter[str] = Counter()

    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)

            prompt = _normalize_prompt(row.get(args.prompt_key, ""))
            answer = str(row.get(args.answer_key, "")).strip()
            if not prompt:
                invalid_prompt += 1
                continue
            if not answer:
                invalid_answer += 1
                continue

            matched = _match_dirty_prefix(answer, args.prefix_chars)
            if matched is not None:
                removed_dirty += 1
                removed_by_pattern[matched] += 1
                continue

            out_row = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            }
            dst.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "total": total,
        "kept": kept,
        "invalid_prompt": invalid_prompt,
        "invalid_answer": invalid_answer,
        "removed_dirty": removed_dirty,
        "removed_by_pattern": dict(removed_by_pattern.most_common()),
        "prefix_chars": args.prefix_chars,
    }

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
