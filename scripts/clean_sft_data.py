#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DIRTY_PATTERNS = {
    "as_an_ai": re.compile(r"\bas an ai\b", re.IGNORECASE),
    "happy_to_help": re.compile(r"\bhappy to help\b", re.IGNORECASE),
    "not_factually_coherent": re.compile(r"\bnot factually coherent\b", re.IGNORECASE),
    "must_point_out": re.compile(r"\bi must point out\b", re.IGNORECASE),
    "must_inform": re.compile(r"\bi must inform\b", re.IGNORECASE),
    "as_language_model": re.compile(r"\bas a language model\b", re.IGNORECASE),
    "as_artificial": re.compile(r"\bas an artificial\b", re.IGNORECASE),
    "i_cannot": re.compile(r"\bi cannot\b", re.IGNORECASE),
    "i_cant": re.compile(r"\bi can't\b", re.IGNORECASE),
    "im_not_able": re.compile(r"\bi'?m not able\b", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter obviously dirty SFT samples.")
    parser.add_argument("--input", type=Path, required=True, help="Input SFT JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output cleaned SFT JSONL path.")
    parser.add_argument("--prefix-chars", type=int, default=300, help="Only inspect this many leading assistant characters.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def _last_assistant_text(row: dict) -> str:
    messages = row.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def _match_dirty_prefix(text: str, prefix_chars: int) -> str | None:
    prefix = text[:prefix_chars]
    for name, pattern in DIRTY_PATTERNS.items():
        if pattern.search(prefix):
            return name
    return None


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    removed = 0
    invalid = 0
    removed_by_pattern: Counter[str] = Counter()

    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            assistant_text = _last_assistant_text(row)
            if not assistant_text:
                invalid += 1
                continue
            matched = _match_dirty_prefix(assistant_text, args.prefix_chars)
            if matched is not None:
                removed += 1
                removed_by_pattern[matched] += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "total": total,
        "kept": kept,
        "removed": removed,
        "invalid": invalid,
        "removed_by_pattern": dict(removed_by_pattern.most_common()),
        "prefix_chars": args.prefix_chars,
    }

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
