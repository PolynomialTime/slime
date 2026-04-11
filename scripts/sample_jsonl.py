#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a deterministic subset from a JSONL file.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--count", type=int, required=True, help="Number of rows to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit(f"--count must be > 0, got {args.count}")

    with args.input.open(encoding="utf-8") as f:
        rows = [line.rstrip("\n") for line in f if line.strip()]
    total = len(rows)
    if total == 0:
        raise SystemExit(f"No JSONL rows found in {args.input}")

    sample_size = min(args.count, total)
    chosen = sorted(random.Random(args.seed).sample(range(total), sample_size))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for idx in chosen:
            f.write(rows[idx] + "\n")

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "requested": args.count,
        "available": total,
        "written": sample_size,
        "seed": args.seed,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
