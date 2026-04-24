#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

DISCLAIMER_PATTERNS = (
    "i'm unable",
    "i cannot",
    "i can't",
    "i'm not able",
    "i am unable",
    "as an ai",
    "as a language model",
    "i don't have personal",
    "i am not able",
    "i'm sorry, but",
    "i don't have the ability",
    "i don't have access",
    "i do not have the ability",
    "i don't have real-time",
    "i don't have realtime",
)

UF_DIR = Path("/mnt/shared-storage-gpfs2/wangqianyi2/slime/ultrafeedback")


def has_disclaimer(text: str) -> bool:
    head = text[:300].lower()
    return any(pattern in head for pattern in DISCLAIMER_PATTERNS)


@dataclass
class FilterStats:
    input: int = 0
    disclaimer: int = 0
    chosen_equals_rejected: int = 0
    kept: int = 0

    @property
    def filtered(self) -> int:
        return self.disclaimer + self.chosen_equals_rejected


def filter_file(input_path: Path, output_path: Path) -> FilterStats:
    stats = FilterStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            stats.input += 1
            obj = json.loads(line)
            chosen = str(obj.get("chosen", ""))
            rejected = str(obj.get("rejected", ""))

            if has_disclaimer(chosen):
                stats.disclaimer += 1
                continue
            if chosen.strip() == rejected.strip():
                stats.chosen_equals_rejected += 1
                continue

            dst.write(line)
            stats.kept += 1

    print(
        f"{input_path.name}: input={stats.input} kept={stats.kept} "
        f"filtered={stats.filtered} "
        f"(disclaimer={stats.disclaimer}, chosen==rejected={stats.chosen_equals_rejected}) "
        f"-> {output_path.name}"
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter low-quality rows from preference files.")
    parser.add_argument("--train-input",  type=Path, default=UF_DIR / "uf-train-synth-prefs.jsonl")
    parser.add_argument("--train-output", type=Path, default=UF_DIR / "uf-train-synth-prefs-clean.jsonl")
    parser.add_argument("--test-input",   type=Path, default=UF_DIR / "uf-test-synth-prefs.jsonl")
    parser.add_argument("--test-output",  type=Path, default=UF_DIR / "uf-test-synth-prefs-clean.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_file(args.train_input, args.train_output)
    filter_file(args.test_input, args.test_output)


if __name__ == "__main__":
    main()
