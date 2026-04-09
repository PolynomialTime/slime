#!/usr/bin/env python3
"""
Convert ultrafeedback_binarized parquet files into the JSONL formats used by SLIME.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing ultrafeedback parquet files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write SLIME JSONL files into.")
    parser.add_argument("--sft-samples", type=int, default=10_000, help="Number of random SFT rows to export.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SFT sampling.")
    return parser.parse_args()


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except ImportError as exc:
        raise SystemExit(
            "Parquet support requires pandas with pyarrow installed. "
            "Run this script with /home/wangqianyi/anaconda3/bin/python3."
        ) from exc


def normalize_messages(raw_messages) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if raw_messages is None:
        return messages

    for item in list(raw_messages):
        if hasattr(item, "as_py"):
            item = item.as_py()
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role is None or content is None:
            continue
        messages.append({"role": str(role), "content": str(content)})
    return messages


def extract_last_assistant(raw_messages, allow_empty: bool = False) -> str | None:
    messages = normalize_messages(raw_messages)
    if not messages:
        return None
    last_message = messages[-1]
    if last_message["role"] != "assistant":
        return None
    content = last_message["content"].strip()
    if content or allow_empty:
        return content
    return None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_train_outputs(train_prefs: pd.DataFrame) -> tuple[list[dict], list[dict], dict[str, int]]:
    train_rows: list[dict] = []
    pref_rows: list[dict] = []
    stats = {"ties": 0, "invalid": 0}

    for row in train_prefs.itertuples(index=False):
        if pd.isna(row.score_chosen) or pd.isna(row.score_rejected) or row.score_chosen == row.score_rejected:
            stats["ties"] += 1
            continue

        prompt = row.prompt.strip() if isinstance(row.prompt, str) else ""
        chosen = extract_last_assistant(row.chosen)
        rejected = extract_last_assistant(row.rejected, allow_empty=True)
        if not prompt or chosen is None or rejected is None:
            stats["invalid"] += 1
            continue

        train_rows.append({"text": prompt, "label": chosen})
        pref_rows.append({"text": prompt, "chosen": chosen, "rejected": rejected})

    return train_rows, pref_rows, stats


def build_sft_outputs(sft_df: pd.DataFrame, limit: int, seed: int) -> tuple[list[dict], int]:
    indices = list(range(len(sft_df)))
    random.Random(seed).shuffle(indices)

    rows: list[dict] = []
    skipped = 0
    for idx in indices:
        messages = normalize_messages(sft_df.iloc[idx]["messages"])
        if not messages:
            skipped += 1
            continue
        rows.append({"messages": messages})
        if len(rows) >= limit:
            break

    return rows, skipped


def build_test_outputs(test_prefs: pd.DataFrame) -> tuple[list[dict], dict[str, int]]:
    test_rows: list[dict] = []
    stats = {"ties": 0, "invalid": 0}

    for row in test_prefs.itertuples(index=False):
        if pd.isna(row.score_chosen) or pd.isna(row.score_rejected) or row.score_chosen == row.score_rejected:
            stats["ties"] += 1
            continue

        prompt = row.prompt.strip() if isinstance(row.prompt, str) else ""
        chosen = extract_last_assistant(row.chosen)
        rejected = extract_last_assistant(row.rejected, allow_empty=True)
        if not prompt or chosen is None or rejected is None:
            stats["invalid"] += 1
            continue

        test_rows.append({"text": prompt, "chosen": chosen, "rejected": rejected})

    return test_rows, stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_prefs_path = args.data_dir / "train_prefs-00000-of-00001.parquet"
    train_sft_path = args.data_dir / "train_sft-00000-of-00001.parquet"
    test_prefs_path = args.data_dir / "test_prefs-00000-of-00001.parquet"

    train_prefs = read_parquet(
        train_prefs_path,
        columns=["prompt", "chosen", "rejected", "score_chosen", "score_rejected"],
    )
    train_rows, pref_rows, train_stats = build_train_outputs(train_prefs)

    sft_source_path = train_sft_path if train_sft_path.exists() else train_prefs_path
    sft_df = read_parquet(sft_source_path, columns=["messages"])
    sft_rows, sft_skipped = build_sft_outputs(sft_df, limit=args.sft_samples, seed=args.seed)

    test_prefs = read_parquet(
        test_prefs_path,
        columns=["prompt", "chosen", "rejected", "score_chosen", "score_rejected"],
    )
    test_rows, test_stats = build_test_outputs(test_prefs)

    write_jsonl(args.output_dir / "uf-train.jsonl", train_rows)
    write_jsonl(args.output_dir / "uf-train-prefs.jsonl", pref_rows)
    write_jsonl(args.output_dir / "uf-sft.jsonl", sft_rows)
    write_jsonl(args.output_dir / "uf-test.jsonl", test_rows)

    print(f"train prefs source: {train_prefs_path}")
    print(f"train rows written: {len(train_rows)} (ties skipped: {train_stats['ties']}, invalid skipped: {train_stats['invalid']})")
    print(f"train pref rows written: {len(pref_rows)}")
    print(f"sft source: {sft_source_path}")
    print(f"sft rows written: {len(sft_rows)} (invalid skipped while sampling: {sft_skipped})")
    print(f"test prefs source: {test_prefs_path}")
    print(f"test rows written: {len(test_rows)} (ties skipped: {test_stats['ties']}, invalid skipped: {test_stats['invalid']})")


if __name__ == "__main__":
    main()
