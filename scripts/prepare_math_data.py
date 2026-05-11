#!/usr/bin/env python3
"""Prepare math IRL/PPO/eval JSONL files for SLIME.

Reads numinamath/train (5 parquet shards), numinamath/test, and four mathtest
sources (aime24/25/26 + olympiadbench) and produces the JSONL files consumed by
the SLIME IRL pipeline. Every retained sample is verified with
``grade_answer_verl(pseudo_response, label)`` so that the label can recover
itself; rows that fail the self-check are dropped and accounted for in
``prepare_report.json``.

Outputs (under --output-dir):
    train_demo.jsonl     - IRL demo data ({text, chosen, source_row_id, metadata})
    train_prompts.jsonl  - PPO/bootstrap prompts ({text, source_row_id, label, metadata})
    test_demo.jsonl      - numinamath/test demo (RM diagnostics, optional)
    eval_prompts.jsonl   - merged 4 mathtest sources ({text, label, metadata.source})
    prepare_report.json  - per-source totals / kept / filter reasons
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import random
import sys
import traceback
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "prepare_math_data.py requires pyarrow. Install it (`pip install pyarrow`) and retry."
    ) from exc

from tqdm import tqdm


def _load_math_utils():
    """Load math_utils.py directly, bypassing slime.rollout.rm_hub.__init__.

    The package __init__ pulls in ray/aiohttp/etc which are unnecessary for pure
    label grading and may not be installed in the data-prep environment.
    """
    module_path = REPO_ROOT / "slime" / "rollout" / "rm_hub" / "math_utils.py"
    spec = importlib.util.spec_from_file_location("_math_utils_standalone", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load math_utils from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_math_utils = _load_math_utils()
extract_answer = _math_utils.extract_answer
grade_answer_verl = _math_utils.grade_answer_verl

PROMPT_SUFFIX = "\n\nPlease reason step by step and put your final answer within \\boxed{}."
PARQUET_BATCH_SIZE = 1024
POOL_CHUNK_SIZE = 64


@dataclass(slots=True)
class ReportStats:
    expected_reasons: tuple[str, ...] = ()
    total_input: int = 0
    kept: int = 0
    filtered: Counter[str] = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.filtered[reason] += 1

    def as_dict(self) -> dict[str, Any]:
        ordered: dict[str, int] = {reason: self.filtered.get(reason, 0) for reason in self.expected_reasons}
        for reason in sorted(self.filtered):
            if reason not in ordered:
                ordered[reason] = self.filtered[reason]
        return {
            "total_input": self.total_input,
            "kept": self.kept,
            "filtered": ordered,
        }


@dataclass(slots=True)
class Candidate:
    label: str
    rows: dict[str, dict[str, Any]]


CandidateBuilder = Callable[[dict[str, Any], int, Path], tuple[str | None, Candidate | None]]


class AtomicTextWriter:
    """Write to a ``.tmp`` file and atomically rename on clean exit.

    Failure path: if the context manager exits with an exception the tmp file is
    discarded and the destination is left untouched.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.tmp_path = path.with_name(f"{path.name}.tmp")
        self.handle: TextIO | None = None

    def __enter__(self) -> "AtomicTextWriter":
        self.tmp_path.unlink(missing_ok=True)
        self.handle = self.tmp_path.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, text: str) -> None:
        assert self.handle is not None
        self.handle.write(text)

    def write_jsonl(self, row: dict[str, Any]) -> None:
        self.write(json.dumps(row, ensure_ascii=False))
        self.write("\n")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()
        if exc_type is None:
            os.replace(self.tmp_path, self.path)
        else:
            self.tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--numinamath-train-dir", type=Path, default=Path("numinamath/train"))
    parser.add_argument("--numinamath-test-path", type=Path, default=Path("numinamath/test/test-00000-of-00001.parquet"))
    parser.add_argument("--aime24-path", type=Path, default=Path("mathtest/aime24/test-00000-of-00001.parquet"))
    parser.add_argument("--aime25-path", type=Path, default=Path("mathtest/aime25/test.jsonl"))
    parser.add_argument("--aime26-path", type=Path, default=Path("mathtest/aime26/aime2026.jsonl"))
    parser.add_argument("--olympiad-path", type=Path, default=Path("mathtest/olympiadbench/test.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("math"))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on total numinamath train rows BEFORE filtering. Useful for smoke tests.",
    )
    parser.add_argument("--num-workers", type=int, default=16, help="Worker count for label self-check.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise SystemExit(f"--limit must be > 0 when provided, got {args.limit}")
    if args.num_workers <= 0:
        raise SystemExit(f"--num-workers must be > 0, got {args.num_workers}")


def ensure_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SystemExit(f"--output-dir must point to a directory, got {output_dir}")
        if not force:
            raise SystemExit(f"Output directory already exists: {output_dir}. Pass --force to overwrite.")
        return
    output_dir.mkdir(parents=True, exist_ok=False)


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def stringify(value: Any) -> str:
    return "" if value is None else str(value)


def build_prompt(problem: str) -> str:
    return f"{problem.rstrip()}{PROMPT_SUFFIX}"


def get_source_row_id(row: dict[str, Any], fallback_row_id: int) -> Any:
    row_id = row.get("id")
    if row_id in (None, ""):
        return fallback_row_id
    return row_id


def strip_outer_dollar_math(label: str) -> str:
    """Strip a single pair of outer dollar wrappers, conservatively.

    Only strips when:
      - The label starts/ends with matching ``$$`` or ``$`` markers.
      - For single ``$...$``, the interior contains no unescaped ``$`` (avoids
        corrupting strings like ``$x$ and $y$``).
    """
    stripped = label.strip()
    if len(stripped) >= 4 and stripped.startswith("$$") and stripped.endswith("$$"):
        return stripped[2:-2].strip()
    if len(stripped) >= 2 and stripped[0] == "$" and stripped[-1] == "$":
        interior = stripped[1:-1]
        # Reject if interior has any unescaped $ (would mean multi-segment).
        idx = 0
        has_unescaped_dollar = False
        while idx < len(interior):
            ch = interior[idx]
            if ch == "\\" and idx + 1 < len(interior):
                idx += 2
                continue
            if ch == "$":
                has_unescaped_dollar = True
                break
            idx += 1
        if not has_unescaped_dollar:
            return interior.strip()
    return stripped


def coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def resolve_parquet_columns(
    path: Path,
    parquet_file: pq.ParquetFile,
    required_columns: Sequence[str],
    optional_columns: Sequence[str] = (),
) -> list[str]:
    available = set(parquet_file.schema_arrow.names)
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise SystemExit(f"Missing parquet columns in {path}: {missing}")
    return [*required_columns, *[column for column in optional_columns if column in available]]


def run_label_self_check(label: str) -> bool:
    pseudo_response = f"Reasoning steps...\n\\boxed{{{label}}}"
    try:
        return bool(grade_answer_verl(pseudo_response, label))
    except Exception:
        return False


def apply_self_check(
    candidates: list[Candidate],
    writers: dict[str, AtomicTextWriter],
    stats: ReportStats,
    pool: Pool | None,
) -> None:
    if not candidates:
        return

    labels = [candidate.label for candidate in candidates]
    if pool is None:
        keep_mask = [run_label_self_check(label) for label in labels]
    else:
        chunksize = max(1, min(POOL_CHUNK_SIZE, len(labels) // 8 or 1))
        keep_mask = list(pool.imap(run_label_self_check, labels, chunksize=chunksize))

    for candidate, keep in zip(candidates, keep_mask, strict=True):
        if not keep:
            stats.reject("label_self_check_failed")
            continue
        stats.kept += 1
        for writer_name, row in candidate.rows.items():
            writers[writer_name].write_jsonl(row)


def build_numinamath_candidate(
    row: dict[str, Any],
    source_row_id: int,
    source_path: Path,
    *,
    source_dataset: str,
    demo_writer_name: str,
    include_prompt_row: bool,
) -> tuple[str | None, Candidate | None]:
    solution = stringify(row.get("solution"))
    if not solution.strip():
        return "empty_solution", None

    label = extract_answer(solution)
    if label is None:
        return "no_boxed", None
    label = label.strip()

    prompt = build_prompt(stringify(row.get("problem")))
    source_file = display_path(source_path)
    # NOTE: We do NOT write a top-level ``source_row_id`` field. The slime IRL
    # link aligns demo and PPO rollouts via the JSONL row index produced by
    # ``read_file_with_source_row_ids`` (which is also what
    # ``Dataset.metadata.setdefault('source_row_id', line_idx)`` falls back
    # to). Writing a per-shard ``source_row_id`` here would shadow that
    # alignment and break ``match_token_sample(strict_row_id=True)``.
    rows: dict[str, dict[str, Any]] = {
        demo_writer_name: {
            "text": prompt,
            "chosen": solution,
            "metadata": {
                "source_dataset": source_dataset,
                "source_file": source_file,
                "shard_row_id": source_row_id,
            },
        }
    }
    if include_prompt_row:
        rows["train_prompts"] = {
            "text": prompt,
            "label": label,
            "metadata": {
                "source_dataset": source_dataset,
                "source_file": source_file,
                "shard_row_id": source_row_id,
            },
        }
    return None, Candidate(label=label, rows=rows)


def build_numinamath_train_candidate(row, source_row_id, source_path):
    return build_numinamath_candidate(
        row, source_row_id, source_path,
        source_dataset="numinamath_train", demo_writer_name="train_demo",
        include_prompt_row=True,
    )


def build_numinamath_test_candidate(row, source_row_id, source_path):
    return build_numinamath_candidate(
        row, source_row_id, source_path,
        source_dataset="numinamath_test", demo_writer_name="test_demo",
        include_prompt_row=False,
    )


def build_aime24_candidate(row, fallback_row_id, source_path):
    del source_path
    solution = stringify(row.get("solution"))
    if not solution.strip():
        return "empty_solution", None
    label = extract_answer(solution)
    if label is None:
        return "no_boxed", None
    label = label.strip()
    source_row_id = get_source_row_id(row, fallback_row_id)
    return None, Candidate(
        label=label,
        rows={
            "eval_prompts": {
                "text": build_prompt(stringify(row.get("problem"))),
                "label": label,
                "metadata": {"source": "aime24", "source_row_id": source_row_id},
            }
        },
    )


def _build_plain_eval(row, fallback_row_id, source_name):
    label = stringify(row.get("answer")).strip()
    if not label:
        return "empty_answer", None
    source_row_id = get_source_row_id(row, fallback_row_id)
    return None, Candidate(
        label=label,
        rows={
            "eval_prompts": {
                "text": build_prompt(stringify(row.get("problem"))),
                "label": label,
                "metadata": {"source": source_name, "source_row_id": source_row_id},
            }
        },
    )


def build_aime25_candidate(row, fallback_row_id, source_path):
    del source_path
    return _build_plain_eval(row, fallback_row_id, "aime25")


def build_aime26_candidate(row, fallback_row_id, source_path):
    del source_path
    return _build_plain_eval(row, fallback_row_id, "aime26")


def build_olympiad_candidate(row, fallback_row_id, source_path):
    del source_path
    modality = stringify(row.get("modality")).strip()
    if modality.casefold() != "text-only":
        return "non_text_only", None

    answers = coerce_list(row.get("final_answer"))
    if len(answers) != 1:
        return "multi_answer", None

    label = strip_outer_dollar_math(stringify(answers[0]))
    if not label:
        return "empty_answer", None
    source_row_id = get_source_row_id(row, fallback_row_id)
    return None, Candidate(
        label=label,
        rows={
            "eval_prompts": {
                "text": build_prompt(stringify(row.get("question"))),
                "label": label,
                "metadata": {"source": "olympiad", "source_row_id": source_row_id},
            }
        },
    )


def process_parquet_source(
    *,
    path: Path,
    required_columns: Sequence[str],
    optional_columns: Sequence[str],
    desc: str,
    limit: int | None,
    stats: ReportStats,
    build_candidate: CandidateBuilder,
    writers: dict[str, AtomicTextWriter],
    pool: Pool | None,
) -> int:
    parquet_file = pq.ParquetFile(path)
    columns = resolve_parquet_columns(path, parquet_file, required_columns, optional_columns)
    max_rows = parquet_file.metadata.num_rows
    if limit is not None:
        max_rows = min(max_rows, limit)

    processed = 0
    row_offset = 0
    with tqdm(total=max_rows, desc=desc, unit="row") as progress:
        for batch in parquet_file.iter_batches(columns=columns, batch_size=PARQUET_BATCH_SIZE):
            if limit is not None and processed >= limit:
                break

            rows = batch.to_pylist()
            if limit is not None:
                rows = rows[: limit - processed]
            if not rows:
                break

            stats.total_input += len(rows)
            processed += len(rows)
            progress.update(len(rows))

            candidates: list[Candidate] = []
            for local_index, row in enumerate(rows):
                reason, candidate = build_candidate(row, row_offset + local_index, path)
                if candidate is None:
                    stats.reject(reason or "unknown")
                    continue
                candidates.append(candidate)

            apply_self_check(candidates, writers, stats, pool)
            row_offset += len(rows)

    return processed


def process_jsonl_source(
    *,
    path: Path,
    desc: str,
    stats: ReportStats,
    build_candidate: CandidateBuilder,
    writers: dict[str, AtomicTextWriter],
    pool: Pool | None,
) -> None:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    candidates: list[Candidate] = []
    with tqdm(total=len(rows), desc=desc, unit="row") as progress:
        for row_index, row in enumerate(rows):
            stats.total_input += 1
            reason, candidate = build_candidate(row, row_index, path)
            if candidate is None:
                stats.reject(reason or "unknown")
            else:
                candidates.append(candidate)
            progress.update(1)

    apply_self_check(candidates, writers, stats, pool)


def print_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72, file=sys.stderr)
    print("PREPARE MATH DATA — SUMMARY", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    for source_name, stats in report.items():
        if source_name == "config":
            continue
        kept = stats["kept"]
        total = stats["total_input"]
        rate = f"{kept / total * 100:.1f}%" if total else "n/a"
        filtered_str = ", ".join(f"{k}={v}" for k, v in stats["filtered"].items() if v) or "—"
        print(f"  {source_name:18s} kept={kept:>7d}/{total:<7d} ({rate})  filtered: {filtered_str}", file=sys.stderr)
    print("=" * 72 + "\n", file=sys.stderr)


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)

    require_existing_path(args.numinamath_train_dir, "numinamath train directory")
    require_existing_path(args.numinamath_test_path, "numinamath test parquet")
    require_existing_path(args.aime24_path, "aime24 parquet")
    require_existing_path(args.aime25_path, "aime25 jsonl")
    require_existing_path(args.aime26_path, "aime26 jsonl")
    require_existing_path(args.olympiad_path, "olympiad parquet")
    ensure_output_dir(args.output_dir, args.force)

    train_paths = sorted(args.numinamath_train_dir.glob("train-*.parquet"))
    if not train_paths:
        raise SystemExit(f"No train-*.parquet files found under {args.numinamath_train_dir}")

    report_stats = {
        "numinamath_train": ReportStats(expected_reasons=("empty_solution", "no_boxed", "label_self_check_failed")),
        "numinamath_test": ReportStats(expected_reasons=("empty_solution", "no_boxed", "label_self_check_failed")),
        "aime24": ReportStats(expected_reasons=("empty_solution", "no_boxed", "label_self_check_failed")),
        "aime25": ReportStats(expected_reasons=("empty_answer", "label_self_check_failed")),
        "aime26": ReportStats(expected_reasons=("empty_answer", "label_self_check_failed")),
        "olympiad": ReportStats(expected_reasons=("non_text_only", "multi_answer", "empty_answer", "label_self_check_failed")),
    }

    output_paths = {
        "train_demo": args.output_dir / "train_demo.jsonl",
        "train_prompts": args.output_dir / "train_prompts.jsonl",
        "test_demo": args.output_dir / "test_demo.jsonl",
        "eval_prompts": args.output_dir / "eval_prompts.jsonl",
    }
    report_path = args.output_dir / "prepare_report.json"

    with ExitStack() as stack:
        writers = {name: stack.enter_context(AtomicTextWriter(path)) for name, path in output_paths.items()}
        pool = stack.enter_context(mp.Pool(processes=args.num_workers)) if args.num_workers > 1 else None

        remaining_train = args.limit
        for train_path in train_paths:
            if remaining_train is not None and remaining_train <= 0:
                break
            processed = process_parquet_source(
                path=train_path,
                required_columns=["problem", "solution"],
                optional_columns=[],
                desc=f"numinamath_train {train_path.name}",
                limit=remaining_train,
                stats=report_stats["numinamath_train"],
                build_candidate=build_numinamath_train_candidate,
                writers=writers,
                pool=pool,
            )
            if remaining_train is not None:
                remaining_train -= processed

        process_parquet_source(
            path=args.numinamath_test_path,
            required_columns=["problem", "solution"], optional_columns=[],
            desc="numinamath_test", limit=None,
            stats=report_stats["numinamath_test"],
            build_candidate=build_numinamath_test_candidate,
            writers=writers, pool=pool,
        )
        process_parquet_source(
            path=args.aime24_path,
            required_columns=["problem", "solution"], optional_columns=["id"],
            desc="aime24", limit=None,
            stats=report_stats["aime24"],
            build_candidate=build_aime24_candidate,
            writers=writers, pool=pool,
        )
        process_jsonl_source(
            path=args.aime25_path, desc="aime25",
            stats=report_stats["aime25"], build_candidate=build_aime25_candidate,
            writers=writers, pool=pool,
        )
        process_jsonl_source(
            path=args.aime26_path, desc="aime26",
            stats=report_stats["aime26"], build_candidate=build_aime26_candidate,
            writers=writers, pool=pool,
        )
        process_parquet_source(
            path=args.olympiad_path,
            required_columns=["question", "final_answer", "modality"], optional_columns=["id"],
            desc="olympiad", limit=None,
            stats=report_stats["olympiad"],
            build_candidate=build_olympiad_candidate,
            writers=writers, pool=pool,
        )

        report = {name: stats.as_dict() for name, stats in report_stats.items()}
        report["config"] = {
            "prompt_suffix": PROMPT_SUFFIX,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "limit": args.limit,
            "output_dir": str(args.output_dir),
        }

    # All data writers committed (or rolled back) by here. Only now write the
    # report so a partial-data run never produces a self-contradicting report.
    report_tmp = report_path.with_name(f"{report_path.name}.tmp")
    report_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(report_tmp, report_path)

    print_summary(report)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
