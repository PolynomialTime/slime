#!/usr/bin/env python3
"""Math accuracy gate for the SLIME IRL pipeline (DATA_MODE=math).

Reads a per-round policy output JSONL (one record per prompt, schema
``{prompt, response}``) and the prepared eval prompts JSONL (schema
``{text, label, metadata.source}``), then computes:

  * Overall pass@1 via ``grade_answer_verl(response, label)``.
  * Per-source pass@1 (aime24 / aime25 / aime26 / olympiad).
  * boxed_hit_rate (fraction of responses containing a ``\\boxed{...}``).

Outputs JSON to ``--output`` (default ``eval/eval_math_round_${ROUND}.json``).

Alignment strategy: outputs are produced by ``eval_generate_sglang.py`` which
preserves prompt order, so we align by row index. We additionally verify that
the truncated prompt strings match — any mismatch is a hard error since it
indicates the eval set was regenerated under the policy outputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_math_utils():
    """Bypass slime.rollout.rm_hub.__init__ (which pulls in ray/aiohttp)."""
    module_path = REPO_ROOT / "slime" / "rollout" / "rm_hub" / "math_utils.py"
    spec = importlib.util.spec_from_file_location("_math_utils_standalone", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load math_utils from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_math_utils = _load_math_utils()
grade_answer_verl = _math_utils.grade_answer_verl

BOXED_PATTERN = re.compile(r"\\boxed\s*{")


@dataclass(slots=True)
class EvalSample:
    label: str
    source: str
    eval_text: str
    response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, required=True, help="Per-round policy outputs JSONL.")
    parser.add_argument("--eval-prompts", type=Path, required=True, help="math/eval_prompts.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the per-round eval JSON.")
    parser.add_argument("--round", type=int, required=True, help="Round id (recorded in the JSON).")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--prompt-match-prefix-len", type=int, default=200,
        help="Compare this many leading chars of each prompt as alignment check.",
    )
    parser.add_argument(
        "--allow-prompt-mismatch", action="store_true",
        help="Downgrade prompt-prefix mismatches to warnings instead of hard errors.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return out


def grade_one(args: tuple[str, str]) -> bool:
    response, label = args
    try:
        return bool(grade_answer_verl(response or "", label))
    except Exception:
        return False


def has_boxed(response: str) -> bool:
    return bool(BOXED_PATTERN.search(response or ""))


def main() -> int:
    args = parse_args()

    if not args.outputs.exists():
        raise SystemExit(f"--outputs not found: {args.outputs}")
    if not args.eval_prompts.exists():
        raise SystemExit(f"--eval-prompts not found: {args.eval_prompts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    outputs = read_jsonl(args.outputs)
    eval_prompts = read_jsonl(args.eval_prompts)

    if len(outputs) != len(eval_prompts):
        raise SystemExit(
            f"Row count mismatch: outputs={len(outputs)} vs eval_prompts={len(eval_prompts)}. "
            f"The two files must be aligned 1:1 by row order."
        )

    # Alignment validation by prompt prefix.
    prefix = max(1, args.prompt_match_prefix_len)
    mismatches: list[tuple[int, str, str]] = []
    samples: list[EvalSample] = []
    for idx, (out_row, ev_row) in enumerate(zip(outputs, eval_prompts, strict=True)):
        out_prompt = str(out_row.get("prompt", ""))
        ev_text = str(ev_row.get("text", ""))
        if out_prompt[:prefix].strip() != ev_text[:prefix].strip():
            mismatches.append((idx, out_prompt[:prefix], ev_text[:prefix]))
        samples.append(EvalSample(
            label=str(ev_row.get("label", "")),
            source=str((ev_row.get("metadata") or {}).get("source", "unknown")),
            eval_text=ev_text,
            response=str(out_row.get("response", "")),
        ))

    if mismatches:
        message = (
            f"{len(mismatches)} prompt-prefix mismatches between outputs and eval_prompts. "
            f"First mismatch at row {mismatches[0][0]}:\n"
            f"  outputs: {mismatches[0][1]!r}\n  eval:    {mismatches[0][2]!r}"
        )
        if args.allow_prompt_mismatch:
            print(f"WARNING: {message}", file=sys.stderr)
        else:
            raise SystemExit(f"ERROR: {message}\nPass --allow-prompt-mismatch to downgrade to warning.")

    # Grade in parallel.
    grade_inputs = [(s.response, s.label) for s in samples]
    if args.num_workers > 1 and len(grade_inputs) > 1:
        with mp.Pool(processes=args.num_workers) as pool:
            chunksize = max(1, len(grade_inputs) // (args.num_workers * 4) or 1)
            verdicts = list(pool.imap(grade_one, grade_inputs, chunksize=chunksize))
    else:
        verdicts = [grade_one(x) for x in grade_inputs]

    # Aggregate.
    total = len(samples)
    correct = sum(verdicts)
    boxed_hits = sum(has_boxed(s.response) for s in samples)

    by_source_total: Counter[str] = Counter()
    by_source_correct: Counter[str] = Counter()
    by_source_boxed: Counter[str] = Counter()
    for sample, ok in zip(samples, verdicts, strict=True):
        by_source_total[sample.source] += 1
        if ok:
            by_source_correct[sample.source] += 1
        if has_boxed(sample.response):
            by_source_boxed[sample.source] += 1

    by_source: dict[str, dict[str, Any]] = {}
    for src in sorted(by_source_total):
        n = by_source_total[src]
        by_source[src] = {
            "total": n,
            "correct": by_source_correct[src],
            "pass_at_1": by_source_correct[src] / n if n else 0.0,
            "boxed_hits": by_source_boxed[src],
            "boxed_hit_rate": by_source_boxed[src] / n if n else 0.0,
        }

    report = {
        "round": args.round,
        "outputs_path": str(args.outputs),
        "eval_prompts_path": str(args.eval_prompts),
        "total": total,
        "correct": correct,
        "pass_at_1": correct / total if total else 0.0,
        "boxed_hits": boxed_hits,
        "boxed_hit_rate": boxed_hits / total if total else 0.0,
        "prompt_mismatches": len(mismatches),
        "by_source": by_source,
    }

    tmp = args.output.with_name(f"{args.output.name}.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)

    # Stdout summary for pipeline log readability.
    print(f"\n=== Round {args.round} math accuracy ===", file=sys.stderr)
    print(f"  overall pass@1 = {report['pass_at_1']:.4f} ({correct}/{total})", file=sys.stderr)
    print(f"  boxed_hit_rate = {report['boxed_hit_rate']:.4f}", file=sys.stderr)
    for src, stats in by_source.items():
        print(
            f"    {src:10s} pass@1={stats['pass_at_1']:.4f} ({stats['correct']}/{stats['total']})  "
            f"boxed={stats['boxed_hit_rate']:.4f}",
            file=sys.stderr,
        )
    print(f"  written: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
