#!/usr/bin/env python3
"""Remove no-answer histories from an OmniAgent rollout directory safely."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_dailyomni_output import (  # noqa: E402
    error_reason,
    extract_prediction,
    is_empty_response,
    normalize_record,
    question_data,
    record_id,
)


OUTPUT_FILENAME = "output_test.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def is_no_parseable_prediction(record: dict[str, Any]) -> bool:
    """Match evaluator records counted in `No parseable prediction`."""
    return (
        not error_reason(record)
        and not is_empty_response(record)
        and not extract_prediction(record)
    )


def load_question_record(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    record = normalize_record(payload, path.stem)
    if "response" not in question_data(record):
        return None
    return record


def find_no_prediction_files(output_dir: Path) -> list[Path]:
    matches: list[Path] = []
    for path in sorted(output_dir.glob("*.json")):
        record = load_question_record(path)
        if record and is_no_parseable_prediction(record):
            matches.append(path)
    return matches


def no_prediction_ids_from_records(records: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for record in records:
        normalized = normalize_record(record)
        if is_no_parseable_prediction(normalized):
            sample_id = record_id(normalized)
            if sample_id:
                ids.add(sample_id)
    return ids


def rewrite_aggregate(
    aggregate: Path,
    *,
    removal_ids: set[str],
    backup_dir: Path,
) -> tuple[int, int]:
    records = load_jsonl(aggregate)
    kept: list[dict[str, Any]] = []
    removed = 0
    for record in records:
        sample_id = record_id(normalize_record(record))
        if sample_id in removal_ids:
            removed += 1
            continue
        kept.append(record)

    if aggregate.is_file():
        shutil.copy2(aggregate, backup_dir / aggregate.name)
    temporary = aggregate.parent / f".{aggregate.name}.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for record in kept:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(aggregate)
    return removed, len(kept)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag the command is a dry run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every matching question trajectory filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)

    aggregate = output_dir / OUTPUT_FILENAME
    aggregate_records = load_jsonl(aggregate)
    aggregate_ids = no_prediction_ids_from_records(aggregate_records)
    question_files = find_no_prediction_files(output_dir)
    file_ids = {path.stem for path in question_files}
    removal_ids = aggregate_ids | file_ids

    print(f"Question histories matched: {len(question_files)}")
    print(f"Aggregate no-prediction rows matched: {len(aggregate_ids)}")
    print(f"Unique sample IDs to remove from resume state: {len(removal_ids)}")
    if args.list:
        for sample_id in sorted(removal_ids):
            source = "file+aggregate"
            if sample_id not in aggregate_ids:
                source = "file"
            elif sample_id not in file_ids:
                source = "aggregate"
            print(f"{sample_id}\t{source}")

    if not args.apply:
        print("Dry run only; no files changed. Add --apply to move matched histories.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = output_dir / ".removed_no_prediction_history" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in question_files:
        shutil.move(str(path), backup_dir / path.name)
    removed, kept = rewrite_aggregate(
        aggregate,
        removal_ids=removal_ids,
        backup_dir=backup_dir,
    )
    print(f"Moved {len(question_files)} question histories to: {backup_dir}")
    print(f"Aggregate rewritten: removed={removed}, kept={kept}")


if __name__ == "__main__":
    main()
