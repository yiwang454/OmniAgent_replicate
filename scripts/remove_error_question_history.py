#!/usr/bin/env python3
"""Remove explicit error histories from an OmniAgent rollout directory safely."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_FILENAME = "output_test.jsonl"


def question_data(record: dict[str, Any]) -> dict[str, Any]:
    nested = record.get("question_data")
    return nested if isinstance(nested, dict) else record


def is_explicit_error(record: dict[str, Any], *, include_empty: bool = False) -> bool:
    data = question_data(record)
    response = str(data.get("response") or "").strip()
    if response.startswith("[ERROR]") or data.get("error"):
        return True
    if include_empty and not response:
        return True
    return False


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


def find_error_files(output_dir: Path, *, include_empty: bool) -> list[Path]:
    errors: list[Path] = []
    for path in sorted(output_dir.glob("*.json")):
        # Reports are not question trajectories.
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "response" not in question_data(value):
            continue
        if is_explicit_error(value, include_empty=include_empty):
            errors.append(path)
    return errors


def rewrite_aggregate(
    aggregate: Path,
    *,
    include_empty: bool,
    backup_dir: Path,
) -> tuple[int, int]:
    records = load_jsonl(aggregate)
    kept = [
        record
        for record in records
        if not is_explicit_error(record, include_empty=include_empty)
    ]
    removed = len(records) - len(kept)
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
        "--include-empty",
        action="store_true",
        help="Also remove trajectories whose response is empty.",
    )
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

    error_files = find_error_files(output_dir, include_empty=args.include_empty)
    aggregate = output_dir / OUTPUT_FILENAME
    aggregate_records = load_jsonl(aggregate)
    aggregate_error_count = sum(
        is_explicit_error(record, include_empty=args.include_empty)
        for record in aggregate_records
    )

    print(f"Question histories matched: {len(error_files)}")
    print(f"Aggregate error rows matched: {aggregate_error_count}")
    if args.list:
        for path in error_files:
            print(path.name)

    if not args.apply:
        print("Dry run only; no files changed. Add --apply to move matched histories.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = output_dir / ".removed_error_history" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in error_files:
        shutil.move(str(path), backup_dir / path.name)
    removed, kept = rewrite_aggregate(
        aggregate,
        include_empty=args.include_empty,
        backup_dir=backup_dir,
    )
    print(f"Moved {len(error_files)} question histories to: {backup_dir}")
    print(f"Aggregate rewritten: removed={removed}, kept={kept}")


if __name__ == "__main__":
    main()
