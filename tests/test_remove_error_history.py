from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/remove_error_question_history.py"
SPEC = importlib.util.spec_from_file_location("remove_error_history", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
remove_errors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remove_errors)


class RemoveErrorHistoryTests(unittest.TestCase):
    def test_finds_error_question_files_but_not_reports_or_successes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "success.json").write_text(
                json.dumps({"question_id": "success", "response": "<answer>A</answer>"}),
                encoding="utf-8",
            )
            (root / "error.json").write_text(
                json.dumps({"question_id": "error", "response": "[ERROR] failed"}),
                encoding="utf-8",
            )
            (root / "evaluation.json").write_text(
                json.dumps({"summary": {"errors": 1}}), encoding="utf-8"
            )

            matches = remove_errors.find_error_files(root, include_empty=False)

            self.assertEqual([path.name for path in matches], ["error.json"])

    def test_rewrite_aggregate_removes_only_explicit_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            aggregate = root / "output_test.jsonl"
            records = [
                {"video_id": "ok", "question_data": {"response": "<answer>A</answer>"}},
                {"video_id": "bad", "question_data": {"response": "[ERROR] failed"}},
            ]
            aggregate.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            backup = root / "backup"
            backup.mkdir()

            removed, kept = remove_errors.rewrite_aggregate(
                aggregate,
                include_empty=False,
                backup_dir=backup,
            )

            self.assertEqual((removed, kept), (1, 1))
            self.assertTrue((backup / "output_test.jsonl").is_file())
            remaining = remove_errors.load_jsonl(aggregate)
            self.assertEqual([record["video_id"] for record in remaining], ["ok"])
