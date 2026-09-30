from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/remove_no_prediction_history.py"
SPEC = importlib.util.spec_from_file_location("remove_no_prediction_history", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
remove_no_prediction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remove_no_prediction)


class RemoveNoPredictionHistoryTests(unittest.TestCase):
    def test_finds_no_prediction_question_files_but_not_reports_or_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "answered.json").write_text(
                json.dumps({"question_id": "answered", "response": "<answer>A</answer>"}),
                encoding="utf-8",
            )
            (root / "stopped.json").write_text(
                json.dumps(
                    {
                        "question_id": "stopped",
                        "response": "Agent stopped due to max iterations.",
                    }
                ),
                encoding="utf-8",
            )
            (root / "error.json").write_text(
                json.dumps({"question_id": "error", "response": "[ERROR] failed"}),
                encoding="utf-8",
            )
            (root / "evaluation.json").write_text(
                json.dumps({"no_prediction_ids": ["stopped"]}), encoding="utf-8"
            )

            matches = remove_no_prediction.find_no_prediction_files(root)

            self.assertEqual([path.name for path in matches], ["stopped.json"])

    def test_rewrite_aggregate_removes_selected_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            aggregate = root / "output_test.jsonl"
            records = [
                {"video_id": "ok", "question_data": {"response": "<answer>A</answer>"}},
                {
                    "video_id": "stopped",
                    "question_data": {"response": "Agent stopped due to max iterations."},
                },
            ]
            aggregate.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            backup = root / "backup"
            backup.mkdir()

            removed, kept = remove_no_prediction.rewrite_aggregate(
                aggregate,
                removal_ids={"stopped"},
                backup_dir=backup,
            )

            self.assertEqual((removed, kept), (1, 1))
            self.assertTrue((backup / "output_test.jsonl").is_file())
            remaining = remove_no_prediction.load_jsonl(aggregate)
            self.assertEqual([record["video_id"] for record in remaining], ["ok"])


if __name__ == "__main__":
    unittest.main()
