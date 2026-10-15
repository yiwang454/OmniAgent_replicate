#!/usr/bin/env bash
# Fresh full-tool OmniAgent control on the same 125 questions and sepAV media as
# the tool ablations. Caption calls are live, so their Gemini usage is measured.
# Run with: bash scripts/run_full_omniagent_selectedval125_measured.sh
# Validate without model calls: bash scripts/run_full_omniagent_selectedval125_measured.sh --check-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="/mnt/ceph_rbd/data/avqa_project/daily_omni"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.omniagent_venv/bin/python}"
INPUT_JSONL="${INPUT_JSONL:-${DATA_ROOT}/daily_omni_cuts_selectedVal125.jsonl}"
MEDIA_MANIFEST_JSONL="${MEDIA_MANIFEST_JSONL:-${DATA_ROOT}/daily_omni_cuts_v3_sepAV.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/OmniAgent_full_selectedVal125_measured_sepav}"
CONFIG_JSON="${REPO_DIR}/configs/tool_ablation/full.json"
WORKERS="${WORKERS:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"

if [[ "$#" -gt 1 ]] || { [[ "$#" -eq 1 ]] && [[ "$1" != "--check-only" ]]; }; then
  echo "Usage: bash $0 [--check-only]" >&2
  exit 2
fi
if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS and MAX_ITERATIONS must be positive integers." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
for required in "${INPUT_JSONL}" "${MEDIA_MANIFEST_JSONL}" "${CONFIG_JSON}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file does not exist: ${required}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" - "${REPO_DIR}" "${INPUT_JSONL}" "${MEDIA_MANIFEST_JSONL}" "${CONFIG_JSON}" <<'PY'
import json
import sys
from pathlib import Path

repo, selected, media, config_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo))
from omni_agent.tool_registry import (
    default_tools,
    default_user_template,
    load_tool_selection_config,
)

def ids(path):
    return [json.loads(line)["id"] for line in path.open(encoding="utf-8") if line.strip()]

selected_ids = ids(selected)
if len(selected_ids) != 125 or len(set(selected_ids)) != 125:
    raise SystemExit(f"Expected exactly 125 unique selected questions, got {len(selected_ids)} rows")
media_ids = set(ids(media))
missing = set(selected_ids) - media_ids
if missing:
    raise SystemExit(f"Media manifest is missing {len(missing)} selected questions")

config = load_tool_selection_config(config_path)
original = default_tools()
configured = config.tools()
if config.user_template != default_user_template():
    raise SystemExit("Full-control prompt differs from the original prompt")
for old, new in zip(original, configured, strict=True):
    if (old.name, old.description, old.args) != (new.name, new.description, new.args):
        raise SystemExit(f"Full-control tool differs from original: {old.name}")
print(f"Validated {len(selected_ids)} questions, sepAV media coverage, original prompt and all {len(original)} tools.")
PY

echo "Output: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}; max iterations: ${MAX_ITERATIONS}"
echo "Caption cache: disabled (caption source calls are measured live)"
if [[ "${1:-}" == "--check-only" ]]; then
  echo "Check only: no model calls or output files were created."
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/run.log") 2>&1

"${PYTHON_BIN}" main.py \
  --input-jsonl "${INPUT_JSONL}" \
  --media-manifest-jsonl "${MEDIA_MANIFEST_JSONL}" \
  --experiment-config "${CONFIG_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers "${WORKERS}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --no-print-steps

"${PYTHON_BIN}" scripts/aggregate_inference_metrics.py \
  "${OUTPUT_DIR}/output_test.jsonl" \
  --expected-count 125 \
  --output-json "${OUTPUT_DIR}/metrics_summary.json"

"${PYTHON_BIN}" scripts/evaluate_dailyomni_output.py \
  "${OUTPUT_DIR}/output_test.jsonl" \
  --json-output "${OUTPUT_DIR}/accuracy_summary.json"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/metrics_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
if summary["row_count"] != 125 or summary["unique_id_count"] != 125:
    raise SystemExit("Refusing to report an incomplete or duplicate-ID control run")
if summary["cache_latency"]["cache_hits"]["sum"] != 0:
    raise SystemExit("Unexpected caption-cache hit: this control must measure live captions")

planner = summary["token_usage"]["planner"]
gemini = summary["token_usage"]["perception"]
total = lambda component, key: component[key]["sum"]
input_tokens = total(planner, "input_tokens") + total(gemini, "prompt_tokens")
thinking_tokens = total(planner, "reasoning_output_tokens") + total(gemini, "thoughts_tokens")
visible_tokens = (
    total(planner, "output_tokens")
    - total(planner, "reasoning_output_tokens")
    + total(gemini, "candidate_tokens")
)
cost = summary["estimated_cost_usd"]["totals"]["live_marginal_total"]
print("\nDailyOmni README O-row token/cost estimates from this 125-question control:")
print(f"  Input tok.:                {input_tokens / 125 / 1000:.3f}k")
print(f"  Output thinking tok.:      {thinking_tokens / 125 / 1000:.3f}k")
print(f"  Output non-thinking tok.:  {visible_tokens / 125 / 1000:.3f}k")
print(f"  Cost:                      ${cost / 125:.5f} per question")
print("  Source:", path)
print("  Keep README O-row accuracy/tool calls from the full 1,197-question run;")
print("  label these token/cost cells as a 125-question subset estimate.")
for name, component in (("o3 planner", planner), ("Gemini tools", gemini)):
    measured = total(component, "calls_with_usage")
    calls = total(component, "calls")
    if measured != calls:
        print(f"WARNING: {name} has usage for {measured:g}/{calls:g} calls only.")
failures = summary["failures"]["count"]
if failures:
    print(f"WARNING: {failures} terminal failure(s); inspect metrics_summary.json before updating README.")
PY
