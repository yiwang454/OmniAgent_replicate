#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.omniagent_venv/bin/python}"

INPUT_JSONL="${AVUT_INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/avut/mllm_instruct_opt/avut_gemini2.5flash_direct_safe_duration_sorted_input/avut_full1734_duration_ascending_safe_videos.jsonl}"
OUTPUT_DIR="${AVUT_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/avut/OmniAgent_avut_full1734_safe_duration_with_metrics}"
EXPECTED_COUNT=1734

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [ ! -f "${INPUT_JSONL}" ]; then
  echo "AVUT input JSONL does not exist: ${INPUT_JSONL}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
LOG_FILE="${LOG_FILE_OVERRIDE:-${OUTPUT_DIR}/run.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Benchmark: AVUT"
echo "Input JSONL: ${INPUT_JSONL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Workers: 4"

"${PYTHON_BIN}" main.py \
  "$@" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers 4

"${PYTHON_BIN}" scripts/aggregate_inference_metrics.py \
  "${OUTPUT_DIR}/output_test.jsonl" \
  --expected-count "${EXPECTED_COUNT}" \
  --output-json "${OUTPUT_DIR}/inference_metrics_summary.json"
