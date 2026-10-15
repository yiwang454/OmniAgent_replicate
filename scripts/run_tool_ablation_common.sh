#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 EXPERIMENT CONFIG_JSON USE_CAPTION_CACHE [main.py args...]" >&2
  exit 2
fi

EXPERIMENT="$1"
CONFIG_JSON="$2"
USE_CAPTION_CACHE="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.omniagent_venv/bin/python}"
INPUT_JSONL="${INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl}"
MEDIA_MANIFEST_JSONL="${MEDIA_MANIFEST_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3_sepAV.jsonl}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_audio_global_caption_cache_omniagent_repeat1_completed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_tool_ablation_selectedVal125}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${OUTPUT_ROOT}/${EXPERIMENT}}"
WORKERS="${WORKERS:-4}"
EXPECTED_COUNT="${EXPECTED_COUNT:-${RUN_LIMIT:-125}}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
for required in "${INPUT_JSONL}" "${MEDIA_MANIFEST_JSONL}" "${CONFIG_JSON}"; do
  if [ ! -f "${required}" ]; then
    echo "Required file does not exist: ${required}" >&2
    exit 2
  fi
done
if [ "${USE_CAPTION_CACHE}" = "1" ] && [ ! -d "${CAPTION_CACHE_DIR}" ]; then
  echo "Caption cache directory does not exist: ${CAPTION_CACHE_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
LOG_FILE="${LOG_FILE_OVERRIDE:-${OUTPUT_DIR}/run.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

MAIN_ARGS=(
  --input-jsonl "${INPUT_JSONL}"
  --media-manifest-jsonl "${MEDIA_MANIFEST_JSONL}"
  --experiment-config "${CONFIG_JSON}"
  --output-dir "${OUTPUT_DIR}"
  --workers "${WORKERS}"
)
if [ "${USE_CAPTION_CACHE}" = "1" ]; then
  MAIN_ARGS+=(--caption-cache-dir "${CAPTION_CACHE_DIR}")
fi
if [ -n "${RUN_LIMIT:-}" ]; then
  MAIN_ARGS+=(--limit "${RUN_LIMIT}")
fi

echo "Experiment: ${EXPERIMENT}"
echo "Question subset: ${INPUT_JSONL}"
echo "Media manifest: ${MEDIA_MANIFEST_JSONL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Caption cache: $([ "${USE_CAPTION_CACHE}" = "1" ] && echo "${CAPTION_CACHE_DIR}" || echo disabled)"
echo "Workers: ${WORKERS}; limit: ${RUN_LIMIT:-none}"

"${PYTHON_BIN}" main.py "${MAIN_ARGS[@]}" "$@"

"${PYTHON_BIN}" scripts/aggregate_inference_metrics.py \
  "${OUTPUT_DIR}/output_test.jsonl" \
  --expected-count "${EXPECTED_COUNT}" \
  --output-json "${OUTPUT_DIR}/metrics_summary.json"

"${PYTHON_BIN}" scripts/evaluate_dailyomni_output.py \
  "${OUTPUT_DIR}/output_test.jsonl" \
  --json-output "${OUTPUT_DIR}/accuracy_summary.json"
