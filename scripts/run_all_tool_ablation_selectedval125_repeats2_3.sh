#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMON_LAUNCHER="${COMMON_LAUNCHER_OVERRIDE:-${SCRIPT_DIR}/run_tool_ablation_common.sh}"

DATA_ROOT="${DATA_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
REPEAT2_OUTPUT_ROOT="${REPEAT2_OUTPUT_ROOT:-${DATA_ROOT}/OmniAgent_tool_ablation_selectedVal125_repeat2}"
REPEAT3_OUTPUT_ROOT="${REPEAT3_OUTPUT_ROOT:-${DATA_ROOT}/OmniAgent_tool_ablation_selectedVal125_repeat3}"
WORKERS_PER_EXPERIMENT="${WORKERS_PER_EXPERIMENT:-4}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.omniagent_venv/bin/python}"

if ! [[ "${WORKERS_PER_EXPERIMENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS_PER_EXPERIMENT must be a positive integer: ${WORKERS_PER_EXPERIMENT}" >&2
  exit 2
fi
if [ ! -x "${COMMON_LAUNCHER}" ]; then
  echo "Common launcher is not executable: ${COMMON_LAUNCHER}" >&2
  exit 2
fi
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "Required command not found: setsid" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "Required command not found: flock" >&2
  exit 2
fi

mkdir -p "${DATA_ROOT}"
LOCK_FILE="${DATA_ROOT}/.OmniAgent_tool_ablation_selectedVal125_repeats2_3.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another repeat2/repeat3 launcher is already running (lock: ${LOCK_FILE})." >&2
  exit 2
fi

EXPERIMENTS=(
  "minus_a|configs/tool_ablation/minus_a.json|1"
  "minus_e|configs/tool_ablation/minus_e.json|1"
  "minus_a_e|configs/tool_ablation/minus_a_e.json|1"
  "minus_c|configs/tool_ablation/minus_c.json|0"
  "minus_q|configs/tool_ablation/minus_q.json|1"
  "minus_m|configs/tool_ablation/minus_m.json|1"
  "minimal|configs/tool_ablation/minimal.json|1"
)

ACTIVE_PIDS=()

terminate_active_jobs() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      # Each task is started under setsid, so its PID is also its process-group ID.
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
}

on_signal() {
  echo "Interrupted; terminating all active ablation tasks..." >&2
  terminate_active_jobs
  exit 130
}

trap on_signal INT TERM

run_repeat() {
  local repeat_name="$1"
  local output_root="$2"
  local -a names=()
  local -a pids=()
  local spec experiment config_relative use_cache config_path task_log pid index status
  local failed=0

  mkdir -p "${output_root}"
  echo
  echo "=== Starting ${repeat_name} ==="
  echo "Output root: ${output_root}"
  echo "Parallel experiments: ${#EXPERIMENTS[@]}"
  echo "Workers per experiment: ${WORKERS_PER_EXPERIMENT}"
  echo "Maximum question workers in this repeat: $(( ${#EXPERIMENTS[@]} * WORKERS_PER_EXPERIMENT ))"

  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment config_relative use_cache <<<"${spec}"
    config_path="${REPO_DIR}/${config_relative}"
    if [ ! -f "${config_path}" ]; then
      echo "Experiment config does not exist: ${config_path}" >&2
      terminate_active_jobs
      return 2
    fi

    mkdir -p "${output_root}/${experiment}"
    task_log="${output_root}/${experiment}/parallel_launcher.log"
    echo "Launching ${repeat_name}/${experiment}; console log: ${task_log}"

    setsid env \
      -u OUTPUT_DIR_OVERRIDE \
      -u OUTPUT_ROOT \
      -u RUN_LIMIT \
      -u EXPECTED_COUNT \
      -u LOG_FILE_OVERRIDE \
      OUTPUT_ROOT="${output_root}" \
      EXPECTED_COUNT=125 \
      WORKERS="${WORKERS_PER_EXPERIMENT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      "${COMMON_LAUNCHER}" \
      "${experiment}" \
      "${config_path}" \
      "${use_cache}" \
      --no-print-steps \
      >"${task_log}" 2>&1 &
    pid=$!
    names+=("${experiment}")
    pids+=("${pid}")
    ACTIVE_PIDS+=("${pid}")
  done

  for index in "${!pids[@]}"; do
    pid="${pids[${index}]}"
    experiment="${names[${index}]}"
    if wait "${pid}"; then
      echo "Completed ${repeat_name}/${experiment}"
    else
      status=$?
      failed=1
      echo "FAILED ${repeat_name}/${experiment} (exit ${status}); see ${output_root}/${experiment}/parallel_launcher.log" >&2
    fi
  done
  ACTIVE_PIDS=()

  if [ "${failed}" -ne 0 ]; then
    echo "${repeat_name} had failed experiment process(es); stopping the launcher." >&2
    return 1
  fi
  echo "=== Completed ${repeat_name}: all seven experiments ==="
}

echo "Repository: ${REPO_DIR}"
echo "Repeat execution order: repeat2, then repeat3"
echo "Each repeat runs all seven experiments concurrently."

run_repeat "repeat2" "${REPEAT2_OUTPUT_ROOT}"
run_repeat "repeat3" "${REPEAT3_OUTPUT_ROOT}"

echo
echo "Both repeats completed successfully."
echo "Repeat 2: ${REPEAT2_OUTPUT_ROOT}"
echo "Repeat 3: ${REPEAT3_OUTPUT_ROOT}"
