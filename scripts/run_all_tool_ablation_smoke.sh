#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../smoke_outputs/tool_ablation_5q}"
export RUN_LIMIT=5
export EXPECTED_COUNT=5
export WORKERS="${WORKERS:-5}"

for launcher in \
  run_ablation_minus_a.sh \
  run_ablation_minus_e.sh \
  run_ablation_minus_a_e.sh \
  run_ablation_minus_c.sh \
  run_ablation_minus_q.sh \
  run_ablation_minus_m.sh \
  run_ablation_minimal.sh; do
  "${SCRIPT_DIR}/${launcher}" --no-print-steps "$@"
done
