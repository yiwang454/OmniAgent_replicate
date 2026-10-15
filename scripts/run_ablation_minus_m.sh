#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_tool_ablation_common.sh" minus_m "${SCRIPT_DIR}/../configs/tool_ablation/minus_m.json" 1 "$@"
