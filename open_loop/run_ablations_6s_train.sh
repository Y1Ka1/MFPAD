#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

WOMEMORY_DIR="${WOMEMORY_DIR:-${SCRIPT_DIR}/work_dirs/momad_6s_womemory}"
WOFORGETTING_DIR="${WOFORGETTING_DIR:-${SCRIPT_DIR}/work_dirs/momad_6s_woforgetting}"

# w/o Memory (MLP-only)
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
python3 tools/train.py projects/configs/MomAD_small_stage2_roboAD_6s_womemory.py \
  --work-dir "${WOMEMORY_DIR}" \
  --no-validate

# w/o Forgetting (Memory only)
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
python3 tools/train.py projects/configs/MomAD_small_stage2_roboAD_6s_woforgetting.py \
  --work-dir "${WOFORGETTING_DIR}" \
  --no-validate
