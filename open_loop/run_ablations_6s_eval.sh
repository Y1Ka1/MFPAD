#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

WOMEMORY_DIR="${WOMEMORY_DIR:-${SCRIPT_DIR}/work_dirs/momad_6s_womemory}"
WOFORGETTING_DIR="${WOFORGETTING_DIR:-${SCRIPT_DIR}/work_dirs/momad_6s_woforgetting}"

# w/o Memory
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
python3 tools/test.py \
  projects/configs/MomAD_small_stage2_roboAD_6s_womemory.py \
  "${WOMEMORY_DIR}/latest.pth" \
  --eval bbox \
  --cfg-options evaluation.eval_mode.with_planning=True \
               evaluation.eval_mode.with_motion=False \
               evaluation.eval_mode.with_det=False \
               evaluation.eval_mode.with_map=False \
               evaluation.eval_mode.with_tracking=False \
               data.samples_per_gpu=1 data.workers_per_gpu=2 \
  --out "${WOMEMORY_DIR}/results.pkl"

# w/o Forgetting
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
python3 tools/test.py \
  projects/configs/MomAD_small_stage2_roboAD_6s_woforgetting.py \
  "${WOFORGETTING_DIR}/latest.pth" \
  --eval bbox \
  --cfg-options evaluation.eval_mode.with_planning=True \
               evaluation.eval_mode.with_motion=False \
               evaluation.eval_mode.with_det=False \
               evaluation.eval_mode.with_map=False \
               evaluation.eval_mode.with_tracking=False \
               data.samples_per_gpu=1 data.workers_per_gpu=2 \
  --out "${WOFORGETTING_DIR}/results.pkl"
