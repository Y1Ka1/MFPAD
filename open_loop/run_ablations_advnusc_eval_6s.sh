#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

WOMEMORY_DIR="${WOMEMORY_DIR:-${SCRIPT_DIR}/work_dirs/momad_6s_womemory}"
WOFORGETTING_DIR="${WOFORGETTING_DIR:-${SCRIPT_DIR}/work_dirs/momad_6s_woforgetting}"

# w/o Memory
MOMAD_NUSC_ROOT=data/advnusc/ \
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
               data.val.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
               data.test.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
  --out "${WOMEMORY_DIR}/results_advnusc.pkl"

# w/o Forgetting
MOMAD_NUSC_ROOT=data/advnusc/ \
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
               data.val.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
               data.test.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
  --out "${WOFORGETTING_DIR}/results_advnusc.pkl"
