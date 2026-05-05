#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/work_dirs/momad_3s_advnusc_val}"
CKPT_PATH="${CKPT_PATH:-ckpt/MomAD_stage2_3s_iter_29300.pth}"
mkdir -p "${WORK_DIR}"

MOMAD_USE_ALL_SCENES=1 \
MOMAD_NUSC_ROOT=data/advnusc/ \
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
python3 tools/test.py \
  projects/configs/MomAD_small_stage2_roboAD.py \
  "${CKPT_PATH}" \
  --eval bbox \
  --cfg-options \
    data.test.data_root=data/advnusc/ \
    data.val.data_root=data/advnusc/ \
    data.test.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
    data.val.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
    data.test.eval_config.data_root=data/advnusc/ \
    data.val.eval_config.data_root=data/advnusc/ \
    data.test.eval_config.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
    data.val.eval_config.ann_file=data/infos_advnusc/advnusc_infos_val.pkl \
    evaluation.eval_mode.with_planning=True \
    evaluation.eval_mode.with_motion=False \
    evaluation.eval_mode.with_det=False \
    evaluation.eval_mode.with_map=False \
    evaluation.eval_mode.with_tracking=False \
  --out "${WORK_DIR}/results.pkl"
