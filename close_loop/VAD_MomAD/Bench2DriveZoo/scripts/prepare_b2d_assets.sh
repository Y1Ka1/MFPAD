#!/bin/bash
set -euo pipefail

SRC_DIR="${1:-/path/to/b2d_assets}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFO_DST="${ZOO_ROOT}/data/infos"
KMEANS_DST="${ZOO_ROOT}/data/kmeans"
SPLIT_DST="${ZOO_ROOT}/data/splits"
CKPT_DST="${ZOO_ROOT}/ckpts"

mkdir -p "${INFO_DST}" "${KMEANS_DST}" "${SPLIT_DST}" "${CKPT_DST}"

required_infos=(
  "b2d_infos_train.pkl"
  "b2d_infos_val.pkl"
  "b2d_map_infos.pkl"
)
required_kmeans=(
  "kmeans_det_900.npy"
  "kmeans_map_100.npy"
  "kmeans_motion_6.npy"
  "kmeans_plan_1.npy"
  "kmeans_plan_3.npy"
  "kmeans_plan_6.npy"
)
required_ckpts=(
  "momad_small_b2d_stage1.pth"
)

missing=0
for f in "${required_infos[@]}" "${required_kmeans[@]}" "${required_ckpts[@]}"; do
  if [ ! -f "${SRC_DIR}/${f}" ]; then
    echo "Missing: ${SRC_DIR}/${f}"
    missing=1
  fi
done

if [ "${missing}" -ne 0 ]; then
  echo "One or more required files are missing. Aborting."
  exit 1
fi

cp -f "${SRC_DIR}/b2d_infos_train.pkl" "${INFO_DST}/"
cp -f "${SRC_DIR}/b2d_infos_val.pkl" "${INFO_DST}/"
cp -f "${SRC_DIR}/b2d_map_infos.pkl" "${INFO_DST}/"

for f in "${required_kmeans[@]}"; do
  cp -f "${SRC_DIR}/${f}" "${KMEANS_DST}/"
done

cp -f "${SRC_DIR}/momad_small_b2d_stage1.pth" "${CKPT_DST}/"

if [ -f "${SRC_DIR}/bench2drive_base_train_val_split.json" ]; then
  cp -f "${SRC_DIR}/bench2drive_base_train_val_split.json" "${SPLIT_DST}/"
fi

echo "Done. Assets copied into Bench2DriveZoo."
