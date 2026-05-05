#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
MOMAD_LIGHT_IMPORT=1 \
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
python3 tools/data_converter/nuscenes_converter_6s.py nuscenes \
  --root-path ./data/nuscenes \
  --canbus ./data/nuscenes \
  --out-dir ./data/infos/ \
  --extra-tag nuscenes \
  --version v1.0
