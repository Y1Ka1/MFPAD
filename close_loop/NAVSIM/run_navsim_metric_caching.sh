#!/usr/bin/env bash
set -euo pipefail

if [ -z "${NAVSIM_DEVKIT_ROOT:-}" ]; then
  echo "Please set NAVSIM_DEVKIT_ROOT to your official NAVSIM v1.1 checkout."
  exit 1
fi

SCRIPT_PATH="${NAVSIM_DEVKIT_ROOT}/scripts/run_metric_caching.sh"

if [ ! -f "${SCRIPT_PATH}" ]; then
  echo "Official NAVSIM metric caching script not found: ${SCRIPT_PATH}"
  exit 1
fi

cd "${NAVSIM_DEVKIT_ROOT}/scripts"
bash "./run_metric_caching.sh" "$@"
