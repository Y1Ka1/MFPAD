#!/usr/bin/env bash
set -euo pipefail

if [ -z "${NAVSIM_DEVKIT_ROOT:-}" ]; then
  echo "Please set NAVSIM_DEVKIT_ROOT to your official NAVSIM v1.1 checkout."
  exit 1
fi

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
NAVSIM_AGENT="${NAVSIM_AGENT:-mfpad_navsim_agent_template}"
NAVSIM_WORKER="${NAVSIM_WORKER:-single_machine_thread_pool}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mfpad_navsim_train}"

cd "${NAVSIM_DEVKIT_ROOT}"

python navsim/planning/script/run_training.py \
  "train_test_split=${TRAIN_TEST_SPLIT}" \
  "agent=${NAVSIM_AGENT}" \
  "worker=${NAVSIM_WORKER}" \
  "experiment_name=${EXPERIMENT_NAME}" \
  "$@"
