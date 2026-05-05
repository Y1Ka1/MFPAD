#!/usr/bin/env bash
set -euo pipefail

if [ -z "${NAVSIM_DEVKIT_ROOT:-}" ]; then
  echo "Please set NAVSIM_DEVKIT_ROOT to your official NAVSIM v1.1 checkout."
  exit 1
fi

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
NAVSIM_AGENT="${NAVSIM_AGENT:-constant_velocity_agent}"
NAVSIM_WORKER="${NAVSIM_WORKER:-single_machine_thread_pool}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mfpad_navsim_eval}"

EXTRA_OVERRIDES=()
if [ -n "${CKPT_PATH:-}" ]; then
  EXTRA_OVERRIDES+=("agent.checkpoint_path=${CKPT_PATH}")
fi

cd "${NAVSIM_DEVKIT_ROOT}"

python navsim/planning/script/run_pdm_score.py \
  "train_test_split=${TRAIN_TEST_SPLIT}" \
  "agent=${NAVSIM_AGENT}" \
  "worker=${NAVSIM_WORKER}" \
  "experiment_name=${EXPERIMENT_NAME}" \
  "${EXTRA_OVERRIDES[@]}" \
  "$@"
