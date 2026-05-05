#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ZOO_ROOT="${ROOT}/Bench2DriveZoo"
BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-${ROOT}/../../../Bench2Drive-main}"
CARLA_ROOT="${CARLA_ROOT:-${ROOT}/carla}"

BASE_ROUTES="${BASE_ROUTES:-${BENCH2DRIVE_ROOT}/leaderboard/data/bench2drive220}"
TEAM_AGENT="${ZOO_ROOT}/team_code/vad_b2d_agent.py"
CONFIG_FILE="${CONFIG_FILE:-${ZOO_ROOT}/adzoo/vad/configs/VAD/MomAD_base_e2e_b2d.py}"
CKPT_PATH="${CKPT_PATH:-/path/to/momad_ckpt.pth}"
TEAM_CONFIG="${CONFIG_FILE}+${CKPT_PATH}"

ALGO="vad"
PLANNER_TYPE="traj"
TASK_NUM=1
GPU_RANK="${GPU_RANK:-0}"
PORT="${PORT:-30000}"
TM_PORT="${TM_PORT:-50000}"
SAVE_PATH="${SAVE_PATH:-${ROOT}/leaderboard/eval_bench2drive220_momad_${ALGO}_${PLANNER_TYPE}}"
CHECKPOINT_DIR="${ROOT}/leaderboard/${ALGO}_b2d_${PLANNER_TYPE}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-${CHECKPOINT_DIR}/momad_single.json}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "CKPT not found: ${CKPT_PATH}"
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}"

if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
  python "${ROOT}/tools/split_xml.py" "${BASE_ROUTES}" "${TASK_NUM}" "${ALGO}" "${PLANNER_TYPE}"
  touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
fi

ROUTES="${BASE_ROUTES}_0_${ALGO}_${PLANNER_TYPE}.xml"

export CARLA_ROOT
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH="${PYTHONPATH}:${ROOT}/leaderboard"
export PYTHONPATH="${PYTHONPATH}:${ROOT}/leaderboard/team_code"
export PYTHONPATH="${PYTHONPATH}:${ROOT}/scenario_runner"
export SCENARIO_RUNNER_ROOT="${ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${ROOT}/leaderboard"
export CHALLENGE_TRACK_CODENAME="SENSORS"
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=True
export IS_BENCH2DRIVE=True

CUDA_VISIBLE_DEVICES="${GPU_RANK}" python "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTES}" \
  --repetitions="${REPETITIONS}" \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug="${DEBUG_CHALLENGE}" \
  --resume="${RESUME}" \
  --port="${PORT}" \
  --traffic-manager-port="${TM_PORT}" \
  --gpu-rank="${GPU_RANK}"
