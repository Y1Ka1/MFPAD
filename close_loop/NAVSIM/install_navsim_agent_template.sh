#!/usr/bin/env bash
set -euo pipefail

if [ -z "${NAVSIM_DEVKIT_ROOT:-}" ]; then
  echo "Please set NAVSIM_DEVKIT_ROOT to your official NAVSIM v1.1 checkout."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/templates"
AGENT_DST_DIR="${NAVSIM_DEVKIT_ROOT}/navsim/agents"
CFG_DST_DIR="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/config/common/agent"

AGENT_SRC="${TEMPLATE_DIR}/mfpad_navsim_agent_template.py"
CFG_SRC="${TEMPLATE_DIR}/mfpad_navsim_agent_template.yaml"
AGENT_DST="${AGENT_DST_DIR}/mfpad_navsim_agent_template.py"
CFG_DST="${CFG_DST_DIR}/mfpad_navsim_agent_template.yaml"

if [ ! -f "${AGENT_SRC}" ] || [ ! -f "${CFG_SRC}" ]; then
  echo "Template files are missing under ${TEMPLATE_DIR}"
  exit 1
fi

if [ ! -d "${AGENT_DST_DIR}" ]; then
  echo "NAVSIM agent directory not found: ${AGENT_DST_DIR}"
  exit 1
fi

if [ ! -d "${CFG_DST_DIR}" ]; then
  echo "NAVSIM agent config directory not found: ${CFG_DST_DIR}"
  exit 1
fi

cp "${AGENT_SRC}" "${AGENT_DST}"
cp "${CFG_SRC}" "${CFG_DST}"

echo "Installed NAVSIM adapter template:"
echo "  ${AGENT_DST}"
echo "  ${CFG_DST}"
echo
echo "You can now use:"
echo "  NAVSIM_AGENT=mfpad_navsim_agent_template"
