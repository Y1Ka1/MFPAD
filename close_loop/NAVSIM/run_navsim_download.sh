#!/usr/bin/env bash
set -euo pipefail

if [ -z "${NAVSIM_DEVKIT_ROOT:-}" ]; then
  echo "Please set NAVSIM_DEVKIT_ROOT to your official NAVSIM v1.1 checkout."
  exit 1
fi

DOWNLOAD_DIR="${NAVSIM_DEVKIT_ROOT}/download"
TARGETS="${NAVSIM_DOWNLOAD_TARGETS:-maps navtrain test}"

if [ ! -d "${DOWNLOAD_DIR}" ]; then
  echo "Download directory not found: ${DOWNLOAD_DIR}"
  exit 1
fi

cd "${DOWNLOAD_DIR}"

for target in ${TARGETS}; do
  case "${target}" in
    maps)
      script="download_maps.sh"
      ;;
    mini)
      script="download_mini.sh"
      ;;
    trainval)
      script="download_trainval.sh"
      ;;
    test)
      script="download_test.sh"
      ;;
    navtrain)
      script="download_navtrain.sh"
      ;;
    private_test_e2e)
      script="download_private_test_e2e.sh"
      ;;
    *)
      echo "Unknown NAVSIM download target: ${target}"
      exit 1
      ;;
  esac

  if [ ! -f "${script}" ]; then
    echo "Missing official NAVSIM download script: ${DOWNLOAD_DIR}/${script}"
    exit 1
  fi

  echo "Running ${script}"
  bash "./${script}"
done
