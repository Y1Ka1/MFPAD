#!/bin/bash
set -euo pipefail

DEST_ROOT="/root/autodl-tmp/carla"
CARLA_VER="0.9.15"
CARLA_TAR="CARLA_${CARLA_VER}.tar.gz"
MAP_TAR="AdditionalMaps_${CARLA_VER}.tar.gz"

mkdir -p "${DEST_ROOT}"
cd "${DEST_ROOT}"

if [ ! -d "CARLA_${CARLA_VER}" ]; then
  wget "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/${CARLA_TAR}"
  tar -xvf "${CARLA_TAR}"
fi

if [ -d "CARLA_${CARLA_VER}/Import" ]; then
  cd "CARLA_${CARLA_VER}/Import"
  if [ ! -f "${MAP_TAR}" ]; then
    wget "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/${MAP_TAR}"
  fi
  cd ..
  if [ -x "./ImportAssets.sh" ]; then
    bash ./ImportAssets.sh
  fi
fi

echo "CARLA_ROOT is: ${DEST_ROOT}/CARLA_${CARLA_VER}"
echo "Remember to export CARLA_ROOT and add the carla egg to your Python path."
