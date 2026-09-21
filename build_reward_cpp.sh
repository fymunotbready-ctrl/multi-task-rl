#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
CXX=${CXX:-c++}

EXT_SUFFIX=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
"$CXX" -O3 -Wall -shared -std=c++17 -fPIC \
  $("$PYTHON" -m pybind11 --includes) \
  reward.cpp \
  -o "reward_cpp${EXT_SUFFIX}"
