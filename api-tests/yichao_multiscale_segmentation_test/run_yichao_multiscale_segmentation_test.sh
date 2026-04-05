#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONNOUSERSITE=1
conda run --no-capture-output -n organoid python \
  "$ROOT/api-tests/yichao_multiscale_segmentation_test/run_yichao_multiscale_segmentation_test.py" \
  "$@"
