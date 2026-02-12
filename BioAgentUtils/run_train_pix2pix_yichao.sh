#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/BioAgentUtils/train_pix2pix_yichao.py"
TRAIN_ROOT="$ROOT_DIR/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object"
VERIFY_ROOT="$ROOT_DIR/Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all_by_object"
RESULTS_ROOT="$ROOT_DIR/results"

echo "[run_train] project root: $ROOT_DIR"
echo "[run_train] train root:   $TRAIN_ROOT"
echo "[run_train] verify root:  $VERIFY_ROOT"
echo "[run_train] results root: $RESULTS_ROOT"
echo "[run_train] args:         $*"

if [[ "${CONDA_DEFAULT_ENV:-}" == "organoid" ]]; then
  echo "[run_train] using active env: organoid"
  PYTHONUNBUFFERED=1 \
    python -u "$SCRIPT_PATH" \
    --train-root "$TRAIN_ROOT" \
    --verify-root "$VERIFY_ROOT" \
    --results-root "$RESULTS_ROOT" \
    "$@"
else
  echo "[run_train] active env is not 'organoid'; using conda run"
  PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n organoid \
    python -u "$SCRIPT_PATH" \
    --train-root "$TRAIN_ROOT" \
    --verify-root "$VERIFY_ROOT" \
    --results-root "$RESULTS_ROOT" \
    "$@"
fi
