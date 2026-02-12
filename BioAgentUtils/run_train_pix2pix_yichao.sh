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

RUNNING_PIDS="$(pgrep -fa "train_pix2pix_yichao.py" || true)"
if [[ -n "$RUNNING_PIDS" ]]; then
  echo "[run_train] warning: existing training processes detected:"
  echo "$RUNNING_PIDS"
  echo "[run_train] this can make new runs look slow. stop old runs if needed:"
  echo "[run_train]   pkill -f train_pix2pix_yichao.py"
fi

DEFAULT_PREP_ARGS=(
  --image-size 512
  --scan-log-interval 200
  --train-first-pair-per-object
  --train-crops-per-image 4
  --train-crop-size 512
)

if [[ "${CONDA_DEFAULT_ENV:-}" == "organoid" ]]; then
  echo "[run_train] using active env: organoid"
  PYTHONUNBUFFERED=1 \
    python -u "$SCRIPT_PATH" \
    --train-root "$TRAIN_ROOT" \
    --verify-root "$VERIFY_ROOT" \
    --results-root "$RESULTS_ROOT" \
    "${DEFAULT_PREP_ARGS[@]}" \
    "$@"
else
  echo "[run_train] active env is not 'organoid'; attempting inline activate"
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # Prefer inline activation over `conda run` so logs stream directly.
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate organoid
    PYTHONUNBUFFERED=1 \
      python -u "$SCRIPT_PATH" \
      --train-root "$TRAIN_ROOT" \
      --verify-root "$VERIFY_ROOT" \
      --results-root "$RESULTS_ROOT" \
      "${DEFAULT_PREP_ARGS[@]}" \
      "$@"
  else
    echo "[run_train] conda.sh not found; falling back to conda run"
    PYTHONUNBUFFERED=1 \
      conda run --no-capture-output -n organoid \
      python -u "$SCRIPT_PATH" \
      --train-root "$TRAIN_ROOT" \
      --verify-root "$VERIFY_ROOT" \
      --results-root "$RESULTS_ROOT" \
      "${DEFAULT_PREP_ARGS[@]}" \
      "$@"
  fi
fi
