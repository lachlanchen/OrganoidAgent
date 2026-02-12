#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda run -n organoid python "$ROOT_DIR/BioAgentUtils/train_pix2pix_yichao.py" \
  --train-root "$ROOT_DIR/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object" \
  --verify-root "$ROOT_DIR/Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all_by_object" \
  --results-root "$ROOT_DIR/results" \
  "$@"
