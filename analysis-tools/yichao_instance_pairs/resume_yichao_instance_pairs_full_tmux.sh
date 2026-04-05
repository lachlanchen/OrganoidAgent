#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_ROOT="${1:-$ROOT/analysis-outputs/yichao_instance_pairs}"
SESSION_NAME="${2:-yichao_instance_pairs_full}"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/full_run.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "cd \"$ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "export PYTHONNOUSERSITE=1" C-m
tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=0" C-m
tmux send-keys -t "$SESSION_NAME" "exec > >(tee -a \"$LOG_PATH\") 2>&1" C-m
tmux send-keys -t "$SESSION_NAME" "echo RESUME_START \$(date -Iseconds)" C-m
tmux send-keys -t "$SESSION_NAME" "bash \"$ROOT/analysis-tools/yichao_instance_pairs/run_yichao_instance_pair_extraction.sh\" --gpu true --output-root \"$OUTPUT_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "bash \"$ROOT/analysis-tools/yichao_instance_pairs/build_yichao_instance_pair_database.sh\" --output-root \"$OUTPUT_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "echo RESUME_FINISHED \$(date -Iseconds)" C-m

echo "started tmux session: $SESSION_NAME"
echo "log: $LOG_PATH"
