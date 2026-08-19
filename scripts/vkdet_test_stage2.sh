#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
CHECKPOINT="${1:-workdirs/dior/stage2/epoch_12.pth}"
CONFIG="configs/dior/vkdet_dior_stage2_test.py"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/test.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --launcher none \
    --eval mAP
