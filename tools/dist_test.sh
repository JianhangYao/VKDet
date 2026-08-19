#!/usr/bin/env bash
set -euo pipefail

CONFIG="$1"
CHECKPOINT="$2"
GPUS="$3"
shift 3

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PORT="${PORT:-29500}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    tools/test.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --launcher pytorch \
    "$@"
