#!/usr/bin/env bash
set -euo pipefail

CONFIG="$1"
GPUS="$2"
shift 2

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PORT="${PORT:-29500}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    tools/train.py \
    --config "$CONFIG" \
    --launcher pytorch \
    "$@"
