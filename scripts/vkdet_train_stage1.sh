#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-1}"
PORT="${PORT:-29500}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    tools/train.py \
    --config configs/dior/vkdet_dior_stage1.py \
    --launcher pytorch \
    --work-dir workdirs/dior/stage1 \
    --seed 42 \
    --deterministic \
    --no-validate
