#!/usr/bin/env bash
# Backward-compatible wrapper; use vkdet_test_stage2.sh in new commands.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/vkdet_test_stage2.sh" "$@"
