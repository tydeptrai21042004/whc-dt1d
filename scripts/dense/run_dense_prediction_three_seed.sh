#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0,1,2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/dense_prediction}"
python "$ROOT/tools/run_dense_paper.py" \
  --target all --methods target --seeds "$SEEDS" \
  --data-root "$DATA_ROOT" --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" --skip-if-complete
python "$ROOT/tools/aggregate_dense_results.py" \
  --input-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/summary" \
  --require-seeds "$SEEDS"
