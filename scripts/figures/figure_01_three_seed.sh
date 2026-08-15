#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEEDS="${SEEDS:-0,1,2}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/cnn_paper_three_seed}"
python "$ROOT/tools/run_cnn_paper.py" --target figure_01 --seeds "$SEEDS" --data-path "$DATA_DIR" --device "$DEVICE" --output-root "$OUTPUT_ROOT" --skip-if-complete
python "$ROOT/tools/aggregate_cnn_paper.py" --root "$OUTPUT_ROOT" --target figure_01 --require-seeds "$SEEDS"
python "$ROOT/tools/plot_figure_01_three_seed.py" --root "$OUTPUT_ROOT" --require-seeds "$SEEDS"
