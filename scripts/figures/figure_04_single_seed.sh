#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIGURE_SEED="${FIGURE_SEED:-0}"; DATA_DIR="${DATA_DIR:-$ROOT/data}"; DEVICE="${DEVICE:-cuda}"; OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/cnn_paper_revision}"
[[ "$FIGURE_SEED" != *,* ]] || { echo "Figure policy requires exactly one seed" >&2; exit 2; }
python "$ROOT/tools/run_cnn_paper.py" --target figure_04 --seeds "$FIGURE_SEED" --data-path "$DATA_DIR" --device "$DEVICE" --output-root "$OUTPUT_ROOT" --skip-if-complete
python "$ROOT/tools/aggregate_cnn_paper.py" --root "$OUTPUT_ROOT" --target figure_04 --require-seeds "$FIGURE_SEED"
python "$ROOT/tools/plot_figure_04_tradeoff.py" --root "$OUTPUT_ROOT" --seed "$FIGURE_SEED" --output "$OUTPUT_ROOT/figures/figure_04_seed${FIGURE_SEED}.png"
