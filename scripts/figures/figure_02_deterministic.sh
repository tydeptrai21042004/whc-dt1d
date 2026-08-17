#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/cnn_paper_revision}"
python "$ROOT/tools/plot_figure_02_spectral.py" --output "$OUTPUT_ROOT/figures/figure_02_dt1d_spectral.png"
