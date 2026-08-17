#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-0,1,2}"; DATA_DIR="${DATA_DIR:-$ROOT/data}"; DEVICE="${DEVICE:-cuda}"; OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/cnn_paper_revision}"
python -c 'import sys; s=[x.strip() for x in sys.argv[1].split(",") if x.strip()]; assert len(s)>=3 and len(s)==len(set(s)), "Table policy requires at least three unique seeds"' "$SEEDS"
python "$ROOT/tools/run_cnn_paper.py" --target tables --seeds "$SEEDS" --data-path "$DATA_DIR" --device "$DEVICE" --output-root "$OUTPUT_ROOT" --skip-if-complete
python "$ROOT/tools/aggregate_cnn_paper.py" --root "$OUTPUT_ROOT" --require-seeds "$SEEDS"
