#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
TABLE_SEEDS="${TABLE_SEEDS:-0,1,2}"; FIGURE_SEED="${FIGURE_SEED:-0}"; DATA_DIR="${DATA_DIR:-$ROOT/data}"; DEVICE="${DEVICE:-cuda}"; OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/kaggle_dt1d_revision}"; RUN_VALIDATION="${RUN_VALIDATION:-1}"
python -c 'import sys; s=[x.strip() for x in sys.argv[1].split(",") if x.strip()]; f=[x.strip() for x in sys.argv[2].split(",") if x.strip()]; assert len(s)>=3 and len(s)==len(set(s)); assert len(f)==1' "$TABLE_SEEDS" "$FIGURE_SEED"
python -m pip install -q --upgrade-strategy only-if-needed -r "$ROOT/requirements-kaggle.txt"
if [[ "$RUN_VALIDATION" == "1" ]]; then bash scripts/validate_release.sh; else python tools/verify_reproducibility_package.py; python tools/validate_dt1d.py; python tools/validate_all_configs.py; fi
SEEDS="$TABLE_SEEDS" DATA_DIR="$DATA_DIR" DEVICE="$DEVICE" OUTPUT_ROOT="$OUTPUT_ROOT/main" bash scripts/run_all_cnn_tables_three_seed.sh
FIGURE_SEED="$FIGURE_SEED" OUTPUT_ROOT="$OUTPUT_ROOT/main" DATA_DIR="$DATA_DIR" DEVICE="$DEVICE" bash scripts/figures/run_all_figures.sh
python tools/run_cnn_paper.py --target ablations --seeds "$TABLE_SEEDS" --data-path "$DATA_DIR" --device "$DEVICE" --output-root "$OUTPUT_ROOT/ablations" --skip-if-complete
python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT/ablations" --require-seeds "$TABLE_SEEDS"
python -c 'import shutil,sys; from pathlib import Path; root=Path(sys.argv[1]); print(shutil.make_archive(str(root),"zip",root_dir=root.parent,base_dir=root.name))' "$OUTPUT_ROOT"
