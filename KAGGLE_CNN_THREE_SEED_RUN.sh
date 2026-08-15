#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-0,1,2}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/kaggle_dt1d_revision}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"

[[ "$SEEDS" == "0,1,2" ]] || {
  echo "Publication runs require SEEDS=0,1,2" >&2
  exit 2
}

python -m pip install -q --upgrade-strategy only-if-needed -r "$ROOT/requirements-kaggle.txt"

if [[ "$RUN_VALIDATION" == "1" ]]; then
  bash scripts/validate_release.sh
else
  python tools/verify_reproducibility_package.py
  python tools/validate_dt1d.py
  python tools/validate_all_configs.py
fi

python tools/run_cnn_paper.py \
  --target all \
  --seeds "$SEEDS" \
  --data-path "$DATA_DIR" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT/main" \
  --skip-if-complete

python tools/run_cnn_paper.py \
  --target ablations \
  --seeds "$SEEDS" \
  --data-path "$DATA_DIR" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT/ablations" \
  --skip-if-complete

python - "$OUTPUT_ROOT" <<'PY'
import shutil
import sys
from pathlib import Path
root = Path(sys.argv[1])
print(shutil.make_archive(str(root), "zip", root_dir=root.parent, base_dir=root.name))
PY
