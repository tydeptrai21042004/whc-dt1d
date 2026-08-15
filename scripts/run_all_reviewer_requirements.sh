#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-0,1,2}"; DATA_ROOT="${DATA_ROOT:-$ROOT/data}"; DEVICE="${DEVICE:-cuda}"; PLAN_ONLY="${PLAN_ONLY:-0}"
python "$ROOT/tools/validate_all_configs.py"
extra=(--skip-if-complete); [[ "$PLAN_ONLY" == "1" ]] && extra=(--plan-only)
python "$ROOT/tools/run_cnn_paper.py" --target ablations --seeds "$SEEDS" --data-path "$DATA_ROOT" --device "$DEVICE" --output-root "$ROOT/outputs/reviewer_ablations" "${extra[@]}"
if [[ "$PLAN_ONLY" != "1" ]]; then
  python "$ROOT/tools/benchmark_reviewer_efficiency.py" --methods dt1d,routing_reference,direct_symmetric,plain_axial,full,linear,conv,residual,bam,ssf,lora_conv,bitfit --device "$DEVICE" --input-size 224 --batch-size 1 --warmup 20 --iterations 100 --output "$ROOT/outputs/reviewer_efficiency"
fi
