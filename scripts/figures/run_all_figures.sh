#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIGURE_SEED="${FIGURE_SEED:-0}"; export FIGURE_SEED
bash "$ROOT/scripts/figures/figure_01_single_seed.sh"
bash "$ROOT/scripts/figures/figure_02_deterministic.sh"
bash "$ROOT/scripts/figures/figure_03_deterministic.sh"
bash "$ROOT/scripts/figures/figure_04_single_seed.sh"
