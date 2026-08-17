#!/usr/bin/env bash
set -Eeuo pipefail
echo "[deprecated] Use KAGGLE_CNN_REVIEWER_RUN.sh. Tables remain >=3 seeds; figures are exactly one seed." >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/KAGGLE_CNN_REVIEWER_RUN.sh" "$@"
