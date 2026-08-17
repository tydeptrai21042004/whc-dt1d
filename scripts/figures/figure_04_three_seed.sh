#!/usr/bin/env bash
set -Eeuo pipefail
echo "[deprecated] figure_04_three_seed.sh: figures now use exactly one seed; forwarding to figure_04_single_seed.sh." >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/figure_04_single_seed.sh" "$@"
