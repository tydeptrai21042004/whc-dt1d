#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(cat "$ROOT/VERSION")"
TAG="${TAG:-v${VERSION}}"
NOTES_FILE="${NOTES_FILE:-}"
command -v gh >/dev/null || { echo "GitHub CLI (gh) is required." >&2; exit 2; }
cd "$ROOT"
bash scripts/build_release_archive.sh
args=(
  release create "$TAG"
  --title "DT1D-Adapter reproducibility release $TAG"
)
if [[ -n "$NOTES_FILE" ]]; then
  [[ -f "$NOTES_FILE" ]] || { echo "NOTES_FILE does not exist: $NOTES_FILE" >&2; exit 2; }
  args+=(--notes-file "$NOTES_FILE")
else
  args+=(--notes "DT1D-Adapter $TAG reproducibility release: self-contained experiment configs, shared method-level validation LR selection across seeds, reviewer ablations, and release validation.")
fi
args+=(
  "dist/DT1D-Adapter-v${VERSION}.zip"
  "dist/DT1D-Adapter-v${VERSION}.tar.gz"
  dist/SHA256SUMS.txt
)
gh "${args[@]}"
