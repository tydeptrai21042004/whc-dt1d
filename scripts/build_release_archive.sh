#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(cat "$ROOT/VERSION")"
DIST="${DIST:-$ROOT/dist}"
NAME="DT1D-Adapter-v${VERSION}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$ROOT"
bash scripts/validate_release.sh
mkdir -p "$DIST" "$STAGE/$NAME"
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct 2>/dev/null || date -d '2026-08-01 00:00:00 UTC' +%s)}"

# Copy publication source while excluding local datasets, stochastic result runs,
# caches, VCS internals, and previously built archives. Deterministic reference
# Figures 2 and 3 and compact validation logs are retained.
tar \
  --exclude='.git' --exclude='data' --exclude='runs' --exclude='outputs' --exclude='dist' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc' \
  --exclude='*.zip' --exclude='*.tar.gz' \
  -cf - . | tar -xf - -C "$STAGE/$NAME"

find "$STAGE/$NAME" -exec touch -h -d "@$SOURCE_DATE_EPOCH" {} +
rm -f "$DIST/$NAME.zip" "$DIST/$NAME.tar.gz" "$DIST/SHA256SUMS.txt"
(
  cd "$STAGE"
  find "$NAME" -type f -print | LC_ALL=C sort | zip -X -q "$DIST/$NAME.zip" -@
)
tar --sort=name --mtime="@$SOURCE_DATE_EPOCH" --owner=0 --group=0 --numeric-owner \
  -C "$STAGE" -cf - "$NAME" | gzip -n > "$DIST/$NAME.tar.gz"
sha256sum "$DIST/$NAME.zip" "$DIST/$NAME.tar.gz" > "$DIST/SHA256SUMS.txt"
cat "$DIST/SHA256SUMS.txt"
