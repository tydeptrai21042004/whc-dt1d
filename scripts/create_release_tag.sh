#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(cat "$ROOT/VERSION")"
TAG="${TAG:-v${VERSION}}"
MESSAGE="${MESSAGE:-DT1D-Adapter CNN three-seed reproducibility release ${TAG}}"
cd "$ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "Working tree must be clean." >&2; exit 2; }
python tools/verify_reproducibility_package.py
git tag -a "$TAG" -m "$MESSAGE"
git push origin "$TAG"
echo "Pushed annotated tag: $TAG"
