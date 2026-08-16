#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python proposal_fingerprint.py
python tools/validate_dt1d.py
python tools/validate_all_configs.py
python tools/verify_reproducibility_package.py
python -m compileall -q main.py dense_main.py engine.py utils.py memory_utils.py proposal_contract.py proposal_fingerprint.py models datasets dense_prediction tools tests

# Fast, offline-safe release gate. The full suite contains intentional end-to-end
# training tests that can take several minutes on CPU.
python -m pytest -q \
  tests/test_dt1d_adapter.py \
  tests/test_dt1d_robustness_extended.py \
  tests/test_proposal_contract.py \
  tests/test_reproducibility_package.py \
  tests/test_fair_protocol_contract.py -k 'not two_seed_smoke' \
  tests/test_shell_scripts.py

while IFS= read -r -d '' file; do bash -n "$file"; done < <(find . -maxdepth 5 -type f -name '*.sh' -print0)

if [[ "${RUN_FULL_TESTS:-0}" == "1" ]]; then
  python -m pytest -q tests
fi

echo "CNN release validation passed. Set RUN_FULL_TESTS=1 for the long CPU suite."
