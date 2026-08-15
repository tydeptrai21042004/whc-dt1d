#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python tools/verify_reproducibility_package.py
python tools/validate_dt1d.py
python tools/validate_all_configs.py
python -m compileall -q main.py dense_main.py engine.py datasets dense_prediction models tools tests

# Split the suite into fresh pytest processes. This keeps release validation
# deterministic and bounds per-process resource accumulation.
pytest -q \
  tests/test_aggregation_seed_enforcement.py \
  tests/test_all_config_validation.py \
  tests/test_baseline_integration.py \
  tests/test_baseline_modules.py \
  tests/test_baseline_preservation.py \
  tests/test_cnn_runner.py \
  tests/test_documentation_contract.py \
  tests/test_dt1d_adapter.py \
  tests/test_efficiency_benchmark.py \
  tests/test_execution_order_and_config_inventory.py \
  tests/test_fair_protocol_contract.py

pytest -q \
  tests/test_full_linear_trainability.py \
  tests/test_hook_adapter_connectivity.py \
  tests/test_plain_axial_depthwise.py \
  tests/test_reproducibility_package.py \
  tests/test_reviewer_ablation_manifest.py \
  tests/test_reviewer_axial_routing.py \
  tests/test_shell_scripts.py \
  tests/test_three_way_splits.py

pytest -q \
  tests/test_amp_dtype_safety.py \
  tests/test_dense_datasets.py \
  tests/test_dense_direct_configs.py \
  tests/test_dense_end_to_end_fake.py \
  tests/test_dense_manifest.py \
  tests/test_dense_models.py \
  tests/test_end_to_end_fake_training.py \
  tests/test_run_dense_paper_selection.py

while IFS= read -r -d '' file; do
  bash -n "$file"
done < <(find . -maxdepth 5 -type f -name '*.sh' -print0)

rm -rf .pytest_cache
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete

echo "Release validation passed."
