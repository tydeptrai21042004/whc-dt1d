# Kaggle run instructions

## Fresh session

1. Enable a Kaggle GPU accelerator and Internet access.
2. Clone or upload the complete repository.
3. Enter the repository root.
4. Install the Kaggle requirements.

```bash
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
```

## Validate before a real run

```bash
python tools/verify_reproducibility_package.py
python tools/validate_dt1d.py
python tools/validate_all_configs.py
pytest -q
```

The grouped release validator, including CPU smoke training, is also available:

```bash
bash scripts/validate_release.sh
```

## Run one experiment

```bash
python tools/run_experiment.py configs/experiments/table_05.yaml \
  --data-path /kaggle/working/data \
  --device cuda \
  --output-root /kaggle/working/dt1d_results
```

Resume completed methods/seeds safely:

```bash
python tools/run_experiment.py configs/experiments/table_05.yaml \
  --data-path /kaggle/working/data \
  --device cuda \
  --output-root /kaggle/working/dt1d_results \
  --skip-if-complete
```

A partially completed method is rerun rather than mixed into a new shared-LR decision. A method is reused only when its method-level LR selection summary and requested final seed results are complete.

## Run the complete classification revision

The convenience shell script installs dependencies, validates the package, runs all main configs, runs both reviewer-ablation configs, and creates a ZIP of the output directory:

```bash
DATA_DIR=/kaggle/working/data \
OUTPUT_ROOT=/kaggle/working/dt1d_revision \
TABLE_SEEDS=0,1,2 \
FIGURE_SEED=0 \
DEVICE=cuda \
bash KAGGLE_CNN_REVIEWER_RUN.sh
```

Publication tables require at least three unique seeds (current release: `0,1,2`); training-based figures require exactly one representative seed (current release: `0`).

## Plan first

To inspect the exact run count without training:

```bash
python tools/run_cnn_paper.py \
  --target all-with-ablations \
  --data-path /kaggle/working/data \
  --device cuda \
  --plan-only
```

`execution_plan.json` records the method list, seeds, LR grid, number of validation runs, number of final test runs, and the shared method-level LR-selection rule.

## Output to retain

Keep the full experiment output, especially:

- `experiment_config.yaml`;
- `execution_plan.json`;
- `<method>/lr_selection_summary.json`;
- each seed's `resolved_config.json`;
- `selection_summary.json`;
- `convergence_summary.json`;
- `test_summary.json`;
- `run_status.json`;
- efficiency profile and logs.

These files are sufficient to audit how the reported test value was selected.
