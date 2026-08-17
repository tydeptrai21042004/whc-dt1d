# Reproducibility protocol

## 1. Proposal identity

There is one proposal: **DT1D-Adapter**, invoked by `--tuning_method dt1d` and implemented by `models/dt1d_adapter.py`.

Reviewer-only implementations are kept separately only for current-component isolation evidence:

- `models/dt1d_ablation_adapter.py` — weighted-shift and L1-projection controls;
- `models/reviewer_axial_routing_adapter.py` — routing/axis/dilation/sharing/gate/pointwise controls.

They must not be described as additional proposals.

No previous-version DT1D implementation, old proposal alias, or compatibility proposal module is shipped in this release.

## 2. One config per experiment

Classification configs live in `configs/experiments/`. Each YAML is self-contained and declares:

- seeds;
- dataset and backbone;
- epoch and batch-size budget;
- LR candidate grid and selection rule;
- preprocessing and optimizer/scheduler settings;
- every compared method and its model-specific arguments;
- deterministic execution order.

The runner does not generate or inherit per-seed YAML fragments. The exact resolved arguments used for a finished run are saved as JSON in its output directory.

## 3. Hyperparameter-selection barrier

For each method:

1. run every LR candidate for all requested seeds with `final_test=false`;
2. require every seed/LR validation run to finish successfully;
3. compute mean `best_val_acc1` for each LR across the requested seeds;
4. select the LR with maximum mean validation accuracy, breaking ties toward the lower LR;
5. for each seed, use that seed's best-validation checkpoint at the shared selected LR;
6. enable the test split and evaluate exactly once for that seed.

This gives one LR per method rather than one LR per seed.

The method-level decision is saved to:

```text
<experiment>/<method>/lr_selection_summary.json
```

Each seed also stores `selection_summary.json`, `resolved_config.json`, `convergence_summary.json`, `test_summary.json`, and `run_status.json`.

## 4. Locked fairness fields

Method-specific config rows may define model structure, such as adapter width, LoRA rank, routing mode, or whether a reviewer control enables pointwise mixing. They may not override the experiment-level training/data budget.



## 5. Seeds and split policy

Classification/reviewer tables declare at least three seeds (the committed release uses `0,1,2`). Training-based manuscript figures declare exactly one representative seed (the committed release uses seed `0`).

- Official train/validation/test partitions are used when available.
- Datasets with official train and test but no validation split create validation only from the training partition; official test stays untouched.
- Caltech101 uses committed disjoint 80/10/10 seed-specific split files under `splits/caltech101/`.
- Generated split manifests are saved with run outputs when splits are generated at runtime.

See `DATASETS.md` for dataset-specific details.

## 6. Validation commands

Structural and config validation:

```bash
python tools/verify_reproducibility_package.py
python tools/validate_dt1d.py
python tools/validate_all_configs.py
```

Full test suite:

```bash
pytest -q
```

Release validation with CPU smoke runs:

```bash
bash scripts/validate_release.sh
```

Plan an experiment without training:

```bash
python tools/run_experiment.py configs/experiments/table_05.yaml --plan-only
```

## 7. Reporting rules

For manuscript tables, report mean ± sample standard deviation across at least three final seeds (the committed release uses 0,1,2) of the test metric produced after validation-only LR selection. For training-based figures, use exactly one representative seed and label it explicitly; do not attach mean/SD to a one-seed figure. Do not report maximum test accuracy observed during training.

Real benchmark tables must be regenerated with the current fair configs. Historical results produced under a different LR or preprocessing protocol must not be mixed into the new comparison table as if they were directly comparable.

## 8. Environment and release metadata

Python dependencies are provided in `requirements.txt` and `requirements-kaggle.txt`; `environment.yml` provides a Conda entry point. Release metadata are in `VERSION`, `CITATION.cff`, `codemeta.json`, and `.zenodo.json`.
