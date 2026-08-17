# DT1D-Adapter

Reproducibility code for **DT1D-Adapter: Lightweight Axial Filtering for Parameter-Efficient Fine-Tuning of Visual Backbones**.

## Method identity

The repository has exactly one proposal method:

```text
--tuning_method dt1d
```

The canonical implementation is `models/dt1d_adapter.py`. Its frozen realization is **R124-P2-G16-Axis-LearnedGate**:

| Component | Setting |
|---|---|
| Axes | H + W |
| Group size | 16 |
| Base offsets | {1, 2, 4} |
| Channel detail | normalized `psi4` |
| Weighted shift | p = 2 |
| Shift coefficient | learned per axis, init 0, bounded to [-0.5, 0.5] |
| Joint H/W L1 cap | 1 |
| Residual gate | learned, init 0.01 |
| Pointwise channel mixing | off |
| Padding | replicate |
| Effective kernel | 13 |
| Axial convolution calls | 2 |



## Cross-repository proposal lock

`proposal_spec.json` is the frozen machine-readable method contract shared byte-for-byte with `tydeptrai21042004/whc-vit`. It does not implement the model; it prevents accidental architecture drift.

```bash
python proposal_fingerprint.py
python proposal_fingerprint.py --compare /path/to/whc-vit
```

Every final run records the repository version, Git commit (when the checkout contains `.git`), and proposal-contract SHA256. Inference kernel caching may be enabled for latency measurement; caching changes execution only and not the DT1D operator or learned parameters.

## Preserved baselines

Previous proposal/compatibility code is removed, but the comparison baselines are preserved. The manuscript configs still contain the same applicable rows for **Full fine-tuning, Linear probing, BitFit, SSF, BAM, LoRA-Conv, Residual Adapter, Side-Tuning, and Conv-Adapter budget variants**. `tests/test_baseline_preservation.py` locks the exact method inventory and order for every main experiment.

## Fair classification protocol

Every classification experiment is one complete YAML file in `configs/experiments/`. There are no inherited config fragments and no generated per-seed YAML files.

For each method, `tools/run_experiment.py`:

1. trains every declared LR candidate for all requested seeds with the same dataset split, preprocessing, epoch budget, batch size, optimizer, and scheduler;
2. selects **one LR for the method** using mean best-validation Acc1 across the seeds;
3. uses the lower LR as the deterministic tie-break;
4. selects the best checkpoint within each seed using validation only;
5. evaluates each seed's test split exactly once after method-level LR selection;
6. records the full LR-selection trace and resolved arguments.

The test split is never used to choose a learning rate, checkpoint, architecture, or ablation setting.


## Quick validation

```bash
python tools/verify_reproducibility_package.py
python tools/validate_dt1d.py
python tools/validate_all_configs.py
python -m pytest -q
```

The release helper additionally performs CPU smoke runs:

```bash
bash scripts/validate_release.sh
```

## Run one experiment

Plan without training:

```bash
python tools/run_experiment.py configs/experiments/table_05.yaml --plan-only
```

Run on GPU:

```bash
python tools/run_experiment.py configs/experiments/table_05.yaml \
  --data-path ./data \
  --device cuda \
  --output-root ./outputs/experiments
```

Run all main classification experiments:

```bash
python tools/run_cnn_paper.py \
  --target all \
  --data-path ./data \
  --device cuda \
  --skip-if-complete
```

Run reviewer ablations (including the earlier plain axial depthwise side-by-side control requested during review):

```bash
python tools/run_cnn_paper.py \
  --target ablations \
  --data-path ./data \
  --device cuda \
  --skip-if-complete
```

## Result semantics

The primary classification metric is:

```text
Test Acc1 using the shared method LR selected from validation across seeds,
with the best-validation checkpoint selected within each seed.
```

`test_summary.json` records the selected LR, validation accuracy, selected checkpoint epoch, LR-selection scope, and the fact that test data were not used for hyperparameter selection.

## Repository guide

- `models/dt1d_adapter.py` — canonical proposal.
- `configs/experiments/` — one self-contained YAML per classification experiment.
- `tools/run_experiment.py` — fair validation-selection/test-once runner.
- `tools/run_cnn_paper.py` — multi-experiment launcher.
- `tools/aggregate_cnn_paper.py` — multi-seed table aggregation and single-seed figure summaries.
- `configs/dense/index.yaml` plus `configs/dense/experiments/*.yaml` — exploratory dense-prediction experiments.
- `tests/` — unit, protocol, config, split, integration, and smoke tests.

See `EXPERIMENTS.md`, `REPRODUCIBILITY.md`, `DATASETS.md`, and `KAGGLE_RUN_INSTRUCTIONS.md` for the remaining operational details.
