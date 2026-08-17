# Experiments and reviewer ablations

`configs/experiments/index.yaml` is the authoritative classification experiment index. Every listed experiment has one complete YAML file containing its data setting, seeds, budget, preprocessing, fairness policy, methods, and method order.

## Main experiment inventory

**Seed policy:** table configs and reviewer ablations use at least three seeds (current release: `0,1,2`); training-based figure configs use exactly one representative seed (current release: `0`).

| Config | Dataset | Backbone | Epochs | Batch | Manuscript target |
|---|---|---|---:|---:|---|
| `table_05.yaml` | Flowers102 | ResNet-18 | 10 | 64 | Table 5 |
| `table_09.yaml` | Oxford-IIIT Pet | ResNet-50 | 10 | 128 | Table 9 |
| `table_12.yaml` | Oxford-IIIT Pet | EfficientNet-B0 | 10 | 64 | Table 12 |
| `figure_04.yaml` | FGVC-Aircraft | ResNet-18 | 10 | 64 | Figure 4 |
| `table_14_15.yaml` | Caltech101 | ResNet-18 | 10 | 64 | Tables 14–15 |
| `figure_01.yaml` | Caltech101 | ResNet-18 | 100 | 64 | Figure 1 |
| `table_06.yaml` | Flowers102 | ResNet-18 | 100 | 64 | Table 6 |
| `table_03.yaml` | DTD | ResNet-50 | 100 | 128 | Table 3 |
| `table_04.yaml` | Flowers102 | ResNet-50 | 100 | 128 | Table 4 |
| `table_18_19.yaml` | EuroSAT | MobileNetV3-Small | 25 | 32 | Tables 18–19 |
| `table_07.yaml` | Flowers102 | ResNet-18 | 100 | 32 | Table 7 |
| `table_13.yaml` | Oxford-IIIT Pet | EfficientNet-B0 | 100 | 64 | Table 13 |
| `table_08.yaml` | SVHN | ResNet-50 | 10 | 128 | Table 8 |
| `table_11.yaml` | Food-101 | EfficientNet-B0 | 10 | 64 | Table 11 |
| `table_10.yaml` | Food-101 | ResNet-18 | 10 | 32 | Table 10 |

Only the `dt1d` row is the proposal. All other rows in these configs are baselines.
 The release tests preserve each table/figure method list exactly, so removing legacy proposal code cannot silently remove a baseline.

## Reviewer ablations

Two independent settings are committed:

| Config | Dataset | Backbone | Seeds |
|---|---|---|---|
| `ablation_dtd_resnet18.yaml` | DTD | ResNet-18 | 0, 1, 2 |
| `ablation_flowers102_resnet50.yaml` | Flowers102 | ResNet-50 | 0, 1, 2 |

The controls isolate the requested design questions:

| Question | Controlled comparison |
|---|---|
| Direct vs shifted symmetric filtering | `routing_reference` vs `direct_symmetric` |
| Shared vs unshared coefficients | `routing_reference` vs `unshared_coefficients` |
| One axis vs two axes | `routing_reference` vs `height_only` |
| Fixed vs learned routing | `routing_reference` vs `fixed_average_routing` |
| One vs multiple dilations | `routing_reference` vs `single_dilation` |
| Residual gate on vs off | `routing_reference` vs `gate_off` |
| Pointwise mixing off vs on | `routing_reference` vs `pointwise_on` |
| Previous plain axial version vs final proposal | `previous_plain_axial` vs `dt1d` |
| Weighted-shift core | `dt1d` vs `dt1d_no_weighted_shift` |
| Joint-L1 stability projection | `dt1d` vs `dt1d_no_l1_projection` |

All non-`dt1d` rows in the ablation files are explicitly marked `reviewer_control: true`.

## Fairness contract


Every method receives the same LR grid. One shared LR is selected for the method by mean best-validation Acc1 across the requested seeds. No test process is started until every required method/seed/LR validation candidate for that method has completed successfully.

## Commands

One experiment:

```bash
python tools/run_experiment.py configs/experiments/table_05.yaml \
  --data-path ./data --device cuda
```

Selected methods or seeds for debugging:

```bash
python tools/run_experiment.py configs/experiments/table_05.yaml \
  --methods dt1d,linear --seeds 0 --smoke
```

All table experiments (>=3 seeds):

```bash
python tools/run_cnn_paper.py --target tables --seeds 0,1,2 --data-path ./data --device cuda
```

All training-based figures (exactly one seed):

```bash
python tools/run_cnn_paper.py --target figures --seeds 0 --data-path ./data --device cuda
```

All reviewer ablations:

```bash
python tools/run_cnn_paper.py --target ablations --data-path ./data --device cuda
```

Aggregate one completed experiment:

```bash
python tools/aggregate_cnn_paper.py \
  --root ./outputs/experiments \
  --target table_05 \
  --require-seeds 0,1,2
```

## Dense prediction

Dense prediction remains exploratory feasibility evidence under `configs/dense/index.yaml` and the six self-contained configs under `configs/dense/experiments/`. Its proposal key is also `dt1d`. The detection target is intentionally a short sanity-check protocol and should not be presented as a definitive detection benchmark.
