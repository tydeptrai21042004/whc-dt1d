# Dataset and split protocol

Use one writable data root:

```bash
export DATA_DIR=/absolute/path/to/data
```

Classification runs pass this directory with `--data-path "$DATA_DIR"`.

## Classification datasets

### Official train/validation/test splits

- **DTD** — official `train`, `val`, and `test` partitions.
- **Flowers102** — official `train`, `val`, and `test` partitions.

No generated index split is used for these two datasets.

### Official test split with validation created only from training data

- **SVHN** — training partition is split into train/validation; official test is untouched.
- **Food-101** — training partition is split into train/validation; official test is untouched.
- **Oxford-IIIT Pet** — `trainval` is split into train/validation; official `test` is untouched.
- **FGVC-Aircraft** — `trainval` is split into train/validation; official `test` is untouched.

The split is deterministic for a given seed. Generated indices are recorded with the run outputs.

### Caltech101

The repository commits a disjoint 80% train / 10% validation / 10% test partition for each publication seed:

```text
splits/caltech101/seed0_holdout20.json
splits/caltech101/seed1_holdout20.json
splits/caltech101/seed2_holdout20.json
```

Each file contains 6,942 train, 868 validation, and 867 test indices over 8,677 images. The loader validates index range, completeness, uniqueness, and pairwise disjointness.

### EuroSAT

EuroSAT is divided deterministically into disjoint 80% train / 10% validation / 10% test subsets per seed. The generated split manifest is recorded with the run so validation and test cannot reuse the same image.

## Pretrained weights and preprocessing

Publication classification configs request torchvision `weights: DEFAULT`, 224×224 input unless the individual config states otherwise, and ImageNet normalization. FakeData smoke tests explicitly use random initialization (`weights: none`) and must never be reported as benchmark results.

Backbone downloads use torchvision's normal model cache. Use the same pinned project environment when reproducing results.

## Split validation

Run:

```bash
pytest -q tests/test_three_way_splits.py tests/test_reproducibility_package.py
```

To create a deterministic split manifest manually:

```bash
python tools/generate_split.py \
  --dataset caltech101 \
  --length 8677 \
  --seed 0 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --output /tmp/caltech101_seed0.json
```

## Dense-prediction datasets

### PennFudan binary segmentation

Expected layout:

```text
DATA_ROOT/PennFudanPed/
├── PNGImages/
└── PedMasks/
```

A deterministic seed-specific 60/20/20 train/validation/test split is used.

### DRIVE retinal-vessel segmentation

Supported layout includes:

```text
DATA_ROOT/DRIVE/
├── training/
│   ├── images/
│   └── 1st_manual/
└── test/
    ├── images/
    └── 1st_manual/
```

The official test set remains untouched; validation is created from official training data.

### Oxford-IIIT Pet segmentation

`torchvision.datasets.OxfordIIITPet` is used with segmentation targets. `trainval` is divided into train/validation and the official test split remains untouched.

### Oxford-IIIT Pet detection

The dense feasibility code deterministically derives one pet bounding box from the trimap mask. The committed dense detection experiment config uses a short sanity-check setting; use `max_test_samples: 0` only when deliberately running the full official test split.

### Synthetic dense datasets

`fake_binary`, `fake_semantic`, and `fake_detection` are execution tests only and must not be cited as empirical results.
