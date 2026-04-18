# HYDRA: Explainability-Driven Intrusion Detection

A reproducible experimental pipeline for binary intrusion detection on tabular network-flow data with first-class explainability (global + local SHAP). Supports two datasets (TON_IoT, CIC-IoT-2023), cross-dataset generalisation, and leakage-safe evaluation.

## Repo Structure

```
hydra/                  # Main Python package
├── config/             # Dataset configs (datasets.yaml, defaults.yaml)
├── data/               # Data loading, splitting, preprocessing, alignment
├── models/             # Model builders (logreg, RF, GBDT, XGBoost, LightGBM, CNN-LSTM)
├── experiments/        # Experiment runners (single-dataset, cross-dataset, matrix)
├── evaluation/         # Metrics, thresholds
├── xai/                # SHAP explainability (global, local, per-type)
├── analysis/           # Result aggregation and plotting
└── features/           # Feature engineering
scripts/                # Utility scripts (data prep, overnight runs, figure generation)
tests/                  # Test suite
data/                   # Datasets (gitignored)
results/                # Experiment outputs (gitignored)
models/                 # Saved model weights
docs/                   # Documentation
Makefile                # Experiment targets
```

## Quickstart

Single run (TON_IoT):
```bash
python -m hydra.experiments.run_tabular \
  --dataset ton_iot \
  --feature_regime behaviour_only \
  --split_strategy host \
  --group_col src_ip \
  --seed 42
```

Single run (CIC-IoT-2023):
```bash
python -m hydra.experiments.run_tabular \
  --dataset cic_iot2023 \
  --feature_regime behaviour_only \
  --split_strategy stratified \
  --seed 42
```

Cross-dataset generalisation:
```bash
python -m hydra.experiments.run_cross_dataset \
  --source ton_iot --target cic_iot2023 \
  --models logreg random_forest xgboost \
  --seed 42
```

Full experiment matrix (TON_IoT, 36 runs):
```bash
make ton-matrix
```

Cross-dataset (both directions):
```bash
make cross-dataset
```

## Leakage Safeguards

**Permutation probe:**
```bash
python -m hydra.experiments.run_tabular \
  --dataset ton_iot --feature_regime behaviour_only \
  --split_strategy host --group_col src_ip --seed 42 \
  --label_permutation_probe
```
ROC-AUC under null permutation should be ~0.5. Use `--permutation_repeats N` (default 3) to average.

**Duplicate leakage audit:**
Each run writes `evaluation_meta.json` with train/val/test label counts and post-preprocessing row-hash overlap rates. Use `--duplicate_leakage_threshold` (default 0.001) and `--fail_on_duplicate_leakage` to hard-fail on excessive overlap.

## Datasets

Edit `hydra/config/datasets.yaml` to point to your local dataset paths. Supports `.csv` and `.parquet`.

Required per dataset:
- `path`: local file path
- `label_col`: binary label column
- `positive_label`: value mapped to 1

Optional: `timestamp_col`, `group_col`, `categorical_cols`, `numeric_cols`

### Supported datasets
- **TON_IoT** (~211k rows) -- split strategies: `host`, `temporal`, `stratified`
- **CIC-IoT-2023** (~3.18M rows) -- split strategy: `stratified`

## Models

- Logistic Regression
- Random Forest
- Gradient Boosted Trees (sklearn)
- XGBoost
- LightGBM (optional, falls back to sklearn GBDT if unavailable)
- CNN-LSTM (deep learning, requires PyTorch)

## Explainability

- Global feature importance (SHAP bar plots)
- Local explanations (SHAP beeswarm)
- Per-attack-type SHAP heatmaps (Stage 2 multiclass)

## Installation

```bash
pip install -e .
```

With optional dependencies:
```bash
pip install -e ".[xgboost,lightgbm,shap,plots,torch]"
```

## Testing

```bash
pytest
```
