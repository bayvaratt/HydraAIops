# HYDRA: Explainability-Driven Intrusion Detection

This repo provides a clean, reproducible experimental pipeline for binary intrusion detection on tabular network-flow data. Explanations are first-class outputs (global + local). The pipeline emphasizes leakage-safe splits, honest baselines, and deterministic runs.

## Quickstart

Single run:
```bash
python -m hydra.pipelines.run_tabular --dataset ton_iot --feature_regime behaviour_only --split_strategy host --group_col src_ip --seed 42
```

Permutation leakage probe:
```bash
python -m hydra.pipelines.run_tabular --dataset ton_iot --feature_regime behaviour_only --split_strategy host --group_col src_ip --seed 42 --label_permutation_probe
```

Matrix runner:
```bash
python -m hydra.pipelines.run_experiments --dataset ton_iot --max_rows 5000 --seed 42 --label_permutation_probe
```

Consolidate results:
```bash
python scripts/consolidate_results.py --runs_dir runs/ton_iot --out_dir runs/ton_iot/consolidated
```

## Dataset configuration
Edit `hydra/config/datasets.yaml` to point to your local dataset paths and columns. The loader supports `.csv` and `.parquet`.

Required per dataset:
- `path`: local file path
- `label_col`: binary label column
- `positive_label`: value mapped to 1 (e.g., 1 or "attack")

Optional:
- `timestamp_col`
- `group_col` (default for host split)
- `duration_col`, `src_bytes_col`, `dst_bytes_col`, `src_pkts_col`, `dst_pkts_col` (for baseline_threshold)
- `categorical_cols` and `numeric_cols` (overrides inference)

## Notes
- Splits:
  - `host`: GroupShuffleSplit with strict disjointness checks.
  - `temporal`: chronological split (train/val/test).
  - `stratified`: allowed only as a naive baseline (labeled not deployment-realistic).
- LightGBM is optional; if unavailable, a fallback model is used with identical interface.
- Explainability:
  - Global importance via built-in or permutation importance.
  - Local explanations via SHAP if available; otherwise a deterministic occlusion fallback.

## Installation
Base dependencies:
```bash
pip install -e .
```
Optional:
```bash
pip install -e .[lightgbm,shap,plots]
```
