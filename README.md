# HydraAIops

This repository now includes a reproducible, test-driven pipeline for **HYDRA: Explainability-Driven Intrusion Detection** with tabular IDS baselines (LR/RF/GBT) and a graph-based GNN edge classifier.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Dataset Paths (Single Source of Truth)

Default paths live in `hydra/config.py`:

- `data/ton_iot/raw/train_test_network.csv`
- `data/cic_iot2023/raw/cic_iot2023.csv`

Override paths in CLI with `--data_path`.

## Smoke Tests

Smoke tests validate schema, label values, missingness, and high-cardinality fields.

```bash
python scripts/smoke_test.py --dataset ton_iot --max_rows 5000
python scripts/smoke_test.py --dataset cic_iot2023 --max_rows 5000
```

Artifacts (per run):

- `missingness.json`
- `high_cardinality_columns.json`
- `dataset_summary.json`
- `smoke_summary.json`

## Run Experiments

### Tabular IDS (LR/RF/GBT)

```bash
python -m hydra.pipelines.run_tabular --dataset ton_iot --feature_regime behaviour_only
python -m hydra.pipelines.run_tabular --dataset ton_iot --feature_regime identifier_inclusive
```

### Honest Evaluation (Host Split + Baselines + Probe)

Host split by `src_ip`:

```bash
python -m hydra.pipelines.run_tabular --dataset ton_iot --feature_regime behaviour_only --split_strategy host --group_col src_ip
```

Host split by `dst_ip`:

```bash
python -m hydra.pipelines.run_tabular --dataset ton_iot --feature_regime behaviour_only --split_strategy host --group_col dst_ip
```

Label permutation probe:

```bash
python -m hydra.pipelines.run_tabular --dataset ton_iot --feature_regime behaviour_only --split_strategy host --group_col src_ip --label_permutation_probe
```

Matrix run + consolidated summary:

```bash
python scripts/run_experiments.py --max_rows 5000 --seed 42 --label_permutation_probe
python scripts/consolidate_results.py --runs_dir runs/ton --out_dir runs/ton/consolidated
```

### Graph IDS (GNN Edge Classification)

```bash
python -m hydra.pipelines.run_gnn --dataset cic_iot2023 --feature_regime behaviour_only
```

### Run All

```bash
python -m hydra.pipelines.run_all
# or
python scripts/run_experiments.py --max_rows 5000
```

## Tests

```bash
pytest -q
```

## Artifacts

Runs are written to `runs/<dataset>/<timestamp>/` (or `<dataset>_gnn`). Key outputs:

- `run_config.json` (full config snapshot)
- `missingness.json`, `high_cardinality_columns.json`, `dataset_summary.json`
- `metrics_summary.csv` (tabular)
- `metrics.json` (per-model or GNN)
- `explanations.jsonl`
- `permutation_importance.json` (tabular)

Metrics include PR-AUC and FPR@fixed recall (threshold calibrated on validation only).

## Legacy Heads

Original heads remain available:

```bash
python -m hydra_pipeline ton --csv data/raw/ton_iot_network.csv --out runs/ton
python -m hydra_pipeline hdfs --hdfs_dir data/raw/HDFS_1 --out runs/hdfs
python -m hydra_pipeline lanl --auth_gz data/raw/auth.txt.gz --redteam_gz data/raw/redteam.txt.gz --out runs/lanl
```

## Self-check

```bash
python -m py_compile $(find . -name "*.py")
```
