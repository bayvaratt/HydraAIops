# HYDRA Experiment Plan — TON_IoT Matrix

## Purpose

Produce a frozen, defensible evaluation of binary intrusion-detection models on the
TON_IoT dataset under multiple split strategies, feature-selection regimes, and seeds.
Every run must write a reproducibility metadata block so results can be traced back to
a specific commit and dataset state.

---

## Dataset

| Field | Value |
|-------|-------|
| Name | `ton_iot` |
| Config | `hydra/config/datasets.yaml` |
| Label column | `label` (binary: 0 = normal, 1 = attack) |
| Group column | `src_ip` |
| Timestamp column | `timestamp` |
| Type column | `type` (attack category) |
| Normal type value | `normal` |
| Primary feature regime | `behaviour_only` |

CIC-IoT2023 is reserved for later replication once the TON_IoT matrix is stable.

---

## Experiment Matrix

### Split strategies

| Name (CLI) | Description | Extra args |
|------------|-------------|------------|
| `host` | GroupShuffleSplit on `src_ip`; deployment-realistic | `--group_col src_ip` |
| `temporal` | Chronological split on `timestamp` | `--timestamp_col timestamp` |
| `group_type_stratified` | Group split ensuring all attack types in train+test | `--group_col src_ip --type_col type --normal_type_value normal` |

### Models

| CLI name | Description |
|----------|-------------|
| `baseline_majority` | Constant-score majority baseline |
| `baseline_threshold` | Heuristic threshold on flow features |
| `logreg` | Logistic regression (SAGA solver) |
| `random_forest` | Random forest (300 trees) |
| `xgboost` | XGBoost gradient boosting (optional; skipped if not installed) |

### Feature selection

| Setting | CLI flags | Applied to |
|---------|-----------|------------|
| None | `--feature_selection none` | All models |
| Mutual-info k=20 | `--feature_selection mutual_info --feature_selection_k 20` | logreg, random_forest, xgboost |
| Mutual-info k=40 | `--feature_selection mutual_info --feature_selection_k 40` | logreg, random_forest, xgboost |

Baselines are always run with `--feature_selection none` (they do not use preprocessed
features and are unaffected by feature selection).

### Seeds

`21`, `42`, `84`

### Runs per seed × split

1. Baselines with `feature_selection=none`
2. ML models with `feature_selection=none`
3. ML models with `feature_selection=mutual_info, k=20`
4. ML models with `feature_selection=mutual_info, k=40`

**Total: 3 splits × 3 seeds × 4 run-configs = 36 runs**

---

## Metrics Reported

| Metric | Column in CSV | Primary? |
|--------|---------------|----------|
| PR-AUC | `pr_auc` | **Yes** |
| PR-Lift (pr_auc − prevalence) | `pr_lift` | |
| ROC-AUC | `roc_auc` | |
| FPR @ Recall=0.90 | `fpr_at_recall_0_90` | |
| Recall @ threshold (0.90 target) | `recall_at_0_90` | |
| F1 @ threshold (0.90 recall target) | `f1_at_0_90` | |
| Coverage (fraction predicted positive) | `coverage` | |

---

## Reproducibility Requirements

Every run **must** write `evaluation_meta.json` containing a `"reproducibility"` block
with:

- `git_commit_hash` — SHA from `git rev-parse HEAD`
- `git_dirty` — bool; warns if working tree is unclean
- `config_snapshot` — resolved `DatasetConfig` dict
- `dataset_fingerprint` — SHA-256 of `path:size:mtime`
- `n_rows_loaded` — row count before subsampling
- `n_rows_used` — row count after subsampling
- `split_params` — strategy, group_col, timestamp_col, feature_selection, k, type_unknown_threshold
- `seed`

**Rule: no run is accepted if `evaluation_meta.json` is missing or lacks the
`"reproducibility"` key.**

---

## Commands

### Run the full matrix
```bash
bash scripts/run_ton_matrix.sh
# With row cap for smoke-testing:
bash scripts/run_ton_matrix.sh --max_rows 5000
```

### Aggregate all runs into one table
```bash
python -m hydra.analysis.aggregate_runs \
    --runs_dir runs/ton_iot \
    --out_dir  runs/ton_iot/aggregated
```

### Generate report figures
```bash
python -m hydra.analysis.make_report_plots \
    --csv     runs/ton_iot/aggregated/results_summary.csv \
    --out_dir runs/ton_iot/aggregated/report_figures
```

### Or use Make
```bash
make ton-matrix
make aggregate RUNS_DIR=runs/ton_iot
make plots     RUNS_DIR=runs/ton_iot
```

### All-in-one
```bash
bash scripts/run_all_ton.sh
```

---

## Deviation Policy

Any deviation from this matrix (different seeds, splits, models, or feature-selection
settings) must be documented in a new dated section below. Ad-hoc runs should live in
`runs/<dataset>/adhoc_<desc>/` to avoid polluting the matrix results.

---

## Deviation Log

*(empty — no approved deviations)*
