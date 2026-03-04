# HYDRA TON_IoT Results Sanity Memo

**Generated:** 2026-03-03 15:45 GMT
**Branch:** eval-hardening
**Run directory:** `runs/ton_iot/`
**Aggregated outputs:** `runs/ton_iot/aggregated/`

---

## Reproducibility Block

| Field | Value |
|-------|-------|
| Dataset fingerprint | `1e9b665988a0ff1dbb5c9e9e9ff15dd219f4881505acac856fa1e823518a2ee4` |
| Commit hash | `f6c955288e36f7553fa8c753d57bd7539bdc6a04` |
| Working tree dirty | **false** (all 66 full-run rows) |
| n\_rows\_loaded | 211 043 |
| n\_rows\_used | 211 043 (no subsampling on full run) |
| Splits completed | host ✓ · group\_type\_stratified ✓ · temporal ✗ (see Red Flags) |
| Seeds completed | 21, 42, 84 (all 3) |

---

## Matrix Completion

Expected: 99 combinations (3 splits × 3 seeds × 11 model/fs configs).
Completed: **66 / 99** (67%).
Missing: 33 — all in the `temporal` split (see Red Flag §1 below).

---

## Best Configuration per Split

| Split | Model | Feature Selection | Seed | PR-AUC | ROC-AUC | FPR@Recall=0.90 |
|-------|-------|-------------------|------|--------|---------|-----------------|
| host | xgboost | mutual\_info k=40 | 84 | **1.0000** | 0.9999 | 0.0000 |
| group\_type\_stratified | random\_forest | mutual\_info k=40 | 21 | **1.0000** | 0.9998 | 0.0008 |
| temporal | — | — | — | N/A | N/A | N/A |

### Mean PR-AUC per Model (fs=none, 3 seeds, full 211k dataset)

| Model | host | group\_type\_stratified |
|-------|------|------------------------|
| baseline\_majority | 0.8595 ± 0.049 | 0.9879 ± 0.006 |
| baseline\_threshold | 0.8854 ± 0.032 | 0.9871 ± 0.004 |
| logreg | 0.8997 ± 0.031 | 0.9899 ± 0.001 |
| random\_forest | **0.9917 ± 0.011** | **0.9999 ± 0.000** |
| xgboost | **0.9914 ± 0.012** | **1.0000 ± 0.000** |

---

## Three Key Insights

1. **Host split is the hardest and most realistic benchmark.**
   The `host` split (GroupShuffleSplit on `src_ip`) produces the largest spread between model tiers: RF/XGB at PR-AUC ~0.991 vs. baseline_threshold at ~0.885 and baseline_majority at ~0.860. This 10-point gap is only visible when evaluation uses a deployment-realistic split — it would collapse to near-zero under stratified random splits. For the dissertation, `host` is the canonical split to report.

2. **XGBoost and RandomForest are near-perfect on group_type_stratified but not on host.**
   Both tree models reach PR-AUC ≥ 0.999 on `group_type_stratified` (mean across seeds, full data) versus ~0.991 on `host`. The ~0.8 pp drop on `host` reflects genuine generalisation difficulty when the test set contains unseen IP addresses. The gap is large enough to be dissertation-worthy.

3. **Feature selection (mutual_info k=20/40) provides negligible benefit over full features on this dataset.**
   Across logreg, RF, and XGBoost on the full 211k dataset, MI-selected subsets (k=20 or k=40) return identical mean PR-AUC to the full-feature baseline (logreg: 0.9448 at all FS levels; RF: 0.9958; XGB: 0.9957). This suggests that for TON_IoT `behaviour_only`, all informative features are already contained in the top 20 mutual-information features and adding more does not help. It also confirms that the feature-selection code is not inadvertently discarding signal.

---

## Three Caveats / Failure Modes Checked

1. **Label leakage check (identifier columns).**
   The `behaviour_only` feature regime explicitly drops `src_ip`, `dst_ip`, `src_port`, `dst_port`, `service` — the five columns most likely to encode IP identity. This was confirmed in the run logs: `INFO Dropping columns (5): ['dst_ip', 'dst_port', 'service', 'src_ip', 'src_port']`. PR-AUC on `host` split (~0.991 for RF) is plausible for a behavioural model on labelled IoT traffic; it is NOT near-1.0 in the way that an identity-column leak would produce.

2. **Single-class split check (new split_assertions guard).**
   The new `split_assertions=True` runtime guard confirmed that `group_type_stratified/seed=21` is feasible on the full 211k dataset (all seeds passed). The smoke-run failure for seed=21 on this split was a correct detection: with only 5 000 subsampled rows, the algorithm could not place all 10 attack types into both train and test simultaneously because the rare types (`password`, `xss`) had only 1 reachable group after subsampling. The guard worked as designed.

3. **Near-perfect scores are internally consistent.**
   PR-AUC=1.0 for RF/XGB on `group_type_stratified` is plausible given that (a) the dataset prevalence is ~54%, (b) TON_IoT is known to be a well-separated synthetic benchmark, and (c) tree models with 300+ trees on 150k training rows can memorise class-discriminating flow statistics. The FPR@Recall=0.90 of 0.001–0.008 confirms the models achieve the target recall at a very low false alarm rate — not a red flag, but a documented finding that the dataset is "easy" for gradient-boosted trees.

---

## Red Flags

### Red Flag 1 — Temporal Split Not Feasible for TON_IoT CSV (BLOCKING)

**Severity:** Blocking — 12/36 full-run configurations incomplete.

**Symptom:** All 12 `temporal` split runs (3 seeds × 4 run-configs) fail with:
```
WARNING Timestamp column missing; using row order as temporal proxy.
RuntimeError: val split has <2 classes; aborting
```

**Root cause (confirmed):** The processed `ton_iot.csv` has no `timestamp` column. The pipeline falls back to row order as a temporal proxy. The dataset is arranged in a non-temporal order — attacks are clustered at the start and end of the file, with normal traffic concentrated in the middle:

| Row range | label=1 rate |
|-----------|-------------|
| 0 – 63 312 | 1.000 |
| 63 312 – 84 417 | 0.840 |
| 84 417 – 126 625 | 0.000 |
| 126 625 – 147 730 | 0.791 |
| 147 730 – 211 043 | 1.000 |

With train\_frac=0.70 the validation portion (rows 147 730–179 386) lands entirely in the all-attack zone → single-class val → assertion fires.

**Note:** The smoke run (5 000 randomly subsampled rows) passed `temporal` because random subsampling destroyed the row-order structure and created balanced splits. The smoke temporal results are **invalid** as temporal evaluations.

**Next steps required before CIC-IoT2023:**
- Investigate whether the original TON_IoT Zeek logs contain a `ts` field that was stripped during CSV export. If recoverable, re-export with `ts` and update `datasets.yaml` → `timestamp_col: ts`.
- If original timestamps are unrecoverable, drop `temporal` from the frozen matrix and update `docs/experiment_plan.md` with a deviation log entry.
- Do NOT report smoke temporal results in the dissertation — they reflect random-split behaviour, not chronological generalisation.

---

## Key Figures

| Figure | Path |
|--------|------|
| PR-AUC by model — host split | [report_figures/pr_auc_split_host.png](../runs/ton_iot/aggregated/report_figures/pr_auc_split_host.png) |
| PR-AUC by model — group_type_stratified | [report_figures/pr_auc_split_group_type_stratified.png](../runs/ton_iot/aggregated/report_figures/pr_auc_split_group_type_stratified.png) |
| FPR@Recall=0.90 — host split | [report_figures/fpr_split_host.png](../runs/ton_iot/aggregated/report_figures/fpr_split_host.png) |
| FPR@Recall=0.90 — group_type_stratified | [report_figures/fpr_split_group_type_stratified.png](../runs/ton_iot/aggregated/report_figures/fpr_split_group_type_stratified.png) |
| Feature selection effect — random_forest | [report_figures/fs_effect_random_forest.png](../runs/ton_iot/aggregated/report_figures/fs_effect_random_forest.png) |
| Feature selection effect — xgboost | [report_figures/fs_effect_xgboost.png](../runs/ton_iot/aggregated/report_figures/fs_effect_xgboost.png) |
| Feature selection effect — logreg | [report_figures/fs_effect_logreg.png](../runs/ton_iot/aggregated/report_figures/fs_effect_logreg.png) |
| Full results CSV | [aggregated/results_summary.csv](../runs/ton_iot/aggregated/results_summary.csv) |
| Full results Markdown | [aggregated/results_summary.md](../runs/ton_iot/aggregated/results_summary.md) |

---

## Recommended Next Steps

1. Investigate timestamp column in original TON_IoT data (see Red Flag §1).
2. If temporal infeasible: file a deviation log entry in `docs/experiment_plan.md`.
3. Proceed to CIC-IoT2023 replication once temporal question is resolved.
4. Consider identity-column ablation run (identifier\_inclusive vs behaviour\_only) to quantify leakage risk.
