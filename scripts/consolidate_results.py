from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script without installing the package.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from hydra.xai.importance_similarity import compute_rank_similarity, write_similarity_json


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.{nd}f}"


def fmt_signed(x, nd=4):
    rendered = fmt(x, nd)
    if rendered == "nan":
        return rendered
    return f"+{rendered}" if float(x) >= 0 else rendered


def best_by_metric(group: pd.DataFrame, metric: str):
    if metric not in group.columns:
        return None
    series = pd.to_numeric(group[metric], errors="coerce")
    if series.notna().sum() == 0:
        return None
    return group.loc[series.idxmax()]


def load_run(run_dir: Path):
    rc_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics_summary.csv"
    meta_path = run_dir / "evaluation_meta.json"
    if not rc_path.exists() or not metrics_path.exists():
        return None
    with open(rc_path, "r", encoding="utf-8") as f:
        rc = json.load(f)
    meta = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    metrics = pd.read_csv(metrics_path)
    records = []
    for _, row in metrics.iterrows():
        split_counts = meta.get("split_label_counts") if meta else {}
        train_counts = split_counts.get("train", {}) if split_counts else {}
        val_counts = split_counts.get("val", {}) if split_counts else {}
        test_counts = split_counts.get("test", {}) if split_counts else {}
        dup_audit = meta.get("duplicate_leakage_audit") if meta else {}
        prevalence_test = test_counts.get("prevalence")
        pr_lift = row.get("pr_lift")
        if pd.isna(pr_lift):
            if prevalence_test is not None:
                pr_lift = row["pr_auc"] - prevalence_test
        records.append(
            {
                "dataset": rc["dataset"]["name"],
                "run_id": rc["run_id"],
                "timestamp": rc["timestamp"],
                "split_strategy": rc["split_strategy"],
                "group_col": rc.get("group_col"),
                "timestamp_col": rc.get("timestamp_col"),
                "feature_regime": rc["feature_regime"],
                "model": row["model"],
                "pr_auc": row["pr_auc"],
                "pr_lift": pr_lift,
                "roc_auc": row["roc_auc"],
                "fpr_at_recall_0_90": row["fpr_at_recall_0_90"],
                "precision_at_recall_0_90": row.get("precision_test_at_recall_0_90", row.get("precision_at_recall_0_90")),
                "recall_at_threshold_test": row.get("recall_test_at_recall_0_90", row.get("recall_test_at_thr")),
                "threshold_from_val": row.get("threshold_at_recall_0_90", row.get("threshold_from_val", row.get("threshold"))),
                "recall_target_met": row.get("recall_target_met"),
                "precision_val_at_recall_0_90": row.get("precision_val_at_recall_0_90"),
                "recall_val_at_recall_0_90": row.get("recall_val_at_recall_0_90"),
                "coverage": row["coverage"],
                "n_train": train_counts.get("n"),
                "n_val": val_counts.get("n"),
                "n_test": test_counts.get("n"),
                "pos_train": train_counts.get("n_pos"),
                "neg_train": train_counts.get("n_neg"),
                "prevalence_train": train_counts.get("prevalence"),
                "pos_val": val_counts.get("n_pos"),
                "neg_val": val_counts.get("n_neg"),
                "prevalence_val": val_counts.get("prevalence"),
                "pos_test": test_counts.get("n_pos"),
                "neg_test": test_counts.get("n_neg"),
                "prevalence_test": prevalence_test,
                "train_test_overlap_rate": (
                    dup_audit.get("train_test_overlap_rate_by_test", dup_audit.get("train_test_overlap_rate"))
                    if dup_audit
                    else None
                ),
                "train_test_overlap_rate_by_test": dup_audit.get("train_test_overlap_rate_by_test") if dup_audit else None,
                "train_test_overlap_rate_by_train": dup_audit.get("train_test_overlap_rate_by_train") if dup_audit else None,
                "train_val_overlap_rate_by_val": dup_audit.get("train_val_overlap_rate_by_val") if dup_audit else None,
                "val_test_overlap_rate_by_test": dup_audit.get("val_test_overlap_rate_by_test") if dup_audit else None,
                "duplicate_leakage_flag": dup_audit.get("duplicate_leakage_flag") if dup_audit else None,
                "seed": rc["seed"],
                "commit_hash": rc["commit_hash"],
            }
        )
    return records


def load_run_config(run_dir: Path):
    rc_path = run_dir / "run_config.json"
    if not rc_path.exists():
        return None
    with open(rc_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Consolidate HYDRA run results")
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    run_infos = []
    for run in runs_dir.iterdir():
        if not run.is_dir():
            continue
        recs = load_run(run)
        if recs:
            all_records.extend(recs)
        rc = load_run_config(run)
        if rc:
            run_infos.append(
                {
                    "run_dir": run,
                    "dataset": rc["dataset"]["name"],
                    "feature_regime": rc["feature_regime"],
                    "split_strategy": rc["split_strategy"],
                    "group_col": rc.get("group_col"),
                    "seed": rc["seed"],
                    "timestamp": rc["timestamp"],
                }
            )

    if not all_records:
        raise SystemExit("No valid runs found")

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "consolidated_metrics.csv", index=False)

    # Explainability stability: compare paper_5feat vs behaviour_only global importance
    latest_by_key = defaultdict(dict)
    for info in run_infos:
        key = (info["dataset"], info["split_strategy"], info["group_col"], info["seed"])
        regime = info["feature_regime"]
        existing = latest_by_key[key].get(regime)
        if existing is None or info["timestamp"] > existing["timestamp"]:
            latest_by_key[key][regime] = info

    similarity_by_key_model = {}
    for key, regimes in latest_by_key.items():
        if "paper_5feat" not in regimes or "behaviour_only" not in regimes:
            continue
        paper = regimes["paper_5feat"]
        behav = regimes["behaviour_only"]
        for model in ["random_forest", "lightgbm"]:
            paper_path = paper["run_dir"] / "explain" / model / "global_importance.csv"
            behav_path = behav["run_dir"] / "explain" / model / "global_importance.csv"
            if not paper_path.exists() or not behav_path.exists():
                continue
            corr, overlap = compute_rank_similarity(paper_path, behav_path, method="spearman")
            for target in [paper, behav]:
                out_path = target["run_dir"] / "explain" / model / "global_importance_similarity.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                write_similarity_json(
                    out_path,
                    corr,
                    compared_regimes=["paper_5feat", "behaviour_only"],
                    method="spearman",
                    overlap_features=overlap,
                )
            similarity_by_key_model[(key, model)] = corr

    # Summary
    summary_lines = []
    summary_lines.append("# HYDRA Consolidated Summary\n")

    group_cols = ["split_strategy", "group_col", "feature_regime"]
    grouped = df.groupby(group_cols, dropna=False)

    summary_lines.append("## Best By PR-Lift")
    for keys, g in grouped:
        best = best_by_metric(g, "pr_lift")
        if best is None:
            summary_lines.append(f"- {keys}: insufficient data for metric pr_lift")
            continue
        summary_lines.append(f"- {keys}: best_model={best['model']} pr_lift={fmt(best['pr_lift'])}")

    summary_lines.append("\n## Best By Precision@Recall=0.90")
    for keys, g in grouped:
        best = best_by_metric(g, "precision_at_recall_0_90")
        if best is None:
            summary_lines.append(f"- {keys}: insufficient data for metric precision_at_recall_0_90")
            continue
        summary_lines.append(
            f"- {keys}: best_model={best['model']} precision@0.90={fmt(best['precision_at_recall_0_90'])}"
        )

    summary_lines.append("\n## Best By PR-AUC (Secondary)")
    for keys, g in grouped:
        best = best_by_metric(g, "pr_auc")
        if best is None:
            summary_lines.append(f"- {keys}: insufficient data for metric pr_auc")
            continue
        summary_lines.append(f"- {keys}: best_model={best['model']} pr_auc={fmt(best['pr_auc'])}")

    summary_lines.append("\n## Notes")
    summary_lines.append("- identifier_inclusive is an upper bound and not deployment-realistic.")

    if (df["split_strategy"] == "stratified").any():
        summary_lines.append(
            "- Stratified splits can inflate results by mixing identities across train/val/test; treat as naive baseline only."
        )

    # Paper replication section
    summary_lines.append("\n## Paper Replication: Minimal 5-Feature Set (Dharini et al., 2026)")

    if "ton_iot" in set(df["dataset"]):
        df_paper = df[df["dataset"] == "ton_iot"].copy()
    else:
        df_paper = df.copy()

    split_defs = [
        ("stratified", {"split_strategy": "stratified", "group_col": None}),
        ("host-src_ip", {"split_strategy": "host", "group_col": "src_ip"}),
        ("host-dst_ip", {"split_strategy": "host", "group_col": "dst_ip"}),
    ]

    delta_cache = {}
    for label, spec in split_defs:
        if spec["split_strategy"] == "stratified":
            df_split = df_paper[df_paper["split_strategy"] == spec["split_strategy"]]
        elif spec["group_col"] is None:
            df_split = df_paper[(df_paper["split_strategy"] == spec["split_strategy"]) & df_paper["group_col"].isna()]
        else:
            df_split = df_paper[
                (df_paper["split_strategy"] == spec["split_strategy"])
                & (df_paper["group_col"] == spec["group_col"])
            ]

        paper = df_split[df_split["feature_regime"] == "paper_5feat"]
        behav = df_split[df_split["feature_regime"] == "behaviour_only"]

        if paper.empty:
            summary_lines.append(f"- {label}: no paper_5feat runs found.")
            continue

        best_pr = best_by_metric(paper, "pr_lift")
        if best_pr is None:
            summary_lines.append(f"- {label}: insufficient data for metric pr_lift")
        else:
            summary_lines.append(
                f"- {label}: paper_5feat best_by_pr_lift={best_pr['model']} pr_lift={fmt(best_pr['pr_lift'])} "
                f"precision@0.90={fmt(best_pr['precision_at_recall_0_90'])} coverage={fmt(best_pr['coverage'])}"
            )
        best_prec = best_by_metric(paper, "precision_at_recall_0_90")
        if best_prec is None:
            summary_lines.append(f"- {label}: insufficient data for metric precision_at_recall_0_90")
        else:
            summary_lines.append(
                f"- {label}: paper_5feat best_by_precision@0.90={best_prec['model']} "
                f"precision@0.90={fmt(best_prec['precision_at_recall_0_90'])} pr_lift={fmt(best_prec['pr_lift'])} "
                f"coverage={fmt(best_prec['coverage'])}"
            )

        if behav.empty:
            summary_lines.append(f"- {label}: behaviour_only runs missing; cannot compare.")
            continue

        # Compare regimes using each regime's best-by-PR-Lift model for a consistent baseline.
        paper_best = best_pr
        if paper_best is None:
            continue
        behav_best = best_by_metric(behav, "pr_lift")
        if behav_best is None:
            summary_lines.append(f"- {label}: insufficient data for metric pr_lift")
            continue
        delta_pr = paper_best["pr_lift"] - behav_best["pr_lift"]
        delta_prec = paper_best["precision_at_recall_0_90"] - behav_best["precision_at_recall_0_90"]
        delta_cov = paper_best["coverage"] - behav_best["coverage"]
        delta_cache[label] = (delta_pr, delta_prec, delta_cov)

        summary_lines.append(
            f"- {label}: paper_5feat vs behaviour_only (best-by-pr_lift) "
            f"ΔPR-Lift={fmt_signed(delta_pr)} ΔPrecision@0.90={fmt_signed(delta_prec)} ΔCoverage={fmt_signed(delta_cov)}"
        )

    summary_lines.append(
        "- Stratified splits can inflate TON-IoT performance due to scenario/host overlap; host/time splits are closer to deployment reality."
    )
    summary_lines.append("- identifier_inclusive is an upper bound / not deployable.")

    # Explainability stability (host-src_ip)
    host_key = None
    for key in similarity_by_key_model:
        (dataset, split_strategy, group_col, seed), _ = key
        if dataset == "ton_iot" and split_strategy == "host" and group_col == "src_ip":
            host_key = (dataset, split_strategy, group_col, seed)
            break
    if host_key:
        for model in ["random_forest", "lightgbm"]:
            corr = similarity_by_key_model.get((host_key, model))
            if corr is None:
                continue
            summary_lines.append(
                f"- Explainability stability (host-src_ip, {model}): spearman={fmt(corr)}"
            )

    # Interpretation bullets (host splits)
    def _interpret(label: str, delta):
        if delta is None:
            return f"- {label}: insufficient runs to compare paper_5feat vs behaviour_only."
        delta_pr, delta_prec, _ = delta
        if delta_pr >= -0.02 and delta_prec >= -0.02:
            return (
                f"- {label}: paper_5feat is competitive on host split "
                f"(ΔPR-Lift={fmt_signed(delta_pr)}, ΔPrecision@0.90={fmt_signed(delta_prec)})."
            )
        return (
            f"- {label}: paper_5feat underperforms on host split "
            f"(ΔPR-Lift={fmt_signed(delta_pr)}, ΔPrecision@0.90={fmt_signed(delta_prec)})."
        )

    summary_lines.append(_interpret("host-src_ip", delta_cache.get("host-src_ip")))
    summary_lines.append(_interpret("host-dst_ip", delta_cache.get("host-dst_ip")))
    if "host-src_ip" in delta_cache and "host-dst_ip" in delta_cache:
        avg_pr = (delta_cache["host-src_ip"][0] + delta_cache["host-dst_ip"][0]) / 2.0
        summary_lines.append(
            f"- host splits overall: average ΔPR-Lift={fmt_signed(avg_pr)} (paper_5feat vs behaviour_only)."
        )
    else:
        summary_lines.append("- host splits overall: insufficient runs to compute average ΔPR-Lift.")

    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
