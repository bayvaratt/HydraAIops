from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_run(run_dir: Path):
    rc_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics_summary.csv"
    if not rc_path.exists() or not metrics_path.exists():
        return None
    with open(rc_path, "r", encoding="utf-8") as f:
        rc = json.load(f)
    metrics = pd.read_csv(metrics_path)
    records = []
    for _, row in metrics.iterrows():
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
                "roc_auc": row["roc_auc"],
                "fpr_at_recall_0_90": row["fpr_at_recall_0_90"],
                "threshold": row["threshold"],
                "coverage": row["coverage"],
                "seed": rc["seed"],
                "commit_hash": rc["commit_hash"],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Consolidate HYDRA run results")
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for run in runs_dir.iterdir():
        if not run.is_dir():
            continue
        recs = load_run(run)
        if recs:
            all_records.extend(recs)

    if not all_records:
        raise SystemExit("No valid runs found")

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "consolidated_metrics.csv", index=False)

    # Summary
    summary_lines = []
    summary_lines.append("# HYDRA Consolidated Summary\n")

    group_cols = ["split_strategy", "group_col", "feature_regime"]
    grouped = df.groupby(group_cols, dropna=False)

    summary_lines.append("## Best By PR-AUC")
    for keys, g in grouped:
        best = g.sort_values("pr_auc", ascending=False).iloc[0]
        summary_lines.append(
            f"- {keys}: best_model={best['model']} pr_auc={best['pr_auc']:.4f}"
        )

    summary_lines.append("\n## Best By Lowest FPR@Recall=0.90")
    for keys, g in grouped:
        best = g.sort_values("fpr_at_recall_0_90", ascending=True).iloc[0]
        summary_lines.append(
            f"- {keys}: best_model={best['model']} fpr_at_recall_0_90={best['fpr_at_recall_0_90']:.4f}"
        )

    summary_lines.append("\n## Notes")
    summary_lines.append("- identifier_inclusive is an upper bound and not deployment-realistic.")

    if (df["split_strategy"] == "stratified").any():
        summary_lines.append(
            "- Stratified splits can inflate results by mixing identities across train/val/test; treat as naive baseline only."
        )

    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
