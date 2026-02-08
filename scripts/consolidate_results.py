from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_run_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _collect_metrics(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in runs_dir.rglob("metrics_summary.csv"):
        run_dir = metrics_path.parent
        run_config_path = run_dir / "run_config.json"
        if not run_config_path.exists():
            logger.warning("Skipping %s (missing run_config.json)", run_dir)
            continue
        run_cfg = _load_run_config(run_config_path)
        df = pd.read_csv(metrics_path)
        # Assume legacy runs without split_strategy used stratified splits.
        split_strategy = run_cfg.get("split_strategy")
        if split_strategy is None:
            logger.warning("Inferring split_strategy=stratified for %s (missing in run_config)", run_dir)
            split_strategy = "stratified"

        for col in [
            "dataset",
            "feature_regime",
            "group_col",
            "timestamp_col",
            "seed",
            "commit",
            "label_permutation_probe",
        ]:
            df[col] = run_cfg.get(col)
        df["split_strategy"] = split_strategy
        df["run_dir"] = str(run_dir)
        df["run_id"] = run_dir.name
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No metrics_summary.csv found under {runs_dir}")
    return pd.concat(rows, ignore_index=True)


def _best_by_group(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["split_strategy", "group_col", "timestamp_col", "feature_regime"]
    sort_cols = ["pr_auc", "fpr_at_recall"]
    df_sorted = df.sort_values(sort_cols, ascending=[False, True])
    return df_sorted.groupby(group_cols, as_index=False).first()


def _markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    display["group_col"] = display["group_col"].fillna("-")
    display["timestamp_col"] = display["timestamp_col"].fillna("-")
    display = display[
        [
            "split_strategy",
            "group_col",
            "timestamp_col",
            "feature_regime",
            "model",
            "pr_auc",
            "fpr_at_recall",
        ]
    ]
    display = display.rename(
        columns={
            "model": "best_model",
            "pr_auc": "best_pr_auc",
            "fpr_at_recall": "best_fpr@0.90",
        }
    )
    return display.to_markdown(index=False)


def _summary_notes(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    strategies = set(df["split_strategy"].dropna().unique())
    if "stratified" in strategies and "host" in strategies:
        strat = df[df["split_strategy"] == "stratified"]["pr_auc"].median()
        host = df[df["split_strategy"] == "host"]["pr_auc"].median()
        notes.append(
            f"Stratified vs host: median PR-AUC stratified={strat:.4f}, host={host:.4f}."
        )
    if "temporal" not in strategies:
        notes.append("Temporal split not present in consolidated runs (timestamp missing or skipped).")
    notes.append("Identifier-inclusive results are an upper bound and not deployable.")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs/ton")
    ap.add_argument("--out_dir", default="runs/ton/consolidated")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_metrics(runs_dir)
    consolidated_csv = out_dir / "consolidated_metrics.csv"
    df.to_csv(consolidated_csv, index=False)

    best = _best_by_group(df)
    summary_md = out_dir / "summary.md"
    table_md = _markdown_table(best)
    notes = _summary_notes(df)

    with open(summary_md, "w") as f:
        f.write("# HYDRA Ton IoT Consolidated Results\n\n")
        f.write("## Best Model Per Split/Regime\n\n")
        f.write(table_md)
        f.write("\n\n## Notes\n\n")
        for note in notes:
            f.write(f"- {note}\n")

    logger.info("Wrote %s", consolidated_csv)
    logger.info("Wrote %s", summary_md)


if __name__ == "__main__":
    main()
