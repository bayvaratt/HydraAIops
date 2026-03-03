"""Aggregate HYDRA experiment runs into a single results table.

Recursively scans a root directory for valid run folders (each must contain
metrics_summary.csv + run_config.json) and writes:

  results_summary.csv  — one row per (run, model), all metrics + metadata
  results_summary.md   — sorted markdown table + "Top configs" section
  missing_runs.txt     — run dirs that were skipped due to missing files

Usage:
    python -m hydra.analysis.aggregate_runs \\
        --runs_dir runs/ton_iot \\
        --out_dir  runs/ton_iot/aggregated
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Columns written to results_summary.csv (order matters for readability)
OUTPUT_COLS = [
    "dataset",
    "feature_regime",
    "split_strategy",
    "model",
    "feature_selection",
    "feature_selection_k",
    "seed",
    "pr_auc",
    "pr_lift",
    "roc_auc",
    "fpr_at_recall_0_90",
    "recall_at_0_90",
    "f1_at_0_90",
    "coverage",
    "n_rows_loaded",
    "n_rows_used",
    "commit_hash",
    "dirty",
    "dataset_fingerprint",
    "run_dir",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _fmt(v) -> str:
    """Format a cell value for markdown tables."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a plain GitHub-flavoured markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def find_run_dirs(root: Path) -> list[Path]:
    """Return sorted list of dirs that contain metrics_summary.csv."""
    return sorted(p.parent for p in root.rglob("metrics_summary.csv"))


def load_run(run_dir: Path, missing_log: list[str]) -> list[dict]:
    """Load one run directory; return a list of per-model row dicts."""
    metrics_path = run_dir / "metrics_summary.csv"
    config_path = run_dir / "run_config.json"
    meta_path = run_dir / "evaluation_meta.json"

    if not metrics_path.exists():
        missing_log.append(f"{run_dir}: missing metrics_summary.csv")
        return []
    if not config_path.exists():
        missing_log.append(f"{run_dir}: missing run_config.json")
        return []

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Prefer evaluation_meta reproducibility block (new runs); fall back to
    # run_config fields for older runs that predate the block.
    repro: dict = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        repro = meta.get("reproducibility", {})
    else:
        missing_log.append(
            f"{run_dir}: missing evaluation_meta.json (reproducibility unavailable)"
        )

    split_params: dict = repro.get("split_params", {})

    # Dataset name lives inside a nested dict in run_config
    ds_cfg = config.get("dataset", {})
    dataset_name = ds_cfg.get("name", "") if isinstance(ds_cfg, dict) else str(ds_cfg)

    # Prefer reproducibility block; fall back to run_config top-level fields
    feature_selection = (
        split_params.get("feature_selection")
        or config.get("feature_selection", "none")
        or "none"
    )
    feature_selection_k = _safe_int(
        split_params.get("feature_selection_k")
        or config.get("feature_selection_k")
    )
    seed = repro.get("seed") or config.get("seed")
    commit_hash = repro.get("git_commit_hash") or config.get("commit_hash", "")

    rows: list[dict] = []
    with open(metrics_path, encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature_regime": config.get("feature_regime", ""),
                    "split_strategy": (
                        split_params.get("strategy")
                        or config.get("split_strategy", "")
                    ),
                    "model": raw.get("model", ""),
                    "feature_selection": feature_selection,
                    "feature_selection_k": feature_selection_k,
                    "seed": seed,
                    "pr_auc": _safe_float(raw.get("pr_auc")),
                    "pr_lift": _safe_float(raw.get("pr_lift")),
                    "roc_auc": _safe_float(raw.get("roc_auc")),
                    "fpr_at_recall_0_90": _safe_float(raw.get("fpr_at_recall_0_90")),
                    "recall_at_0_90": _safe_float(raw.get("recall_test_at_recall_0_90")),
                    "f1_at_0_90": _safe_float(raw.get("f1_test_at_recall_0_90")),
                    "coverage": _safe_float(raw.get("coverage")),
                    "n_rows_loaded": repro.get("n_rows_loaded"),
                    "n_rows_used": repro.get("n_rows_used"),
                    "commit_hash": commit_hash,
                    "dirty": repro.get("git_dirty", False),
                    "dataset_fingerprint": repro.get("dataset_fingerprint", ""),
                    "run_dir": str(run_dir),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(df: pd.DataFrame, path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_runs = df["run_dir"].nunique()
    n_models = df["model"].nunique()

    lines: list[str] = [
        "# HYDRA Experiment Results Summary",
        "",
        f"Generated: {now}  ",
        f"Runs aggregated: {n_runs} | Unique models: {n_models}",
        "",
    ]

    # -- Best PR-AUC per split × model (mean across seeds, fs=none) ----------
    base = df[df["feature_selection"].fillna("none") == "none"].copy()
    if not base.empty:
        best = (
            base.groupby(["split_strategy", "model"], as_index=False)
            .agg(
                pr_auc_mean=("pr_auc", "mean"),
                pr_auc_std=("pr_auc", "std"),
                roc_auc_mean=("roc_auc", "mean"),
                fpr_mean=("fpr_at_recall_0_90", "mean"),
                n_seeds=("seed", "nunique"),
            )
            .sort_values("pr_auc_mean", ascending=False)
            .round(4)
        )
        lines += [
            "## Best PR-AUC per Split × Model",
            "_Averaged across seeds; feature_selection=none only_",
            "",
            _md_table(best),
            "",
        ]

    # -- Feature selection comparison (ML models only) -----------------------
    ml_df = df[~df["model"].isin(["baseline_majority", "baseline_threshold"])].copy()
    if not ml_df.empty and ml_df["feature_selection"].nunique() > 1:
        def _fs_label(row) -> str:
            fs = row.get("feature_selection", "none") or "none"
            if fs == "none":
                return "none"
            k = row.get("feature_selection_k")
            return f"MI k={int(k)}" if k is not None else fs

        ml_df = ml_df.copy()
        ml_df["fs_label"] = ml_df.apply(_fs_label, axis=1)
        fs_comp = (
            ml_df.groupby(["split_strategy", "model", "fs_label"], as_index=False)
            .agg(pr_auc_mean=("pr_auc", "mean"), n_seeds=("seed", "nunique"))
            .sort_values(["split_strategy", "model", "pr_auc_mean"], ascending=[True, True, False])
            .round(4)
        )
        lines += [
            "## Feature Selection Comparison (ML models)",
            "",
            _md_table(fs_comp),
            "",
        ]

    # -- Top 15 configs by PR-AUC -------------------------------------------
    display_cols = [
        "split_strategy", "model", "feature_selection", "feature_selection_k",
        "seed", "pr_auc", "roc_auc", "fpr_at_recall_0_90", "recall_at_0_90",
    ]
    available = [c for c in display_cols if c in df.columns]
    top = df[available].nlargest(15, "pr_auc").round(4)
    lines += [
        "## Top 15 Configurations (by PR-AUC)",
        "",
        _md_table(top),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate HYDRA experiment runs into results_summary.csv/md"
    )
    parser.add_argument(
        "--runs_dir",
        required=True,
        help="Root directory to scan recursively for run folders",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for results_summary.csv, .md, and missing_runs.txt",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)

    if not runs_dir.exists():
        log.error("runs_dir does not exist: %s", runs_dir)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(runs_dir)
    log.info("Found %d candidate run dirs under %s", len(run_dirs), runs_dir)

    all_rows: list[dict] = []
    missing_log: list[str] = []

    for rd in run_dirs:
        rows = load_run(rd, missing_log)
        all_rows.extend(rows)

    if not all_rows:
        log.error("No rows loaded. Check that run dirs contain metrics_summary.csv.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Ensure all expected columns exist (fill with None if absent)
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None

    df = df[OUTPUT_COLS].sort_values("pr_auc", ascending=False, na_position="last")

    # Write outputs
    csv_path = out_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    log.info("CSV: %s  (%d rows)", csv_path, len(df))

    write_markdown(df, out_dir / "results_summary.md")

    if missing_log:
        missing_path = out_dir / "missing_runs.txt"
        missing_path.write_text("\n".join(missing_log) + "\n", encoding="utf-8")
        log.warning(
            "%d issue(s) logged in %s", len(missing_log), missing_path
        )

    log.info("Done. Output: %s", out_dir)


if __name__ == "__main__":
    main()
