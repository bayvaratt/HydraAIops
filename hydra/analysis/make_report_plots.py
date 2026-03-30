"""Generate report-ready plots from results_summary.csv.

Produces (matplotlib only, no seaborn):
  - PR-AUC by model per split strategy        (one PNG per split)
  - FPR@Recall=0.90 by model per split        (one PNG per split)
  - Feature-selection effect for rf / xgboost (one PNG per model)

Usage:
    python -m hydra.analysis.make_report_plots \\
        --csv     runs/ton_iot/aggregated/results_summary.csv \\
        --out_dir runs/ton_iot/aggregated/report_figures
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour cycling — matplotlib default tab10, no hardcoded colours
# ---------------------------------------------------------------------------

def _colour_map(keys: list[str]) -> dict[str, str]:
    """Assign a distinct tab10 colour to each key."""
    cmap = plt.get_cmap("tab10")
    return {k: cmap(i % 10) for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# Bar-chart helper
# ---------------------------------------------------------------------------

def _bar_chart(
    ax: plt.Axes,
    groups: list[str],
    series: dict[str, list[float | None]],
    errors: dict[str, list[float | None]] | None,
    ylabel: str,
    title: str,
) -> None:
    """Grouped bar chart where each group is a model and each bar is a series key."""
    n_groups = len(groups)
    n_series = len(series)
    if n_series == 0 or n_groups == 0:
        ax.set_title(f"{title} (no data)")
        return

    width = 0.8 / n_series
    offsets = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n_series)
    x = np.arange(n_groups)
    colours = _colour_map(list(series.keys()))

    for offset, (label, values) in zip(offsets, series.items()):
        vals = [v if v is not None else float("nan") for v in values]
        errs = None
        if errors and label in errors:
            errs = [e if e is not None else 0.0 for e in errors[label]]
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=label,
            color=colours[label],
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 1, "alpha": 0.7},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=max(1, n_series // 4))
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)


# ---------------------------------------------------------------------------
# Plot 1: PR-AUC by model per split (one figure per split)
# ---------------------------------------------------------------------------

def plot_pr_auc_per_split(df: pd.DataFrame, out_dir: Path) -> None:
    """One bar chart per split strategy: PR-AUC mean ± std across seeds, fs=none only."""
    base = df[df["feature_selection"].fillna("none") == "none"].copy()
    if base.empty:
        log.warning("No rows with feature_selection=none — skipping PR-AUC plots")
        return

    splits = sorted(base["split_strategy"].dropna().unique())
    models = sorted(base["model"].dropna().unique())

    for split in splits:
        sub = base[base["split_strategy"] == split]
        agg = (
            sub.groupby("model")["pr_auc"]
            .agg(["mean", "std"])
            .reindex(models)
        )
        means = {m: agg.loc[m, "mean"] if m in agg.index else None for m in models}
        stds  = {m: agg.loc[m, "std"]  if m in agg.index else None for m in models}

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 4))
        _bar_chart(
            ax,
            groups=models,
            series={m: [means[m]] for m in models},
            errors={m: [stds[m]] for m in models},
            ylabel="PR-AUC",
            title=f"PR-AUC by Model — split={split}",
        )
        ax.set_xlabel("Model")
        fig.tight_layout()
        fname = out_dir / f"pr_auc_split_{split}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        log.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Plot 2: FPR@Recall=0.90 by model per split
# ---------------------------------------------------------------------------

def plot_fpr_per_split(df: pd.DataFrame, out_dir: Path) -> None:
    """One bar chart per split: FPR@Recall=0.90 mean ± std, fs=none only."""
    if "fpr_at_recall_0_90" not in df.columns:
        log.warning("fpr_at_recall_0_90 column absent — skipping FPR plots")
        return

    base = df[df["feature_selection"].fillna("none") == "none"].copy()
    if base.empty:
        return

    splits = sorted(base["split_strategy"].dropna().unique())
    models = sorted(base["model"].dropna().unique())

    for split in splits:
        sub = base[base["split_strategy"] == split]
        agg = (
            sub.groupby("model")["fpr_at_recall_0_90"]
            .agg(["mean", "std"])
            .reindex(models)
        )
        means = {m: agg.loc[m, "mean"] if m in agg.index else None for m in models}
        stds  = {m: agg.loc[m, "std"]  if m in agg.index else None for m in models}

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 4))
        _bar_chart(
            ax,
            groups=models,
            series={m: [means[m]] for m in models},
            errors={m: [stds[m]] for m in models},
            ylabel="FPR @ Recall=0.90",
            title=f"FPR@Recall=0.90 by Model — split={split}",
        )
        ax.set_xlabel("Model")
        fig.tight_layout()
        fname = out_dir / f"fpr_split_{split}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        log.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Plot 3: Feature-selection effect per ML model
# ---------------------------------------------------------------------------

def plot_fs_effect(df: pd.DataFrame, out_dir: Path) -> None:
    """For each ML model: grouped bars by split, each bar = fs setting."""
    ml_models = [
        m for m in df["model"].dropna().unique()
        if m not in ("baseline_majority", "baseline_threshold")
    ]
    if not ml_models:
        log.warning("No ML model rows — skipping feature-selection plots")
        return

    def _fs_label(row) -> str:
        fs = row.get("feature_selection") or "none"
        k  = row.get("feature_selection_k")
        if fs == "none":
            return "none"
        return f"MI k={int(k)}" if k is not None else fs

    df2 = df.copy()
    df2["fs_label"] = df2.apply(_fs_label, axis=1)

    splits = sorted(df2["split_strategy"].dropna().unique())
    fs_settings = ["none"] + sorted(
        {l for l in df2["fs_label"].unique() if l != "none"}
    )

    colours = _colour_map(fs_settings)

    for model in sorted(ml_models):
        sub = df2[df2["model"] == model]
        if sub.empty:
            continue

        agg = (
            sub.groupby(["split_strategy", "fs_label"])["pr_auc"]
            .agg(["mean", "std"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(max(6, len(splits) * 1.5), 4))

        n_fs = len(fs_settings)
        width = 0.8 / n_fs
        offsets = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n_fs)
        x = np.arange(len(splits))

        for offset, fs in zip(offsets, fs_settings):
            vals, errs = [], []
            for split in splits:
                row = agg[(agg["split_strategy"] == split) & (agg["fs_label"] == fs)]
                if row.empty:
                    vals.append(float("nan"))
                    errs.append(0.0)
                else:
                    vals.append(float(row["mean"].iloc[0]))
                    std_val = float(row["std"].iloc[0])
                    errs.append(0.0 if np.isnan(std_val) else std_val)

            ax.bar(
                x + offset,
                vals,
                width=width,
                label=fs,
                color=colours[fs],
                yerr=errs,
                capsize=3,
                error_kw={"elinewidth": 1, "alpha": 0.7},
            )

        ax.set_xticks(x)
        ax.set_xticklabels(splits, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("PR-AUC")
        ax.set_xlabel("Split strategy")
        ax.set_title(f"Feature Selection Effect — {model}")
        ax.legend(title="Feature selection", fontsize=7)
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)
        fig.tight_layout()

        fname = out_dir / f"fs_effect_{model}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        log.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report plots from results_summary.csv"
    )
    parser.add_argument("--csv", required=True, help="Path to results_summary.csv")
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for PNG figures"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)

    if df.empty:
        log.error("CSV is empty — nothing to plot")
        sys.exit(1)

    plot_pr_auc_per_split(df, out_dir)
    plot_fpr_per_split(df, out_dir)
    plot_fs_effect(df, out_dir)

    log.info("Done. Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
