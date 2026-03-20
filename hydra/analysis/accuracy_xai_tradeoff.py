"""Accuracy vs XAI quality tradeoff analysis.

Joins metrics_summary.csv (accuracy) with explain/*/xai_eval.json (XAI criteria)
for a single run and produces:

  - A ranked comparison table (model × metric, with composite scores)
  - Spearman rank correlation between accuracy rank and XAI rank
  - A scatter plot: accuracy composite vs XAI composite
  - Optional: DeLong pairwise significance table (reads delong_tests.csv)

Composite scores
----------------
  accuracy_score  = mean of normalised [ROC-AUC, PR-AUC, F1@0.9]
  xai_score       = mean of normalised [Faithfulness, Stability, Simplicity, Plausibility]
                    (Timeliness excluded from composite — reported separately)

Each metric is min-max normalised to [0,1] across the models in the run,
with direction applied (↑ or ↓ better) before averaging.

Usage
-----
    python -m hydra.analysis.accuracy_xai_tradeoff \\
        --run_dir runs/ton_iot/<run_id>

    python -m hydra.analysis.accuracy_xai_tradeoff \\
        --run_dir runs/ton_iot/<run_id> --csv tradeoff.csv --plot tradeoff.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric specs: (display_name, source, key, higher_is_better, weight)
# ---------------------------------------------------------------------------

_ACCURACY_METRICS = [
    ("ROC-AUC",    "roc_auc",                    True,  1.0),
    ("PR-AUC",     "pr_auc",                     True,  1.0),
    ("F1@0.9",     "f1_test_at_recall_0_90",     True,  1.0),
]

# (display, xai_section, xai_key, higher_is_better, in_composite)
_XAI_METRICS = [
    ("Faith-Comp@5",  "faithfulness", "comprehensiveness_k5",    True,  True),
    ("Faith-Suff@5",  "faithfulness", "sufficiency_k5",          True,  True),
    ("Stab-Spearman", "stability",    "mean_spearman_rank_corr", True,  True),
    ("Stab-Jac@5",    "stability",    "mean_top5_jaccard",       True,  True),
    ("Simp-k90%",     "simplicity",   "k90_frac",                False, True),
    ("Simp-Gini",     "simplicity",   "gini_coeff",              True,  True),
    ("Plaus-RMA@10",  "plausibility", "rma_at_k",                True,  True),
    ("Time-ms/samp",  "timeliness",   "ms_per_sample",           False, False),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_accuracy(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"metrics_summary.csv not found in {run_dir}")
    df = pd.read_csv(path)
    # Exclude baselines
    df = df[~df["model"].str.startswith("baseline_")].copy()
    return df.set_index("model")


def _load_xai(run_dir: Path) -> pd.DataFrame:
    explain_dir = run_dir / "explain"
    if not explain_dir.exists():
        return pd.DataFrame()
    rows = []
    for model_dir in sorted(explain_dir.iterdir()):
        eval_path = model_dir / "xai_eval.json"
        if not eval_path.exists():
            continue
        with open(eval_path) as f:
            data = json.load(f)
        row = {"model": model_dir.name}
        for display, section, key, _, _ in _XAI_METRICS:
            row[display] = data.get(section, {}).get(key, None)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model")


def _load_delong(run_dir: Path) -> Optional[pd.DataFrame]:
    path = run_dir / "delong_tests.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Normalisation + composite
# ---------------------------------------------------------------------------

def _minmax_norm(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Min-max normalise to [0,1] with direction applied."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    normed = (series - lo) / (hi - lo)
    return normed if higher_is_better else (1.0 - normed)


def _composite(df: pd.DataFrame, specs: list, filter_in_composite: bool = True) -> pd.Series:
    """Compute equal-weight composite score from normalised metrics."""
    parts = []
    for item in specs:
        if filter_in_composite and len(item) == 5 and not item[4]:
            continue  # skip if not in composite
        display = item[0]
        hib = item[2] if len(item) >= 3 else True
        if display not in df.columns:
            continue
        col = df[display].dropna()
        if col.empty:
            continue
        norm = _minmax_norm(col, hib)
        parts.append(norm.reindex(df.index))

    if not parts:
        return pd.Series(np.nan, index=df.index)
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _rank_label(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Return rank labels as strings (#1 best = ★)."""
    ranks = series.rank(ascending=not higher_is_better, method="min").astype(int)
    return ranks.map(lambda r: f"#{r}")


def format_tradeoff_table(
    acc_df: pd.DataFrame,
    xai_df: pd.DataFrame,
    acc_composite: pd.Series,
    xai_composite: pd.Series,
    spearman_rho: float,
    spearman_p: float,
) -> str:
    lines = []
    all_models = acc_composite.index.union(xai_composite.index)

    # Build display table
    rows = []
    for model in sorted(all_models):
        row = {"Model": model}
        for display, col, hib, _ in _ACCURACY_METRICS:
            val = acc_df.loc[model, col] if model in acc_df.index and col in acc_df.columns else None
            row[display] = f"{val:.4f}" if val is not None and not pd.isna(val) else "—"
        row["Acc-Score"] = f"{acc_composite.get(model, np.nan):.3f}" if not pd.isna(acc_composite.get(model, np.nan)) else "—"
        for display, *_ in _XAI_METRICS:
            val = xai_df.loc[model, display] if model in xai_df.index and display in xai_df.columns else None
            row[display] = f"{val:.3f}" if val is not None and not pd.isna(val) else "—"
        row["XAI-Score"] = f"{xai_composite.get(model, np.nan):.3f}" if not pd.isna(xai_composite.get(model, np.nan)) else "—"
        row["Acc-Rank"] = _rank_label(acc_composite, higher_is_better=True).get(model, "—")
        row["XAI-Rank"] = _rank_label(xai_composite, higher_is_better=True).get(model, "—")
        rows.append(row)

    display_df = pd.DataFrame(rows).set_index("Model")

    # Accuracy block
    lines.append("=== Accuracy Metrics ===")
    acc_cols = [d for d, *_ in _ACCURACY_METRICS] + ["Acc-Score", "Acc-Rank"]
    lines.append(display_df[acc_cols].to_string())
    lines.append("")

    # XAI block
    lines.append("=== XAI Quality Metrics ===")
    xai_cols = [d for d, *_ in _XAI_METRICS] + ["XAI-Score", "XAI-Rank"]
    available_xai = [c for c in xai_cols if c in display_df.columns]
    lines.append(display_df[available_xai].to_string())
    lines.append("")

    # Tradeoff summary
    lines.append("=== Tradeoff Summary ===")
    if not pd.isna(spearman_rho):
        direction = "positive (accuracy ↑ → explanation quality ↑)" if spearman_rho > 0 else "negative (accuracy ↑ → explanation quality ↓)"
        sig = "p<0.05 *" if spearman_p < 0.05 else f"p={spearman_p:.3f} n.s."
        lines.append(f"Spearman ρ(Acc-Rank, XAI-Rank) = {spearman_rho:+.3f}  [{sig}]")
        lines.append(f"Correlation direction: {direction}")
        if abs(spearman_rho) < 0.3:
            lines.append("Interpretation: weak tradeoff — accuracy and explanation quality are largely independent")
        elif spearman_rho < -0.3:
            lines.append("Interpretation: moderate negative tradeoff — higher-accuracy models explain less well")
        else:
            lines.append("Interpretation: positive correlation — no accuracy-explanation tradeoff detected")
    else:
        lines.append("Spearman ρ: insufficient data (need ≥3 models with both accuracy and XAI scores)")
    lines.append("")
    lines.append("Acc-Score: equal-weight mean of normalised [ROC-AUC, PR-AUC, F1@0.9]")
    lines.append("XAI-Score: equal-weight mean of normalised [Faith, Stability, Simplicity, Plausibility]")
    lines.append("(Timeliness excluded from composite — lower=better does not combine cleanly)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_tradeoff(
    acc_composite: pd.Series,
    xai_composite: pd.Series,
    out_path: Path,
    spearman_rho: float,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    models = acc_composite.index.intersection(xai_composite.index)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for i, model in enumerate(sorted(models)):
        x = acc_composite.get(model, np.nan)
        y = xai_composite.get(model, np.nan)
        if pd.isna(x) or pd.isna(y):
            continue
        ax.scatter(x, y, s=120, color=colors[i], zorder=3, label=model)
        ax.annotate(model, (x, y), textcoords="offset points",
                    xytext=(6, 3), fontsize=9)

    ax.set_xlabel("Accuracy Score (↑ better)\n[norm. mean of ROC-AUC, PR-AUC, F1@0.9]", fontsize=10)
    ax.set_ylabel("XAI Quality Score (↑ better)\n[norm. mean of Faith, Stability, Simplicity, Plaus]", fontsize=10)
    rho_str = f"ρ = {spearman_rho:+.3f}" if not pd.isna(spearman_rho) else "ρ = n/a"
    ax.set_title(f"Accuracy vs XAI Quality Tradeoff  ({rho_str})", fontsize=12)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0.5, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0.5, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Scatter plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    run_dir: Path,
    csv_path: Optional[Path] = None,
    plot_path: Optional[Path] = None,
    show_delong: bool = False,
) -> pd.DataFrame:
    print(f"\n=== Accuracy-XAI Tradeoff: {run_dir.name} ===\n")

    acc_df = _load_accuracy(run_dir)
    xai_df = _load_xai(run_dir)

    if xai_df.empty:
        print("[warn] No xai_eval.json files found — run the pipeline with XAI evaluation first.")
        print("       Run: python3 -m hydra.pipelines.run_tabular ... --models <model_names>")
        return pd.DataFrame()

    # Composite scores
    acc_composite = _composite(acc_df, [(d, None, hib, w) for d, col, hib, w in _ACCURACY_METRICS], filter_in_composite=False)
    xai_composite = _composite(xai_df, _XAI_METRICS)

    # Spearman rank correlation
    common = acc_composite.dropna().index.intersection(xai_composite.dropna().index)
    if len(common) >= 3:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(
            acc_composite.loc[common].values,
            xai_composite.loc[common].values,
        )
        spearman_rho, spearman_p = float(rho), float(pval)
    else:
        spearman_rho, spearman_p = float("nan"), float("nan")

    # Print table
    print(format_tradeoff_table(acc_df, xai_df, acc_composite, xai_composite, spearman_rho, spearman_p))

    # DeLong table
    if show_delong:
        delong_df = _load_delong(run_dir)
        if delong_df is not None:
            print("=== DeLong Pairwise AUC Significance Tests ===")
            print(delong_df[["model_a", "model_b", "auc_a", "auc_b", "delta_auc", "z_stat", "p_value", "significant_005"]].to_string(index=False))
            print()
        else:
            print("[warn] delong_tests.csv not found — run pipeline first.\n")

    # Combine and optionally save CSV
    combined = pd.concat([
        acc_df[["roc_auc", "pr_auc", "f1_test_at_recall_0_90"]].rename(columns={
            "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
            "f1_test_at_recall_0_90": "F1@0.9",
        }),
        acc_composite.rename("Acc-Score"),
        xai_df[[d for d, *_ in _XAI_METRICS if d in xai_df.columns]],
        xai_composite.rename("XAI-Score"),
    ], axis=1)

    if csv_path is not None:
        combined.to_csv(csv_path)
        print(f"[ok] CSV → {csv_path}")

    if plot_path is not None:
        plot_tradeoff(acc_composite, xai_composite, plot_path, spearman_rho)

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Accuracy vs XAI quality tradeoff analysis for a run"
    )
    parser.add_argument("--run_dir",      required=True, type=Path)
    parser.add_argument("--csv",          type=Path, default=None)
    parser.add_argument("--plot",         type=Path, default=None,
                        help="Path for scatter plot PNG")
    parser.add_argument("--show_delong",  action="store_true",
                        help="Print DeLong pairwise AUC significance table")
    args = parser.parse_args()
    run(args.run_dir, csv_path=args.csv, plot_path=args.plot, show_delong=args.show_delong)


if __name__ == "__main__":
    main()
