"""XAI diagnostic analysis: explanation quality metrics across model types.

Computes three novel metrics for a run directory:

1. Type-specificity score
   How much do a model's SHAP explanations differ per attack type?
   High = model learned distinct per-type signatures (good)
   Low  = model uses same features for all types (poor type discriminator)
   Measured as mean Jensen-Shannon divergence between per-type SHAP
   distributions and the global importance distribution.

2. Feature-type breakdown
   What fraction of a model's top-K important features are categorical
   (protocol flags, connection states, DNS fields) vs numerical (bytes, pkts)?
   Categorical-heavy → model exploits protocol signatures
   Numerical-heavy → model relies only on traffic volume

3. Cross-model SHAP divergence
   How differently do models rank features?
   High divergence on a feature = architecturally contested
   Low divergence = universally important (robust signal)

Usage
-----
    python -m hydra.analysis.xai_diagnostics --run_dir runs/ton_iot/<run_id>
    python -m hydra.analysis.xai_diagnostics --run_dir runs/ton_iot/<run_id> --top_k 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_global_importance(explain_dir: Path) -> pd.Series | None:
    """Load global importance CSV → normalised Series (feature → weight)."""
    p = explain_dir / "global_importance.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    imp = df.set_index("feature")["importance"].abs()
    total = imp.sum()
    return imp / total if total > 0 else imp


def _load_type_shap(explain_dir: Path) -> dict[str, pd.Series]:
    """Load per-class SHAP CSVs → {attack_type: normalised mean |SHAP| Series}."""
    type_dir = explain_dir / "type_classifier"
    result = {}
    if not type_dir.exists():
        return result
    for csv_path in sorted(type_dir.glob("type_shap_*.csv")):
        attack = csv_path.stem.replace("type_shap_", "")
        df = pd.read_csv(csv_path)
        feats = [c for c in df.columns if c != "true_label"]
        if not feats:
            continue
        imp = df[feats].abs().mean()
        total = imp.sum()
        result[attack] = imp / total if total > 0 else imp
    return result


def _js_divergence(p: pd.Series, q: pd.Series) -> float:
    """Jensen-Shannon divergence between two importance distributions.
    Both are aligned to their union of features before comparison.
    Returns value in [0, 1] (0 = identical, 1 = completely different).
    """
    all_feats = p.index.union(q.index)
    pv = p.reindex(all_feats).fillna(0).values.astype(float)
    qv = q.reindex(all_feats).fillna(0).values.astype(float)
    # normalise (in case of floating-point drift)
    pv = pv / (pv.sum() + 1e-12)
    qv = qv / (qv.sum() + 1e-12)
    return float(jensenshannon(pv, qv))


def type_specificity_score(global_imp: pd.Series, type_shaps: dict[str, pd.Series]) -> float:
    """Mean JSD between per-type SHAP distributions and the global distribution.

    High score → model uses different features per attack type (specific)
    Low score  → model uses same features for all attack types (generic)
    """
    if not type_shaps or global_imp is None:
        return float("nan")
    scores = [_js_divergence(global_imp, ts) for ts in type_shaps.values()]
    return float(np.mean(scores))


def feature_type_breakdown(global_imp: pd.Series, top_k: int = 10) -> dict:
    """Fraction of top-K features that are categorical vs numerical.

    Categorical features start with 'cat__' (one-hot encoded).
    Numerical features start with 'num__'.
    """
    if global_imp is None:
        return {"categorical_frac": float("nan"), "numerical_frac": float("nan")}
    top = global_imp.nlargest(top_k)
    n_cat = sum(1 for f in top.index if f.startswith("cat__"))
    n_num = sum(1 for f in top.index if f.startswith("num__"))
    total = len(top)
    return {
        "categorical_frac": n_cat / total,
        "numerical_frac":   n_num / total,
        "top_k":            total,
    }


def cross_model_divergence(
    global_imps: dict[str, pd.Series]
) -> pd.DataFrame:
    """Pairwise JSD between model global importance distributions.

    Returns a square DataFrame (models × models).
    """
    models = sorted(global_imps.keys())
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for m1 in models:
        for m2 in models:
            matrix.loc[m1, m2] = _js_divergence(global_imps[m1], global_imps[m2])
    return matrix


def feature_contestedness(
    global_imps: dict[str, pd.Series], top_k: int = 20
) -> pd.DataFrame:
    """Per-feature std of normalised importance across models.

    High std → models disagree (contested)
    Low std  → models agree (robust signal)
    Returns top_k most contested features.
    """
    all_feats = sorted(set().union(*[s.index for s in global_imps.values()]))
    aligned = pd.DataFrame(
        {m: global_imps[m].reindex(all_feats).fillna(0) for m in global_imps}
    )
    aligned["mean"] = aligned.mean(axis=1)
    aligned["std"]  = aligned.std(axis=1)
    return aligned.sort_values("std", ascending=False).head(top_k)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_specificity_bar(scores: dict[str, float], out_path: Path):
    models = list(scores.keys())
    vals   = [scores[m] for m in models]
    colors = ["#2196F3" if "xgboost" in m else
              "#4CAF50" if "forest" in m else
              "#FF9800" if "cnn" in m else "#9C27B0"
              for m in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylabel("Type-Specificity Score\n(mean JS divergence, higher = more specific)", fontsize=10)
    ax.set_title("Explanation Type-Specificity: Do Models Learn Per-Attack Signatures?", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=0.8, label="Low-specificity threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_type_breakdown(breakdowns: dict[str, dict], out_path: Path):
    models = list(breakdowns.keys())
    cat_fracs = [breakdowns[m]["categorical_frac"] for m in models]
    num_fracs = [breakdowns[m]["numerical_frac"]   for m in models]
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, cat_fracs, w, label="Categorical (protocol/flags)", color="#E91E63")
    ax.bar(x + w/2, num_fracs, w, label="Numerical (bytes/packets)",    color="#3F51B5")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Fraction of top-10 features")
    ax.set_title("Feature-Type Breakdown: Categorical vs Numerical in Top-10 SHAP Features")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_model_divergence(matrix: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.values.astype(float), cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticklabels(matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if matrix.values[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Jensen-Shannon Divergence (0=identical, 1=opposite)")
    ax.set_title("Cross-Model SHAP Divergence\n(how differently do models rank features?)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_contestedness(df: pd.DataFrame, out_path: Path, top_k: int = 15):
    top = df.head(top_k)
    models = [c for c in df.columns if c not in ("mean", "std")]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top))
    width = 0.8 / len(models)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, m in enumerate(models):
        ax.bar(x + i * width - 0.4 + width/2, top[m].values,
               width, label=m, color=colors[i % len(colors)], alpha=0.8)
    ax.set_xticks(x)
    feat_labels = [f[:30] for f in top.index]
    ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Normalised SHAP importance")
    ax.set_title(f"Top-{top_k} Most Contested Features (highest cross-model std)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary_table(
    scores: dict[str, float],
    breakdowns: dict[str, dict],
    type_accs: dict[str, float],
    out_path: Path,
):
    """Scatter: type-specificity vs type_acc, coloured by categorical fraction."""
    models = [m for m in scores if not np.isnan(scores[m])]
    xs = [scores[m] for m in models]
    ys = [type_accs.get(m, float("nan")) for m in models]
    cats = [breakdowns[m]["categorical_frac"] for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(xs, ys, c=cats, cmap="RdYlGn", s=150, edgecolors="black",
                    linewidths=0.8, vmin=0, vmax=1)
    for m, x, y in zip(models, xs, ys):
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.colorbar(sc, ax=ax, label="Categorical fraction in top-10 SHAP")
    ax.set_xlabel("Type-Specificity Score (JS divergence, ↑ better)")
    ax.set_ylabel("Attack-Type Classification Accuracy")
    ax.set_title("Specificity vs Accuracy\n(colour = categorical feature reliance)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(run_dir: Path, top_k: int = 10, type_accs: dict[str, float] | None = None):
    explain_dir = run_dir / "explain"
    out_dir = run_dir / "xai_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted([d.name for d in explain_dir.iterdir() if d.is_dir()])
    if not models:
        print(f"No model explain dirs found in {explain_dir}")
        return

    global_imps   = {}
    specificities = {}
    breakdowns    = {}

    for model in models:
        mdir = explain_dir / model
        gimp = _load_global_importance(mdir)
        type_shaps = _load_type_shap(mdir)

        if gimp is not None:
            global_imps[model] = gimp
            specificities[model] = type_specificity_score(gimp, type_shaps)
            breakdowns[model] = feature_type_breakdown(gimp, top_k=top_k)

    # --- Print summary ---
    print(f"\n=== XAI Diagnostics: {run_dir.name} ===\n")
    print(f"{'Model':<20} {'Type-Specificity':>17} {'Categorical%':>13} {'Numerical%':>11}")
    print("-" * 65)
    for m in sorted(specificities):
        spec = specificities[m]
        bd   = breakdowns[m]
        print(f"{m:<20} {spec:>17.4f} {bd['categorical_frac']:>12.1%} {bd['numerical_frac']:>10.1%}")

    # --- Plots ---
    if specificities:
        plot_specificity_bar(specificities, out_dir / "type_specificity.png")
        print(f"\n[ok] type_specificity.png → {out_dir}")

    if breakdowns:
        plot_feature_type_breakdown(breakdowns, out_dir / "feature_type_breakdown.png")
        print(f"[ok] feature_type_breakdown.png → {out_dir}")

    if len(global_imps) >= 2:
        div_matrix = cross_model_divergence(global_imps)
        plot_cross_model_divergence(div_matrix, out_dir / "cross_model_divergence.png")
        print(f"[ok] cross_model_divergence.png → {out_dir}")

        contested = feature_contestedness(global_imps, top_k=20)
        contested.to_csv(out_dir / "feature_contestedness.csv")
        plot_contestedness(contested, out_dir / "feature_contestedness.png")
        print(f"[ok] feature_contestedness.png → {out_dir}")

    if type_accs and specificities:
        plot_summary_table(specificities, breakdowns, type_accs, out_dir / "specificity_vs_accuracy.png")
        print(f"[ok] specificity_vs_accuracy.png → {out_dir}")

    print(f"\nAll outputs saved to {out_dir}\n")
    return {"specificities": specificities, "breakdowns": breakdowns,
            "global_imps": global_imps}


def main():
    parser = argparse.ArgumentParser(description="XAI diagnostic analysis for a run directory")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--top_k",  type=int, default=10,
                        help="Top-K features for feature-type breakdown (default: 10)")
    parser.add_argument("--type_accs", nargs="*", metavar="MODEL=ACC",
                        help="Optional type accuracies e.g. xgboost=0.90 random_forest=0.94")
    args = parser.parse_args()

    type_accs = {}
    if args.type_accs:
        for entry in args.type_accs:
            m, v = entry.split("=")
            type_accs[m] = float(v)

    run(args.run_dir, top_k=args.top_k, type_accs=type_accs)


if __name__ == "__main__":
    main()
