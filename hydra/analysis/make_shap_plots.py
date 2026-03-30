"""make_shap_plots.py — Generate SHAP visualisations from a run directory.

Usage:
    python -m hydra.analysis.make_shap_plots --run_dir runs/ton_iot/<timestamp>
    python -m hydra.analysis.make_shap_plots --run_dir runs/ton_iot/<ts> --model random_forest

Produces (in <run_dir>/explain/<model>/plots/):
    binary_importance.png      — mean |SHAP| bar chart for binary detection
    type_importance.png        — mean |SHAP| bar chart for type classifier (aggregated)
    type_heatmap.png           — features × attack-types mean |SHAP| heatmap
    type_beeswarm_<cls>.png    — per-class SHAP beeswarm (top-N features)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("matplotlib is required for SHAP plots: pip install matplotlib") from e


def plot_importance_bar(
    feature_names: list[str],
    importances: np.ndarray,
    title: str,
    out_path: Path,
    top_n: int = 20,
):
    plt = _require_matplotlib()
    import matplotlib.pyplot as plt  # noqa: F811 (needed after use="Agg")

    idx = np.argsort(importances)[::-1][:top_n]
    feat = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(feat[::-1], vals[::-1], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(
    feature_names: list[str],
    class_names: list[str],
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    top_n: int = 20,
):
    """matrix shape: (n_features, n_classes)"""
    plt = _require_matplotlib()
    import matplotlib.pyplot as plt  # noqa: F811

    # Select top_n features by max importance across classes
    row_max = matrix.max(axis=1)
    top_idx = np.argsort(row_max)[::-1][:top_n]
    mat = matrix[top_idx, :]
    feats = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2), max(4, top_n * 0.4)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_beeswarm(
    df_shap: pd.DataFrame,
    title: str,
    out_path: Path,
    top_n: int = 15,
):
    """df_shap columns = feature names (+ optional 'true_label'); rows = samples."""
    plt = _require_matplotlib()
    import matplotlib.pyplot as plt  # noqa: F811

    feat_cols = [c for c in df_shap.columns if c != "true_label"]
    shap_vals = df_shap[feat_cols].values
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_feats = [feat_cols[i] for i in top_idx]
    top_shap = shap_vals[:, top_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.5)))
    for j in range(top_n - 1, -1, -1):
        y_jitter = np.random.default_rng(j).uniform(-0.3, 0.3, size=top_shap.shape[0])
        ax.scatter(top_shap[:, j], j + y_jitter, alpha=0.4, s=8, c=top_shap[:, j],
                   cmap="coolwarm", vmin=-0.5, vmax=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-model plot generation
# ---------------------------------------------------------------------------

def make_binary_plots(explain_dir: Path, model: str):
    imp_csv = explain_dir / "global_importance.csv"
    if not imp_csv.exists():
        print(f"  [skip] binary global_importance.csv not found: {imp_csv}")
        return

    df = pd.read_csv(imp_csv)
    plots_dir = explain_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_importance_bar(
        df["feature"].tolist(),
        df["importance"].values,
        f"Binary detection — mean |SHAP| ({model})",
        plots_dir / "binary_importance.png",
    )
    print(f"  [ok] binary_importance.png → {plots_dir}")


def make_type_plots(explain_dir: Path, model: str):
    type_dir = explain_dir / "type_classifier"
    if not type_dir.exists():
        print(f"  [skip] type_classifier dir not found: {type_dir}")
        return

    plots_dir = explain_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Type global importance bar (from tree feature_importances_ if saved) ---
    type_imp_csv = type_dir / "type_global_importance.csv"
    if type_imp_csv.exists():
        df = pd.read_csv(type_imp_csv)
        plot_importance_bar(
            df["feature"].tolist(),
            df["importance"].values,
            f"Type classifier — feature importance ({model})",
            plots_dir / "type_importance.png",
        )
        print(f"  [ok] type_importance.png → {plots_dir}")

    # --- Collect per-class SHAP CSVs ---
    shap_csvs = sorted(type_dir.glob("type_shap_*.csv"))
    if not shap_csvs:
        print(f"  [skip] no type_shap_*.csv found in {type_dir}")
        return

    class_names = []
    per_class_mean_abs: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None

    for csv_path in shap_csvs:
        cls_name = csv_path.stem.replace("type_shap_", "")
        df = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns if c != "true_label"]
        if feature_names is None:
            feature_names = feat_cols
        shap_vals = df[feat_cols].values
        per_class_mean_abs[cls_name] = np.mean(np.abs(shap_vals), axis=0)
        class_names.append(cls_name)

        # Beeswarm per class
        plot_beeswarm(
            df,
            f"SHAP beeswarm — {cls_name} ({model})",
            plots_dir / f"type_beeswarm_{cls_name}.png",
        )
        print(f"  [ok] type_beeswarm_{cls_name}.png → {plots_dir}")

    # --- Aggregated type importance bar (mean across all classes) ---
    if feature_names is not None:
        all_mean = np.stack(list(per_class_mean_abs.values()), axis=0).mean(axis=0)
        plot_importance_bar(
            feature_names,
            all_mean,
            f"Type classifier — mean |SHAP| across classes ({model})",
            plots_dir / "type_importance_aggregated.png",
        )
        print(f"  [ok] type_importance_aggregated.png → {plots_dir}")

    # --- Heatmap: features × attack types ---
    if feature_names is not None and len(class_names) > 1:
        matrix = np.stack([per_class_mean_abs[c] for c in class_names], axis=1)  # (n_feat, n_cls)
        plot_heatmap(
            feature_names,
            class_names,
            matrix,
            f"Type classifier — mean |SHAP| heatmap ({model})",
            plots_dir / "type_heatmap.png",
        )
        print(f"  [ok] type_heatmap.png → {plots_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP visualisations from a run directory")
    parser.add_argument("--run_dir", required=True, help="Path to a single run directory")
    parser.add_argument("--model", default=None, help="Only process this model name (default: all)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    explain_root = run_dir / "explain"
    if not explain_root.exists():
        print(f"No explain/ directory found in {run_dir}")
        return

    model_dirs = sorted(explain_root.iterdir()) if explain_root.exists() else []
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if args.model and model != args.model:
            continue
        print(f"\n=== {model} ===")
        make_binary_plots(model_dir, model)
        make_type_plots(model_dir, model)

    print("\nDone.")


if __name__ == "__main__":
    main()
