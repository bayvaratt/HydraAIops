#!/usr/bin/env python3
"""Regenerate detection_comparison.pdf and oxs_radar.pdf with layout fixes."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

# ── Figure 4.2: detection_comparison.pdf ──────────────────────────────────
def fig_detection_comparison():
    models = ["XGBoost", "LightGBM", "LogReg", "Random\nForest", "CNN-LSTM", "GNN"]
    pr_auc  = [1.000, 1.000, 1.000, 1.000, 0.9998, 0.9997]
    roc_auc = [1.000, 1.000, 1.000, 1.000, 0.9995, 0.9993]
    f1      = [1.000, 1.000, 1.000, 0.999, 0.997,  0.997]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, pr_auc,  width, label="PR-AUC",  color="#4c72b0", edgecolor="white")
    bars2 = ax.bar(x,         roc_auc, width, label="ROC-AUC", color="#dd8452", edgecolor="white")
    bars3 = ax.bar(x + width, f1,      width, label="F1",      color="#55a868", edgecolor="white")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Binary Detection Performance Across All Model Families\n(TON_IoT, behaviour-only, stratified split)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_ylim(0.995, 1.003)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Annotate bars with smaller rotated text to avoid overlap
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7, rotation=45,
            )

    fig.tight_layout()
    out = FIG_DIR / "detection_comparison.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4.6: oxs_radar.pdf ────────────────────────────────────────────
def fig_oxs_radar():
    # Data from Table 4.11.  For the radar we need normalised [0,1] scores
    # for each of the 5 OXS criteria.
    # Criteria: Faithfulness, Stability, Simplicity, Plausibility, Timeliness
    #
    # Faithfulness = normalised sufficiency (all 1.0); comprehensiveness is
    # negative for all, so we use sufficiency directly.
    # Stability = mean of S1 and S2
    # Simplicity = Gini
    # Plausibility = RMA@5
    # Timeliness = 1/(1 + t/1000)  where t is SHAP ms

    models_data = {
        "XGBoost": {
            "Faithfulness": 1.0,
            "Stability": (0.831 + 1.000) / 2,
            "Simplicity": 0.998,
            "Plausibility": 0.80,
            "Timeliness": 1 / (1 + 0.04 / 1000),
        },
        "LightGBM": {
            "Faithfulness": 1.0,
            "Stability": (0.836 + 1.000) / 2,
            "Simplicity": 0.998,
            "Plausibility": 0.60,
            "Timeliness": 1 / (1 + 0.48 / 1000),
        },
        "LogReg": {
            "Faithfulness": 1.0,
            "Stability": (0.972 + 0.415) / 2,
            "Simplicity": 0.984,
            "Plausibility": 0.40,
            "Timeliness": 1 / (1 + 0.001 / 1000),
        },
        "Random Forest": {
            "Faithfulness": 1.0,
            "Stability": (0.120 + 0.999) / 2,
            "Simplicity": 0.976,
            "Plausibility": 0.40,
            "Timeliness": 1 / (1 + 1.43 / 1000),
        },
    }

    categories = ["Faithfulness", "Stability", "Simplicity", "Plausibility", "Timeliness"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colours = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for (model, data), colour in zip(models_data.items(), colours):
        values = [data[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=colour)
        ax.fill(angles, values, alpha=0.08, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")

    # Title above chart, legend below
    ax.set_title("OXS Five-Criterion Radar Comparison\n(Behaviour-Only Regime)",
                 fontsize=13, fontweight="bold", pad=30)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=4, fontsize=10, frameon=False)

    fig.tight_layout()
    out = FIG_DIR / "oxs_radar.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_detection_comparison()
    fig_oxs_radar()
    print("Done.")
