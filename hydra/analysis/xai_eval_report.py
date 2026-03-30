"""XAI Evaluation Report — aggregate 5-criteria scores across models in a run.

Reads explain/<model>/xai_eval.json for each model in a run directory and
prints a formatted comparison table.  Optionally writes a CSV.

Usage
-----
    python -m hydra.analysis.xai_eval_report --run_dir runs/ton_iot/<run_id>
    python -m hydra.analysis.xai_eval_report --run_dir runs/ton_iot/<run_id> --csv out.csv

Criteria summary
----------------
  Faithfulness  comp@5  : mean P(attack) drop when top-5 features ablated     ↑ better
                suff@5  : fraction of confidence kept with only top-5 features ↑ better
  Stability     spearman: Spearman ρ of |attribution| ranks under noise        ↑ better
                jaccard5: top-5 feature Jaccard overlap under noise            ↑ better
  Simplicity    k90%    : fraction of features needed to cover 90% attribution  ↓ better
                gini    : Gini concentration of importance distribution         ↑ better
  Plausibility  RMA@10  : fraction of top-10 features matching expert set       ↑ better
  Timeliness    ms/samp : wall-clock ms per sample explanation                  ↓ better
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Metric extraction spec
# ---------------------------------------------------------------------------

# (json_section, json_key, display_name, higher_is_better)
_METRICS = [
    ("faithfulness", "comprehensiveness_k5",   "Faith-Comp@5",   True),
    ("faithfulness", "sufficiency_k5",         "Faith-Suff@5",   True),
    ("faithfulness", "comprehensiveness_k10",  "Faith-Comp@10",  True),
    ("faithfulness", "sufficiency_k10",        "Faith-Suff@10",  True),
    ("stability",    "mean_spearman_rank_corr","Stab-Spearman",  True),
    ("stability",    "mean_top5_jaccard",      "Stab-Jac@5",     True),
    ("simplicity",   "k90_frac",              "Simp-k90%",       False),
    ("simplicity",   "gini_coeff",            "Simp-Gini",       True),
    ("plausibility", "rma_at_k",              "Plaus-RMA@10",    True),
    ("timeliness",   "ms_per_sample",         "Time-ms/samp",    False),
]


def load_eval_results(run_dir: Path) -> pd.DataFrame:
    """Load all xai_eval.json files from run_dir/explain/*/xai_eval.json."""
    explain_dir = run_dir / "explain"
    if not explain_dir.exists():
        raise FileNotFoundError(f"No explain/ directory in {run_dir}")

    rows = []
    for model_dir in sorted(explain_dir.iterdir()):
        eval_path = model_dir / "xai_eval.json"
        if not eval_path.exists():
            continue
        with open(eval_path) as f:
            data = json.load(f)

        row = {"model": model_dir.name}
        for section, key, display, _ in _METRICS:
            section_data = data.get(section, {})
            val = section_data.get(key, None)
            row[display] = val

        # Extra info
        row["n_test"] = data.get("n_test_samples")
        row["n_feat"] = data.get("n_features")
        row["n_attr"] = data.get("n_attributed_samples")
        row["plaus_expert_found"] = len(
            data.get("plausibility", {}).get("expert_features_found", [])
        )
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No xai_eval.json files found under {explain_dir}")

    return pd.DataFrame(rows).set_index("model")


def _fmt(val, higher_is_better: bool, best_val, worst_val) -> str:
    """Format a metric value, marking best with * and worst with (parens)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "  —  "
    s = f"{val:.4f}"
    if best_val is not None and abs(val - best_val) < 1e-6:
        s += "*"
    elif worst_val is not None and abs(val - worst_val) < 1e-6:
        s = f"({s})"
    return s


def format_table(df: pd.DataFrame) -> str:
    """Return a readable text table with best/worst annotations."""
    lines = []
    metric_names = [d for _, _, d, _ in _METRICS]
    available = [d for d in metric_names if d in df.columns]

    # Header
    col_w = 14
    model_w = max(20, max(len(m) for m in df.index) + 2)
    header = f"{'Model':<{model_w}}" + "".join(f"{d:>{col_w}}" for d in available)
    lines.append(header)
    lines.append("-" * len(header))

    # Best/worst per column
    best = {}
    worst = {}
    for _, _, display, hib in _METRICS:
        if display not in df.columns:
            continue
        col = df[display].dropna()
        if col.empty:
            best[display] = worst[display] = None
        else:
            best[display]  = col.max() if hib else col.min()
            worst[display] = col.min() if hib else col.max()

    for model, row in df.iterrows():
        cells = f"{model:<{model_w}}"
        for _, _, display, hib in _METRICS:
            if display not in df.columns:
                continue
            val = row[display]
            fv = float(val) if val is not None and not pd.isna(val) else None
            cells += f"{_fmt(fv, hib, best[display], worst[display]):>{col_w}}"
        lines.append(cells)

    lines.append("")
    lines.append("* = best in column   (x) = worst in column")
    lines.append("Faithfulness ↑, Stability ↑, Simplicity: k90% ↓ Gini ↑, Plausibility ↑, Timeliness ↓")
    return "\n".join(lines)


def print_top_features(run_dir: Path, top_k: int = 5):
    """Print top-k plausibility features per model."""
    explain_dir = run_dir / "explain"
    print("\n=== Plausibility: Top-10 Model Features vs Expert Set ===\n")
    for model_dir in sorted(explain_dir.iterdir()):
        eval_path = model_dir / "xai_eval.json"
        if not eval_path.exists():
            continue
        with open(eval_path) as f:
            data = json.load(f)
        plaus = data.get("plausibility", {})
        top_feats   = plaus.get("top_k_features", [])[:top_k]
        expert_found = plaus.get("expert_features_found", [])
        rma = plaus.get("rma_at_k", "?")
        print(f"{model_dir.name}  RMA@10={rma:.3f}" if isinstance(rma, float) else
              f"{model_dir.name}  RMA@10=?")
        for f in top_feats:
            marker = " ✓" if f in expert_found else ""
            print(f"  {f}{marker}")
        print()


def run(
    run_dir: Path,
    csv_path: Optional[Path] = None,
    show_features: bool = False,
):
    print(f"\n=== XAI Evaluation Report: {run_dir.name} ===\n")
    df = load_eval_results(run_dir)
    print(format_table(df))

    if csv_path is not None:
        df.to_csv(csv_path)
        print(f"\n[ok] CSV written to {csv_path}")

    if show_features:
        print_top_features(run_dir)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Print XAI 5-criteria evaluation table for a run directory"
    )
    parser.add_argument("--run_dir",       required=True, type=Path)
    parser.add_argument("--csv",           type=Path, default=None,
                        help="Optional: write results to CSV")
    parser.add_argument("--show_features", action="store_true",
                        help="Print top plausibility features per model")
    args = parser.parse_args()
    run(args.run_dir, csv_path=args.csv, show_features=args.show_features)


if __name__ == "__main__":
    main()
