"""Prepare CIC-IoT-2023 merged CSVs into a single pipeline-ready file.

Usage:
    python scripts/prep_cic_iot2023.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/cic_iot2023")
OUT_PATH = Path("data/cic_iot2023/cic_iot2023.csv")
BENIGN_LABEL = "BENIGN"


def normalise_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def main():
    raw_files = sorted(RAW_DIR.glob("Merged*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No Merged*.csv found in {RAW_DIR}")

    print(f"Found {len(raw_files)} files: {[f.name for f in raw_files]}")

    chunks = []
    for f in raw_files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [normalise_col(c) for c in df.columns]
        chunks.append(df)
        print(f"  {f.name}: {len(df):,} rows")

    combined = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal rows: {len(combined):,}")

    # Replace inf/-inf with NaN — common in CIC-IoT-2023 rate/IAT features
    num_cols = combined.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(combined[num_cols]).sum()
    total_inf = int(inf_counts.sum())
    if total_inf:
        print(f"Replacing {total_inf:,} inf values with NaN")
        combined[num_cols] = combined[num_cols].replace([np.inf, -np.inf], np.nan)

    # Deduplicate on feature columns (keep type for grouping, exclude label not yet created)
    feat_cols = [c for c in combined.columns if c != "label"]
    before = len(combined)
    combined = combined.drop_duplicates(subset=feat_cols, keep="first").reset_index(drop=True)
    print(f"Deduplicated: {before:,} → {len(combined):,} rows ({before - len(combined):,} duplicates removed)")

    # Rename label → type, create binary label
    combined = combined.rename(columns={"label": "type"})
    combined["label"] = (combined["type"] != BENIGN_LABEL).astype(int)

    # Report
    n_benign = (combined["label"] == 0).sum()
    n_attack = (combined["label"] == 1).sum()
    print(f"Benign: {n_benign:,}  Attack: {n_attack:,}  Prevalence: {n_attack/len(combined):.3f}")
    print(f"\nAttack type distribution:\n{combined['type'].value_counts().to_string()}")

    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
