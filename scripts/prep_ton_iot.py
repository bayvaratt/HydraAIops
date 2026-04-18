"""
Combine 23 split TON_IoT network CSVs into a single ton_iot.csv.

Usage:
    python scripts/prep_ton_iot.py \
        --in_dir  data/raw/ton_iot/OneDrive_1_3-30-2026 \
        --out     data/ton_iot/ton_iot.csv
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/raw/ton_iot/OneDrive_1_3-30-2026")
    parser.add_argument("--out", default="data/ton_iot/ton_iot.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "Network_dataset_*.csv")))
    if not files:
        raise FileNotFoundError(f"No Network_dataset_*.csv files found in {args.in_dir}")

    print(f"Found {len(files)} files — reading...")
    chunks = []
    for f in files:
        df = pd.read_csv(f)
        chunks.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,} rows")

    combined = pd.concat(chunks, ignore_index=True)
    print(f"\nCombined: {len(combined):,} rows")

    # Rename ts → timestamp (expected by datasets.yaml)
    if "ts" in combined.columns and "timestamp" not in combined.columns:
        combined = combined.rename(columns={"ts": "timestamp"})
        print("Renamed 'ts' → 'timestamp'")

    # Sanity checks
    assert "label" in combined.columns, "Missing 'label' column"
    assert "type" in combined.columns, "Missing 'type' column"
    assert "timestamp" in combined.columns, "Missing 'timestamp' column"

    print(f"label distribution:\n{combined['label'].value_counts()}")
    print(f"type distribution:\n{combined['type'].value_counts()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
