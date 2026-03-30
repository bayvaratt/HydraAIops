#!/usr/bin/env python3
"""
HydraAIops interactive pipeline runner.

Run with:  python3 run_pipeline.py
"""

import subprocess
import sys
import json
from pathlib import Path

from hydra.data.io import load_dataset_config, resolve_dataset_path

PYTHON = sys.executable
DATASETS_CONFIG = "hydra/config/datasets.yaml"

DATASETS = {
    "1": "ton_iot",
    "2": "cic_iot2023",
    "3": "both",
}

MODELS = {
    "1": "logreg",
    "2": "random_forest",
    "3": "xgboost",
    "4": "lightgbm",
    "5": "sklearn_gbdt",
    "6": "all",
}

DATASET_DEFAULTS = {
    "ton_iot": {
        "split_strategy": "stratified",
        "feature_regime": "behaviour_only",
        "type_col": "type",
        "max_rows": None,
    },
    "cic_iot2023": {
        "split_strategy": "stratified",
        "feature_regime": "behaviour_only",
        "type_col": "type",
        "max_rows": 200000,
    },
}


def prompt(message: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{message}{suffix}: ").strip()
    return raw if raw else default


def print_header():
    print()
    print("=" * 50)
    print("  HydraAIops Pipeline Runner")
    print("=" * 50)
    print()


def choose_dataset() -> list[str]:
    print("Dataset:")
    print("  1. TON_IoT")
    print("  2. CIC-IoT-2023")
    print("  3. Both")
    choice = prompt("Choice", "1")
    selected = DATASETS.get(choice, "ton_iot")
    if selected == "both":
        return ["ton_iot", "cic_iot2023"]
    return [selected]


def choose_models() -> list[str]:
    print()
    print("Model:")
    print("  1. Logistic Regression")
    print("  2. Random Forest")
    print("  3. XGBoost")
    print("  4. LightGBM")
    print("  5. sklearn GBDT")
    print("  6. All models")
    choice = prompt("Choice", "6")
    selected = MODELS.get(choice, "all")
    if selected == "all":
        return ["logreg", "random_forest", "xgboost", "lightgbm", "sklearn_gbdt"]
    return [selected]


def choose_seeds() -> list[int]:
    print()
    raw = prompt("Seeds (comma-separated)", "42")
    try:
        return [int(s.strip()) for s in raw.split(",") if s.strip()]
    except ValueError:
        print("  Invalid seeds, using default: 42")
        return [42]


def build_command(dataset: str, models: list[str], seed: int) -> list[str]:
    cfg = DATASET_DEFAULTS[dataset]
    cmd = [
        PYTHON, "-m", "hydra.experiments.run_tabular",
        "--dataset", dataset,
        "--split_strategy", cfg["split_strategy"],
        "--feature_regime", cfg["feature_regime"],
        "--type_col", cfg["type_col"],
        "--seed", str(seed),
        "--models", *models,
    ]
    if cfg["max_rows"]:
        cmd += ["--max_rows", str(cfg["max_rows"])]
    return cmd


def print_results(run_dir: Path):
    metrics_file = run_dir / "metrics_summary.csv"
    if not metrics_file.exists():
        print("  (no metrics_summary.csv found)")
        return

    try:
        import pandas as pd
        df = pd.read_csv(metrics_file)
        cols = ["model", "pr_auc", "roc_auc", "attack_type_f1_macro_detected"]
        available = [c for c in cols if c in df.columns]
        # exclude baselines from summary
        df = df[~df["model"].str.startswith("baseline")]
        print()
        print(df[available].to_string(index=False))
    except Exception as e:
        print(f"  Could not read results: {e}")


def find_latest_run(dataset: str) -> Path | None:
    results_dir = Path("results") / dataset
    if not results_dir.exists():
        return None
    dirs = sorted(results_dir.iterdir(), reverse=True)
    for d in dirs:
        if d.is_dir() and (d / "metrics_summary.csv").exists():
            return d
    return None


def check_dataset_availability(dataset: str) -> tuple[bool, str]:
    try:
        cfg = load_dataset_config(DATASETS_CONFIG, dataset)
    except Exception as exc:
        return False, f"dataset config error for '{dataset}': {exc}"

    resolved = resolve_dataset_path(cfg.path)
    if resolved.exists():
        return True, str(resolved)
    return False, f"missing dataset file: configured '{cfg.path}', tried '{resolved}'"


def run_pipeline(dataset: str, models: list[str], seed: int):
    label = dataset.replace("_", "-").upper()
    print(f"\n--- Running {label}  seed={seed}  models={models} ---")
    available, detail = check_dataset_availability(dataset)
    if not available:
        print(f"  Cannot start run: {detail}")
        print(f"  Expected dataset config: {DATASETS_CONFIG}")
        return

    cmd = build_command(dataset, models, seed)
    result = subprocess.run(cmd)
    if result.returncode == 0:
        run_dir = find_latest_run(dataset)
        if run_dir:
            print(f"\nResults from {run_dir.name}:")
            print_results(run_dir)
    else:
        print(f"\n  Run failed (exit code {result.returncode})")


def main():
    print_header()
    datasets = choose_dataset()
    models = choose_models()
    seeds = choose_seeds()

    total = len(datasets) * len(seeds)
    print(f"\nWill run {total} job(s): {datasets} x seeds={seeds}")
    print(f"Models: {models}")
    confirm = prompt("\nStart? (y/n)", "y")
    if confirm.lower() != "y":
        print("Cancelled.")
        return

    for dataset in datasets:
        for seed in seeds:
            run_pipeline(dataset, models, seed)

    print("\n" + "=" * 50)
    print("  All runs complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
