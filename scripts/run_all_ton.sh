#!/usr/bin/env bash
# run_all_ton.sh — convenience wrapper: matrix → aggregate → plots
#
# Usage:
#   bash scripts/run_all_ton.sh
#   bash scripts/run_all_ton.sh --max_rows 5000
#
# All arguments are forwarded to run_ton_matrix.sh (e.g. --max_rows).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="ton_iot"
RUNS_DIR="results/${DATASET}"
AGG_DIR="${RUNS_DIR}/aggregated"
FIG_DIR="${AGG_DIR}/report_figures"

echo "========================================"
echo " HYDRA all-in-one: TON_IoT"
echo "========================================"
echo " Repo root : ${REPO_ROOT}"
echo " Runs dir  : ${RUNS_DIR}"
echo " Agg dir   : ${AGG_DIR}"
echo " Figures   : ${FIG_DIR}"
echo "========================================"

# ── Step 1: run the matrix ────────────────────────────────────────────────────
echo ""
echo "[1/3] Running experiment matrix..."
bash scripts/run_ton_matrix.sh "$@"

# ── Step 2: aggregate ─────────────────────────────────────────────────────────
echo ""
echo "[2/3] Aggregating results..."
python -m hydra.analysis.aggregate_runs \
    --runs_dir "${RUNS_DIR}" \
    --out_dir  "${AGG_DIR}"

# ── Step 3: plots ─────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Generating report figures..."
python -m hydra.analysis.make_report_plots \
    --csv     "${AGG_DIR}/results_summary.csv" \
    --out_dir "${FIG_DIR}"

echo ""
echo "========================================"
echo " Done!"
echo "  Summary CSV : ${AGG_DIR}/results_summary.csv"
echo "  Summary MD  : ${AGG_DIR}/results_summary.md"
echo "  Figures     : ${FIG_DIR}/"
echo "========================================"
