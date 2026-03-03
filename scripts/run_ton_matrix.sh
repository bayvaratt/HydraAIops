#!/usr/bin/env bash
# =============================================================================
# scripts/run_ton_matrix.sh — Run the frozen TON_IoT experiment matrix.
#
# Loops over splits × seeds × feature-selection configs defined in
# docs/experiment_plan.md.  Each individual run failure is captured and
# reported at the end; the script exits non-zero if any run failed.
#
# Usage:
#   bash scripts/run_ton_matrix.sh [--max_rows N]
#
# Requirements:
#   - Run from the HYDRA repo root directory.
#   - bash 4+  (on macOS: brew install bash && hash -r)
#   - The ton_iot dataset must be present (see hydra/config/datasets.yaml).
# =============================================================================
set -euo pipefail

# ---- Bash version guard (3.2+ required; empty-array set -u handled inline) --
if [[ "${BASH_VERSINFO[0]}" -lt 3 ]] || \
   ([[ "${BASH_VERSINFO[0]}" -eq 3 ]] && [[ "${BASH_VERSINFO[1]}" -lt 2 ]]); then
    echo "ERROR: bash 3.2+ is required (found ${BASH_VERSION})." >&2
    exit 1
fi

# ---- Repo root guard --------------------------------------------------------
if [[ ! -f "pyproject.toml" ]]; then
    echo "ERROR: Must be run from the HYDRA repo root directory." >&2
    exit 1
fi

# ---- Frozen matrix config ---------------------------------------------------
DATASET="ton_iot"
FEATURE_REGIME="behaviour_only"
GROUP_COL="src_ip"
TIMESTAMP_COL="timestamp"
TYPE_COL="type"
NORMAL_TYPE="normal"
DATASETS_CFG="hydra/config/datasets.yaml"
DEFAULTS_CFG="hydra/config/defaults.yaml"

declare -a SPLITS=(host temporal group_type_stratified)
declare -a SEEDS=(21 42 84)
declare -a FS_K_VALUES=(20 40)

declare -a BASE_MODELS=(baseline_majority baseline_threshold)
declare -a ML_MODELS=(logreg random_forest)

MAX_ROWS=""

# ---- Parse CLI args ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_rows) MAX_ROWS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---- Detect optional xgboost ------------------------------------------------
if python -c "import xgboost" 2>/dev/null; then
    ML_MODELS+=(xgboost)
    echo "INFO: xgboost detected; included in ML_MODELS."
else
    echo "INFO: xgboost not installed; skipping xgboost models."
fi

# ---- Output setup -----------------------------------------------------------
MATRIX_TS=$(date +%Y%m%d_%H%M%S)
MATRIX_DIR="runs/ton_matrix/${MATRIX_TS}"
mkdir -p "${MATRIX_DIR}"
MANIFEST="${MATRIX_DIR}/manifest.txt"

{
    echo "# HYDRA TON_IoT experiment matrix — ${MATRIX_TS}"
    echo "# dataset=${DATASET}  feature_regime=${FEATURE_REGIME}"
    echo "# splits=${SPLITS[*]}  seeds=${SEEDS[*]}"
    echo "# base_models=${BASE_MODELS[*]}"
    echo "# ml_models=${ML_MODELS[*]}"
    echo "# fs_k_values=${FS_K_VALUES[*]}"
    [[ -n "${MAX_ROWS}" ]] && echo "# max_rows=${MAX_ROWS}"
    echo ""
} > "${MANIFEST}"

declare -a FAILED=()
declare -a SUCCEEDED=()
N_TOTAL=0

# ---- Helper: execute one run config, log result -----------------------------
run_config() {
    local desc="$1"; shift
    local -a cmd=("$@")
    N_TOTAL=$((N_TOTAL + 1))

    printf "\n[%s] (%d) %s\n" "$(date +%T)" "${N_TOTAL}" "${desc}" | tee -a "${MANIFEST}"
    printf "  CMD: %s\n" "${cmd[*]}" >> "${MANIFEST}"

    local ok=true
    "${cmd[@]}" >> "${MANIFEST}" 2>&1 || ok=false

    if $ok; then
        printf "  => OK\n" | tee -a "${MANIFEST}"
        SUCCEEDED+=("${desc}")
    else
        printf "  => FAILED\n" | tee -a "${MANIFEST}"
        FAILED+=("${desc}")
    fi
}

# ---- Main matrix loop -------------------------------------------------------
for SPLIT in "${SPLITS[@]}"; do

    # Build split-specific extra args
    declare -a SPLIT_ARGS=()
    case "${SPLIT}" in
        host)
            SPLIT_ARGS=(--group_col "${GROUP_COL}")
            ;;
        temporal)
            SPLIT_ARGS=(--timestamp_col "${TIMESTAMP_COL}")
            ;;
        group_type_stratified)
            SPLIT_ARGS=(--group_col "${GROUP_COL}")
            ;;
    esac

    # Two-stage type classification args (applied to all splits)
    declare -a TYPE_ARGS=(--type_col "${TYPE_COL}" --normal_type_value "${NORMAL_TYPE}")

    # Optional max_rows arg
    declare -a MAX_ROWS_ARG=()
    [[ -n "${MAX_ROWS}" ]] && MAX_ROWS_ARG=(--max_rows "${MAX_ROWS}")

    for SEED in "${SEEDS[@]}"; do

        # 1. Baselines — always feature_selection=none (unaffected anyway)
        run_config "${DATASET}/${SPLIT}/seed=${SEED}/baselines" \
            python -m hydra.pipelines.run_tabular \
            --dataset "${DATASET}" \
            --feature_regime "${FEATURE_REGIME}" \
            --split_strategy "${SPLIT}" \
            "${SPLIT_ARGS[@]}" \
            "${TYPE_ARGS[@]}" \
            --seed "${SEED}" \
            --models "${BASE_MODELS[@]}" \
            --feature_selection none \
            --datasets "${DATASETS_CFG}" \
            --defaults "${DEFAULTS_CFG}" \
            "${MAX_ROWS_ARG[@]+"${MAX_ROWS_ARG[@]}"}"

        # 2. ML models — no feature selection
        run_config "${DATASET}/${SPLIT}/seed=${SEED}/ml-fs=none" \
            python -m hydra.pipelines.run_tabular \
            --dataset "${DATASET}" \
            --feature_regime "${FEATURE_REGIME}" \
            --split_strategy "${SPLIT}" \
            "${SPLIT_ARGS[@]}" \
            "${TYPE_ARGS[@]}" \
            --seed "${SEED}" \
            --models "${ML_MODELS[@]}" \
            --feature_selection none \
            --datasets "${DATASETS_CFG}" \
            --defaults "${DEFAULTS_CFG}" \
            "${MAX_ROWS_ARG[@]+"${MAX_ROWS_ARG[@]}"}"

        # 3. ML models — mutual_info feature selection (k=20, k=40)
        for K in "${FS_K_VALUES[@]}"; do
            run_config "${DATASET}/${SPLIT}/seed=${SEED}/ml-fs=mi-k${K}" \
                python -m hydra.pipelines.run_tabular \
                --dataset "${DATASET}" \
                --feature_regime "${FEATURE_REGIME}" \
                --split_strategy "${SPLIT}" \
                "${SPLIT_ARGS[@]}" \
                "${TYPE_ARGS[@]}" \
                --seed "${SEED}" \
                --models "${ML_MODELS[@]}" \
                --feature_selection mutual_info \
                --feature_selection_k "${K}" \
                --datasets "${DATASETS_CFG}" \
                --defaults "${DEFAULTS_CFG}" \
                "${MAX_ROWS_ARG[@]+"${MAX_ROWS_ARG[@]}"}"
        done

    done
done

# ---- Summary ----------------------------------------------------------------
{
    echo ""
    echo "=== Matrix summary ==="
    echo "Total run-configs: ${N_TOTAL}"
    echo "Succeeded:         ${#SUCCEEDED[@]}"
    echo "Failed:            ${#FAILED[@]}"
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo ""
        echo "Failed runs:"
        for f in "${FAILED[@]}"; do echo "  - ${f}"; done
    fi
} | tee -a "${MANIFEST}"

echo ""
echo "Manifest : ${MANIFEST}"
echo "Run dirs : runs/${DATASET}/"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "WARNING: ${#FAILED[@]} run(s) failed. See ${MANIFEST} for details." >&2
    exit 1
fi
