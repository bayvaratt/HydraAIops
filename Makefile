# HYDRA Experiment Makefile
#
# Targets:
#   ton-matrix              Run the full 36-run TON_IoT experiment matrix
#   aggregate RUNS_DIR=...  Aggregate run folders into results_summary.csv/md
#   plots     RUNS_DIR=...  Generate report figures from aggregated CSV
#   all-ton                 matrix → aggregate → plots end-to-end
#
# Examples:
#   make ton-matrix
#   make ton-matrix MAX_ROWS=5000
#   make aggregate RUNS_DIR=runs/ton_iot
#   make plots     RUNS_DIR=runs/ton_iot
#   make all-ton
#   make all-ton MAX_ROWS=5000

PYTHON   ?= python
DATASET  := ton_iot
RUNS_DIR ?= runs/$(DATASET)
AGG_DIR  ?= $(RUNS_DIR)/aggregated
FIG_DIR  ?= $(AGG_DIR)/report_figures

# Optional row cap for smoke testing (unset by default)
ifdef MAX_ROWS
  MAX_ROWS_ARG := --max_rows $(MAX_ROWS)
else
  MAX_ROWS_ARG :=
endif

.PHONY: ton-matrix aggregate plots all-ton cross-dataset

## Run the full TON_IoT experiment matrix (36 runs)
ton-matrix:
	bash scripts/run_ton_matrix.sh $(MAX_ROWS_ARG)

## Aggregate run folders into results_summary.csv + .md
aggregate:
	$(PYTHON) -m hydra.analysis.aggregate_runs \
		--runs_dir $(RUNS_DIR) \
		--out_dir  $(AGG_DIR)

## Generate report plots from aggregated CSV
plots:
	$(PYTHON) -m hydra.analysis.make_report_plots \
		--csv     $(AGG_DIR)/results_summary.csv \
		--out_dir $(FIG_DIR)

## End-to-end: matrix → aggregate → plots
all-ton: ton-matrix aggregate plots

## Cross-dataset generalisation (both directions)
## Usage: make cross-dataset [MAX_ROWS=50000]
cross-dataset:
	$(PYTHON) -m hydra.pipelines.run_cross_dataset \
		--source ton_iot --target cic_iot2023 \
		--models logreg random_forest xgboost \
		--seed 42 \
		$(if $(MAX_ROWS),--max_rows_source $(MAX_ROWS) --max_rows_target $(MAX_ROWS),)
	$(PYTHON) -m hydra.pipelines.run_cross_dataset \
		--source cic_iot2023 --target ton_iot \
		--models logreg random_forest xgboost \
		--seed 42 \
		$(if $(MAX_ROWS),--max_rows_source $(MAX_ROWS) --max_rows_target $(MAX_ROWS),)
