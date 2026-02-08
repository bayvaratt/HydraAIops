#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

TON_CSV="$ROOT/data/ton_iot/raw/train_test_network.csv"
HDFS_DIR="$ROOT/data/hdfs"
AUTH_GZ="$ROOT/data/lanl/raw/auth.txt.gz"
RED_GZ="$ROOT/data/lanl/raw/redteam.txt.gz"

if [[ ! -f "$TON_CSV" ]]; then
  echo "Missing TON_IoT CSV: $TON_CSV" >&2
  exit 1
fi
if [[ ! -d "$HDFS_DIR" ]]; then
  echo "Missing HDFS dir: $HDFS_DIR" >&2
  exit 1
fi
if [[ ! -f "$AUTH_GZ" ]]; then
  echo "Missing LANL auth gz: $AUTH_GZ" >&2
  exit 1
fi
if [[ ! -f "$RED_GZ" ]]; then
  echo "Missing LANL redteam gz: $RED_GZ" >&2
  exit 1
fi

run_head() {
  local head="$1"
  local varname="$2"
  shift 2
  local log out_dir
  log="$(mktemp "$ROOT/.smoke_${head}.XXXXXX")"
  PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}" $PY -m hydra_pipeline "$head" "$@" | tee "$log"
  out_dir="$(awk -F': ' -v head="$head" '$0 ~ "^Output directory \\(" head "\\): " {print $2; exit}' "$log")"
  rm -f "$log"
  if [[ -z "$out_dir" ]]; then
    echo "Failed to detect output directory for ${head}" >&2
    exit 1
  fi
  echo "Output directory (${head}): ${out_dir}"
  printf -v "$varname" '%s' "$out_dir"
}

run_head ton ton_out \
  --csv "$TON_CSV" \
  --out "$ROOT/runs/ton" \
  --max_rows 20000

run_head hdfs hdfs_out \
  --hdfs_dir "$HDFS_DIR" \
  --out "$ROOT/runs/hdfs" \
  --max_lines 100000 \
  --epochs 1

run_head lanl lanl_out \
  --auth_gz "$AUTH_GZ" \
  --redteam_gz "$RED_GZ" \
  --out "$ROOT/runs/lanl" \
  --window_seconds 7200 \
  --max_auth_lines 10000 \
  --scan_only

auth_window="$lanl_out/auth_window.csv"
redteam_window="$lanl_out/redteam_window.csv"
if [[ ! -f "$auth_window" ]]; then
  echo "Missing LANL auth window: $auth_window" >&2
  exit 1
fi
if [[ ! -f "$redteam_window" ]]; then
  echo "Missing LANL redteam window: $redteam_window" >&2
  exit 1
fi
echo "LANL window files created: $auth_window, $redteam_window"
