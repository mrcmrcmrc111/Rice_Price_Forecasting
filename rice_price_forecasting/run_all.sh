#!/usr/bin/env bash
# Runs the whole pipeline in order. Usage: bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 01_inspect_data.py
python3 02_prepare_data.py
python3 03_eda.py
python3 04_train_evaluate.py
echo "Done. See outputs/ and models/."
