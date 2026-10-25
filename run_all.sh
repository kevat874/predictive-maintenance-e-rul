#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

echo "=== Unit tests ==="
python -m pytest tests/ -q

echo "=== Train FD001 (all models) ==="
python -m src.train --subset FD001 --model all

echo "=== Train best models on FD002–FD004 ==="
python -m src.train --subset FD002 --model xgb,lstm
python -m src.train --subset FD003 --model xgb,lstm
python -m src.train --subset FD004 --model xgb,lstm

echo "=== Done. Results in reports/results.csv ==="
echo "Launch app: PYTHONPATH=. streamlit run app/streamlit_app.py"
