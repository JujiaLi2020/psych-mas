#!/usr/bin/env sh
set -eu

PORT="${PORT:-8501}"

exec python -m streamlit run ui.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT}" \
  --server.headless=true
