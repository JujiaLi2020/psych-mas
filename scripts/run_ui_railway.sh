#!/usr/bin/env sh
# Streamlit UI entrypoint for Railway, Docker Compose, and similar proxies.
# Docker Compose / Railway UI service start command:
#   sh scripts/run_ui_railway.sh
set -e
PORT="${PORT:-8501}"
export STREAMLIT_SERVER_HEADLESS="${STREAMLIT_SERVER_HEADLESS:-true}"
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="${STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION:-false}"
export STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION="${STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION:-false}"
export STREAMLIT_SERVER_ENABLE_CORS="${STREAMLIT_SERVER_ENABLE_CORS:-false}"

exec streamlit run ui.py \
  --server.port="${PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableXsrfProtection=false \
  --server.enableWebsocketCompression=false \
  --server.enableCORS=false
