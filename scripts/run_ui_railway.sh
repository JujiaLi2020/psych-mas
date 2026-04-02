#!/usr/bin/env sh
# Streamlit UI entrypoint for Railway and similar HTTPS-terminated proxies.
# In Railway → UI service → Start Command:
#   sh scripts/run_ui_railway.sh
# Optional Variables (recommended if uploads still return 400):
#   STREAMLIT_BROWSER_SERVER_ADDRESS = your-ui.up.railway.app
#   STREAMLIT_BROWSER_SERVER_PORT    = 443
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
