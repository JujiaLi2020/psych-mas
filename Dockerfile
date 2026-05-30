# Psych-MAS: FastAPI backend (R/rpy2) + optional Streamlit UI
#
# Recommended local deploy:
#   docker compose up --build
#
# Manual:
#   docker build -t psych-mas .
#   docker compose up backend          # API on :8000
#   docker run -p 8501:8501 -e PORT=8501 -e PSYMAS_BACKEND_URL=http://host.docker.internal:8000 psych-mas sh scripts/run_ui_railway.sh

# rpy2 + embedded R is much more stable on Python 3.11 than 3.13.
FROM python:3.11-slim-bookworm

WORKDIR /app

# System packages (R + build deps for R packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    build-essential \
    gfortran \
    libuv1-dev \
    cmake \
    libmbedtls-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy app files needed for R and Python install
COPY packages.txt install_r_packages.R r_packages.txt ./
COPY pyproject.toml README.md ./
COPY graph.py ui.py main.py mmls.py backend_service.py ./

# Install R packages (mirt, WrightMap, psych) - can take several minutes
RUN Rscript install_r_packages.R

# Python dependencies
RUN pip install --no-cache-dir . gunicorn uvicorn


# Streamlit config + Railway UI launcher (backend image may still be used for a UI service with overridden CMD)
COPY .streamlit .streamlit
COPY scripts/run_ui_railway.sh scripts/run_ui_railway.sh
# Windows checkouts often use CRLF; strip before running in Linux.
RUN sed -i 's/\r$//' scripts/run_ui_railway.sh && chmod +x scripts/run_ui_railway.sh

EXPOSE 8000 8501

# Suppress rpy2 "R is not initialized by the main thread" warning (harmless in cloud)
ENV PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"
# Avoid R JIT initialization issues in some container environments.
ENV R_ENABLE_JIT=0
# Shared detect job files so multiple Gunicorn workers in ONE container see the same run_id (no Redis required).
# For multiple Railway replicas, add Redis — job_store priority is Redis > this directory > in-memory.
ENV PSYMAS_JOB_DIR=/tmp/psymas_detect_jobs

# Railway sets PORT; Streamlit must listen on 0.0.0.0 for external access
#CMD ["sh", "-c", "streamlit run ui.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
#
# Without REDIS_URL, job state is per-process — keep PSYMAS_GUNICORN_WORKERS at 1 (default).
# Add Redis (plugin) + REDIS_URL, then raise PSYMAS_GUNICORN_WORKERS (e.g. 4–8) to spread HTTP load.
CMD ["sh", "-c", "gunicorn backend_service:app --workers ${PSYMAS_GUNICORN_WORKERS:-1} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120"]

