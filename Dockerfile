# Psych-MAS: Streamlit + LangGraph/R backend for Railway
# Build: docker build -t psych-mas .
# Run UI locally:       docker run -p 8501:8501 -e PORT=8501 psych-mas
# Run backend (example): docker run -p 8000:8000 psych-mas uvicorn backend_service:app --host 0.0.0.0 --port 8000

FROM python:3.13-slim-bookworm

WORKDIR /app

# System packages (R + build deps for R packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy app files needed for R and Python install
COPY packages.txt install_r_packages.R r_packages.txt ./
COPY pyproject.toml README.md ./
COPY graph.py ui.py main.py openrouter_models.py backend_service.py ./

# Install R packages (mirt, WrightMap, psych) - can take several minutes
RUN Rscript install_r_packages.R

# Python dependencies
RUN pip install --no-cache-dir .

# Streamlit config (optional: bind to 0.0.0.0 is set in CMD)
COPY .streamlit .streamlit

EXPOSE 8501

# Suppress rpy2 "R is not initialized by the main thread" warning (harmless in cloud)
ENV PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"

# Railway sets PORT; Streamlit must listen on 0.0.0.0 for external access
CMD ["sh", "-c", "streamlit run ui.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
