# Psych-MAS: Streamlit + LangGraph/R backend for Railway
# Build: docker build -t psych-mas .

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

# 核心更新 1：强制在此处一并安装 gunicorn 和 uvicorn，彻底消除找不到可执行文件的错误
RUN pip install --no-cache-dir . gunicorn uvicorn

# Streamlit config
COPY .streamlit .streamlit

# Suppress rpy2 warning
ENV PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"
ENV R_ENABLE_JIT=0

# 核心更新 2：将默认启动命令修改为我们经过内存压测的高并发后端配置
# 这样即便你在 Railway 面板上忘记填写 Custom Start Command，它也能安全启动 8 个进程
CMD ["sh", "-c", "gunicorn backend_service:app --workers 8 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120"]