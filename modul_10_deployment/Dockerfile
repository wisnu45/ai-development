# ============================================================
# DOCKERFILE BNSP - Churn Prediction AI Service (FastAPI)
# ============================================================
# [1] BASE IMAGE: Python 3.10 Slim (ringan, secure, cocok untuk production)
FROM python:3.10-slim-bookworm AS base

# [2] ENVIRONMENT production (hindari .pyc + buffer log)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

# [3] SET WORKDIR
WORKDIR ${APP_HOME}

# [4] Install system dependencies (curl untuk healthcheck, build-essential untuk numpy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# [5] Copy requirements DULU (Docker layer caching!) - install dulu
COPY requirements.txt .
RUN pip install --upgrade pip==24.0 && \
    pip install --no-cache-dir -r requirements.txt

# [6] Copy SELURUH source code (termasuk folder saved_models/)
COPY . .

# [7] (Opsional) Buat user non-root untuk keamanan production
RUN useradd -m appuser && chown -R appuser:appuser ${APP_HOME}
USER appuser

# [8] EXPOSE port yang dipakai FastAPI (uvicorn 8000)
EXPOSE 8000

# [9] HEALTHCHECK = Docker monitor apakah service sehat (restart otomatis jika mati)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# [10] DEFAULT COMMAND jalankan uvicorn server (production: 1 worker per CPU core)
CMD ["uvicorn", "fastapi_churn_app_TEMPLATE:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
