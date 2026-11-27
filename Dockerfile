# syntax=docker/dockerfile:1

########## Stage 1: builder ##########
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time only: compiler toolchain for any dependency lacking a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Isolated virtualenv, copied wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what pip needs first to maximise layer-cache reuse.
COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install .

########## Stage 2: runtime ##########
FROM python:3.11-slim AS runtime

# UID/GID default to 1000 so bind-mounted ./reports is writable for the
# default WSL2 user. Override with --build-arg if your host UID differs.
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    LOG_LEVEL=INFO \
    MODEL_PATH=models/model.pkl \
    LOCAL_MODEL_PATH=models/model.pkl \
    LOCAL_FEATURE_NAMES_PATH=models/feature_names.json \
    DECISION_THRESHOLD=0.5 \
    UVICORN_WORKERS=2 \
    UVICORN_LOG_LEVEL=info \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# libgomp1: OpenMP runtime required by LightGBM/XGBoost at import time.
# curl: used by HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root system user.
RUN groupadd --system --gid ${APP_GID} appuser && \
    useradd --system --uid ${APP_UID} --gid appuser --home-dir /app --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app

# Mount points; models/ is expected to be bind-mounted read-only at runtime.
RUN mkdir -p /app/models /app/reports && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Shell form so ${VAR:-default} expands; `exec` makes uvicorn PID 1 so it
# receives SIGTERM directly and shuts down gracefully.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2} --log-level ${UVICORN_LOG_LEVEL:-info} --no-access-log