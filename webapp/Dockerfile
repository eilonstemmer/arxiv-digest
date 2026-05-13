# syntax=docker/dockerfile:1.6
# Multi-stage build: stage 1 installs deps with uv, stage 2 is the slim runtime.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src/ ./src/
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app
# Install wget for healthcheck (may already be present, --no-install-recommends keeps it tiny)
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
