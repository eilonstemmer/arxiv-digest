# syntax=docker/dockerfile:1.6
#
# pipeline image — batch worker. Behind `profiles: ["pipeline"]` in
# docker-compose.yml so `docker compose up` doesn't auto-start it.
# Trigger via:
#   docker compose run --rm pipeline python -m src.pipeline weekly
# or once a long-running instance exists:
#   docker exec arxiv-digest-pipeline python -m src.pipeline weekly

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

# Install deps first (cached as long as pyproject.toml + uv.lock don't change)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev

# Install project itself
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Pre-download the MiniLM model so the first weekly run doesn't pay the
# download cost. We disable the HuggingFace progress bars to keep build
# logs clean.
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

FROM python:3.12-slim-bookworm
WORKDIR /app

# Runtime deps only: copy the prepared venv and the source.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
# Pre-downloaded MiniLM cache lives under HF_HOME (~/.cache/huggingface by
# default). Copy it so the runtime container doesn't need to fetch.
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# Default command is the health check — the scheduler triggers real runs
# explicitly (per docker-compose.yml). `health` returns immediately if the
# DB is reachable and the model is loadable.
CMD ["python", "-m", "src.pipeline", "health"]
