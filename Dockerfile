# ─────────────────────────────────────────────────
# Stage 1: Builder — install deps and build wheel
# ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for asyncpg compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir --prefix=/install .

# ─────────────────────────────────────────────────
# Stage 2: Runtime — minimal image
# ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="aumos-human-ai-collab"
LABEL org.opencontainers.image.description="Confidence-based task routing between AI and humans"
LABEL org.opencontainers.image.vendor="AumOS Enterprise"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user for security
RUN groupadd --gid 1001 aumos && \
    useradd --uid 1001 --gid aumos --no-create-home aumos

USER aumos

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "aumos_human_ai_collab.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
