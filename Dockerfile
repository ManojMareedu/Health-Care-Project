# Single image serving both surfaces. Which one runs is chosen by the compose
# command, so the API and dashboard can never drift onto different model
# versions -- they are built from the same layer.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by XGBoost's runtime; the slim base does not ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer caches across source edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY exported_model/ ./exported_model/

RUN chmod +x scripts/start_services.sh

# Run as a non-root user. The container only ever reads its model artifacts.
# Hugging Face Spaces requires uid 1000, which this matches.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV HOME=/home/appuser \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 8000 API, 8501 dashboard under compose, 7860 the single port Spaces exposes.
EXPOSE 8000 8501 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default runs both services on one port, which is what single-port hosts like
# Hugging Face Spaces need. docker-compose overrides this per service.
CMD ["bash", "scripts/start_services.sh"]
