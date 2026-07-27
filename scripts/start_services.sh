#!/usr/bin/env bash
# Start the API and the dashboard in one container.
#
# Hugging Face Spaces exposes exactly one port, so the dashboard takes the
# public port and talks to the API on loopback inside the same container. This
# is only for single-port hosts -- docker-compose runs the two as separate
# services, which is the better shape when the platform allows it.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${PORT:-7860}"

uvicorn app.api.main:app --host 127.0.0.1 --port "${API_PORT}" &
API_PID=$!

# Stop the whole container if the API dies, rather than leaving a dashboard up
# that errors on every prediction.
trap 'kill -TERM ${API_PID} 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 40); do
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" > /dev/null 2>&1; then
        echo "API ready on ${API_PORT}"
        break
    fi
    sleep 2
done

exec streamlit run app/dashboard/app.py \
    --server.port "${DASHBOARD_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true
