#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_WORKERS="${MLFLOW_WORKERS:-1}"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///${REPO_ROOT}/mlflow.db}"
MLFLOW_DEFAULT_ARTIFACT_ROOT="${MLFLOW_DEFAULT_ARTIFACT_ROOT:-${REPO_ROOT}/mlruns}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"

mkdir -p "${MLFLOW_DEFAULT_ARTIFACT_ROOT}"

echo "Starting MLflow server on ${MLFLOW_HOST}:${MLFLOW_PORT}"
echo "Backend store URI: ${MLFLOW_BACKEND_STORE_URI}"
echo "Default artifact root: ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
echo "UV cache dir: ${UV_CACHE_DIR}"

exec env UV_CACHE_DIR="${UV_CACHE_DIR}" uv run mlflow server \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --workers "${MLFLOW_WORKERS}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" \
  "$@"
