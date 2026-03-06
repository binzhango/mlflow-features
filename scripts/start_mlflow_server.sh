#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_WORK_DIR="${MLFLOW_WORK_DIR:-$ROOT_DIR/.mlflow}"
MLFLOW_WORKERS="${MLFLOW_WORKERS:-1}"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///$MLFLOW_WORK_DIR/mlflow.db}"
MLFLOW_ARTIFACTS_DIR="${MLFLOW_ARTIFACTS_DIR:-$MLFLOW_WORK_DIR/artifacts}"

mkdir -p "$MLFLOW_WORK_DIR" "$MLFLOW_ARTIFACTS_DIR"

echo "Starting MLflow server"
echo "  host: $MLFLOW_HOST"
echo "  port: $MLFLOW_PORT"
echo "  backend: $MLFLOW_BACKEND_STORE_URI"
echo "  artifacts: $MLFLOW_ARTIFACTS_DIR"

exec mlflow server \
  --host "$MLFLOW_HOST" \
  --port "$MLFLOW_PORT" \
  --workers "$MLFLOW_WORKERS" \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --artifacts-destination "$MLFLOW_ARTIFACTS_DIR" \
  --serve-artifacts
