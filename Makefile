UV_CACHE_DIR ?= .uv-cache

.PHONY: setup test lint

setup:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync

test:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run pytest -q

lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -m compileall -q .
