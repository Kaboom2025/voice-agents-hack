#!/usr/bin/env bash
# Start a local ChromaDB server using uv.
# Data is persisted in .chroma/ relative to the workspace root.
set -euo pipefail

cd "$(dirname "$0")/.."

CHROMA_DIR="${CHROMA_DIR:-.chroma}"
PORT="${CHROMA_PORT:-8000}"

echo "Starting ChromaDB on port ${PORT}, data in ${CHROMA_DIR}/"
exec uvx --from chromadb chroma run --path "${CHROMA_DIR}" --port "${PORT}"
