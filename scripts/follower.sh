#!/usr/bin/env bash
# Start a follower with local Cactus/Gemma 4 embeddings.
# Usage: ./scripts/follower.sh [CAMERA_ID]
set -euo pipefail

cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

CAMERA_ID="${1:-cam-local}"

echo "Building follower (cactus)..."
cargo build --release -p follower --features cactus 2>&1 | tail -3

echo "Starting follower: ${CAMERA_ID}"
exec cargo run --release -p follower --features cactus -- --camera-id "$CAMERA_ID"
