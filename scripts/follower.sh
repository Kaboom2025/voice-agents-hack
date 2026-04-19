#!/usr/bin/env bash
# Start a follower with local Cactus/Gemma 4 embeddings.
# Usage: ./scripts/follower.sh [CAMERA_ID]
set -euo pipefail

cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

CAMERA_ID="${1:-cam-local}"

# HLS recording output. Must match the leader's --recordings-dir (defaults
# match, so override via env to keep both scripts aligned).
RECORDINGS_DIR="${RECORDINGS_DIR:-./recordings}"
RECORD_FPS="${RECORD_FPS:-15}"
RECORD_SEGMENT_SECS="${RECORD_SEGMENT_SECS:-4}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg not found on PATH — HLS recording will be disabled."
    RECORD_FLAGS=(--no-record)
else
    mkdir -p "$RECORDINGS_DIR"
    RECORD_FLAGS=(
        --recordings-dir "$RECORDINGS_DIR"
        --record-fps "$RECORD_FPS"
        --record-segment-secs "$RECORD_SEGMENT_SECS"
    )
fi

echo "Building follower (cactus)..."
cargo build --release -p follower --features cactus 2>&1 | tail -3

echo "Starting follower: ${CAMERA_ID}"
echo "  recordings → ${RECORDINGS_DIR}/${CAMERA_ID}/stream.m3u8"
exec cargo run --release -p follower --features cactus -- \
    --camera-id "$CAMERA_ID" \
    "${RECORD_FLAGS[@]}"
