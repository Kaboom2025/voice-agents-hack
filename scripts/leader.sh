#!/usr/bin/env bash
# Run the full leader stack: ChromaDB + Rust leader + UI dev server.
# Ctrl-C stops all three.
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Config (override via env or .env) ─────────────────────────────────
source .env 2>/dev/null || true
CHROMA_PORT="${CHROMA_PORT:-8000}"
LEADER_HTTP="${LEADER_HTTP_ADDR:-127.0.0.1:8080}"

# ── Colors ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Cleanup on exit ──────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo -e "${GREEN}All stopped.${NC}"
}
trap cleanup EXIT INT TERM

# ── 1. Build leader + follower (release) ─────────────────────────────
echo -e "${CYAN}Building leader (cactus)...${NC}"
cargo build --release -p leader --features cactus 2>&1 | tail -5

# ── 2. Start ChromaDB ────────────────────────────────────────────────
echo -e "${CYAN}Starting ChromaDB on port ${CHROMA_PORT}...${NC}"
uvx --from chromadb chroma run --path .chroma --port "$CHROMA_PORT" \
    > /dev/null 2>&1 &
PIDS+=($!)

# Wait for ChromaDB to be ready (v2 API)
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${CHROMA_PORT}/api/v2/heartbeat" > /dev/null 2>&1; then
        echo -e "${GREEN}ChromaDB ready${NC}"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo -e "${RED}ChromaDB failed to start after 30s${NC}"
        exit 1
    fi
    sleep 1
done

# ── 3. Start the leader ──────────────────────────────────────────────
echo -e "${CYAN}Starting leader on ${LEADER_HTTP}...${NC}"
cargo run --release -p leader --features cactus &
PIDS+=($!)
sleep 2

# ── 4. Start the UI dev server ───────────────────────────────────────
echo -e "${CYAN}Starting UI (Vite + bun)...${NC}"
cd ui
bun install --frozen-lockfile 2>/dev/null || bun install
bun run dev &
PIDS+=($!)
cd ..

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Leader stack running:${NC}"
echo -e "    ChromaDB:  http://localhost:${CHROMA_PORT}"
echo -e "    Leader:    http://${LEADER_HTTP}"
echo -e "    UI:        http://localhost:5173"
echo -e ""
echo -e "  ${YELLOW}Connect a follower:${NC}"
echo -e "    cargo run --release -p follower --features cactus -- --camera-id cam-local"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo ""

# Wait for any child to exit
wait
