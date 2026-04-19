<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 60%;">

# Multi-Camera Video RAG

A distributed multi-camera Video RAG system. Each **follower** captures webcam frames, produces multimodal embeddings on-device (Gemma 4 via Cactus) or via Gemini, and ships them over [iroh](https://www.iroh.computer/) QUIC to a central **leader**. The leader indexes the chunks, answers natural-language queries against them, and serves a React UI for live camera views + chat.

See [docs/PRD-multicam-video-rag.md](docs/PRD-multicam-video-rag.md) for the full design and [IROH.md](IROH.md) for the wire protocol.

## Architecture

```
follower(s) ──iroh QUIC──▶ leader ──HTTP──▶ ui (Vite dev server)
  webcam+embed             :8080           :5173
```

- [common/](common) — wire types, framing, ALPN (`cactus/ingest/v1`).
- [follower/](follower) — webcam → Gemini or Cactus/Gemma embeddings → push chunks.
- [leader/](leader) — accepts ingest, exposes `/api/cameras`, `/api/live/:id`, `/api/query`.
- [ui/](ui) — React + Vite frontend that proxies `/api` to the leader.

## Prerequisites

- macOS (Apple Silicon recommended) or Linux.
- Rust stable (`rustup default stable`).
- Node.js ≥ 18 and `npm`.
- A webcam (or use `--no-camera` on the follower).
- One of the following for real embeddings / completions:
  - **Gemini path (easiest):** a `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/api-keys).
  - **Cactus path (on-device Gemma 4):** `libcactus.dylib` installed and Gemma 4 weights on disk. See [Optional: Cactus / Gemma 4 on-device](#optional-cactus--gemma-4-on-device) below.

## 1. Clone and configure

```sh
git clone <this-repo>
cd voice-agents-hack
cp .env.example .env
```

Edit [.env](.env.example) (all values are optional — defaults work):

```sh
LEADER_KEY_FILE=.leader.key         # leader's persistent iroh identity
LEADER_TICKET_FILE=.leader.ticket   # leader writes a dial-ticket here on startup
# GEMINI_API_KEY=...                                 # add for the Gemini embedder path
# GEMMA_MODEL_PATH=/abs/path/to/gemma-4-e2b-it       # only used with --features cactus
```

## 2. Run the leader

In terminal A:

```sh
# Gemini / synthetic path (no native deps):
cargo run -p leader --release

# Cactus / on-device Gemma 4 path:
cargo run -p leader --release --features cactus
```

On startup the leader:

- binds an iroh endpoint and waits for a relay URL,
- writes a dial-ticket to `.leader.ticket` (override with `LEADER_TICKET_FILE`),
- prints the ticket to stdout,
- serves HTTP on `http://127.0.0.1:8080` (override with `--http-addr` / `LEADER_HTTP_ADDR`).

Leave it running. `Ctrl-C` removes the ticket file cleanly.

## 3. Run one or more followers

In terminal B (repeat in more terminals for more cameras, each with a unique `--camera-id`):

```sh
# Uses .leader.ticket automatically when leader is on the same machine.
# Set GEMINI_API_KEY in .env (or pass --gemini-api-key) for real embeddings;
# otherwise the follower falls back to synthetic vectors.
cargo run -p follower --release -- --camera-id cam-local

# Running across machines: paste the ticket the leader printed.
cargo run -p follower --release -- <TICKET> --camera-id cam-remote

# Headless / no webcam:
cargo run -p follower --release -- --camera-id cam-still --no-camera

# Force synthetic embeddings (no Gemini calls):
cargo run -p follower --release -- --camera-id cam-fake --synthetic

# On-device Gemma 4 via Cactus:
cargo run -p follower --release --features cactus -- --camera-id cam-oncam
```

Useful follower flags:

| flag | default | purpose |
| --- | --- | --- |
| `--camera-id <id>` | `cam-0` | logical id reported to the leader (unique per follower) |
| `--step-ms <ms>` | `5000` | how often to emit an embedding |
| `--window-ms <ms>` | `10000` | sliding video window size |
| `--device-index <n>` | `0` | OS camera index |
| `--no-camera` | off | use a placeholder frame instead of a webcam |
| `--synthetic` | off | skip Gemini/Cactus and send random vectors |
| `--count <n>` | `0` | stop after N chunks; `0` = forever |
| `--frame-dir <dir>` | `./frames` | where captured JPEGs are saved |

The leader logs every chunk it receives and acks by id.

## 4. Run the UI

In terminal C:

```sh
cd ui
npm install        # first time only
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to `http://127.0.0.1:8080` (see [ui/vite.config.ts](ui/vite.config.ts)) so the UI talks to the leader transparently. You'll see each connected follower as a camera tile with live snapshots and can chat against the indexed embeddings.

For a production build:

```sh
cd ui
npm run build      # output in ui/dist
npm run preview    # serve the built assets locally
```

## Optional: Cactus / Gemma 4 on-device

The `cactus` Cargo feature on both crates links `libcactus.dylib` so the follower embeds with Gemma 4 locally and the leader answers queries with local chat completion. Without this feature the project still builds and runs end-to-end using Gemini + synthetic fallbacks.

One-time Cactus setup (from [cactuscompute.com](https://cactuscompute.com/)):

```sh
git clone https://github.com/cactus-compute/cactus
cd cactus && source ./setup && cd ..
cactus build --python
cactus download google/gemma-4-e2b-it --reconvert
cactus auth        # paste your key from https://cactuscompute.com/dashboard/api-keys
```

Then point the leader at your weights and build with the feature:

```sh
export GEMMA_MODEL_PATH=/absolute/path/to/gemma-4-e2b-it
cargo run -p leader   --release --features cactus
cargo run -p follower --release --features cactus -- --camera-id cam-oncam
```

`libcactus.dylib` is discovered via (in order): `$CACTUS_LIB_DIR`, `$CACTUS_PREFIX/lib`, `$HOME/cactus/cactus/build`, `/opt/homebrew/opt/cactus/lib`.

A minimal smoke test (no iroh, no UI) confirms the Cactus install:

```sh
cargo run -p follower --example cactus_smoke --release --features cactus -- \
    --model-path "$GEMMA_MODEL_PATH"
```

## Troubleshooting

- **Follower exits with `no ticket given and ticket file not readable`** — start the leader first, or pass the ticket string as a positional argument.
- **`ticket is empty`** — the leader isn't fully up yet; it writes the ticket only after a relay URL is established.
- **Follower logs `embed_window failed`** — no `GEMINI_API_KEY` and `--synthetic` wasn't passed, or the key is rejected. Set the key or pass `--synthetic`.
- **Camera won't open on macOS** — grant Terminal (or your IDE) camera permission in System Settings → Privacy & Security → Camera, then rerun. Or use `--no-camera`.
- **UI shows no cameras** — confirm the leader is on `127.0.0.1:8080` (`curl http://127.0.0.1:8080/api/cameras`) and at least one follower has connected.
- **Linker errors with `--features cactus`** — `libcactus.dylib` isn't where the build script looks. Set `CACTUS_LIB_DIR=/dir/containing/dylib`.

## Layout

```
common/      wire types + framing (postcard over iroh QUIC)
follower/    capture + embed + push binary
leader/      ingest + HTTP API + (optional) Cactus chat binary
ui/          React + Vite frontend
docs/        PRD + diagrams
recordings/  sample segmented video (HLS-style .ts chunks)
weights/     Gemma 4 weights (populate via `cactus download`)
```
<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 60%;">

## Context
- Cactus (YC S25) is a low-latency engine for mobile devices & wearables. 
- Cactus runs locally on edge devices with hybrid routing of complex tasks to cloud models like Gemini.
- Google DeepMind just released Gemma 4, the first on-device model you can voice-prompt. 
- Gemma 4 on Cactus is multimodal, supporting voice, vision, function calling, transcription and more! 

## Challenge
- All teams MUST build products that use Gemma 4 on Cactus. 
- All products MUST leverage voice functionality in some way. 
- All submissions MUST be working MVPs capable of venture backing. 
- Winner takes all: Guaranteed YC Interview + GCP Credits. 

## Special Tracks 
- Best On-Device Enterprise Agent (B2B): Highest commercial viability for offline tools.
- Ultimate Consumer Voice Experience (B2C): Best use of low-latency compute to create ultra-natural, instantaneous voice interaction.
- Deepest Technical Integration: Pushing the boundaries of the hardware/software stack (e.g., novel routing, multi-agent on-device setups, extreme power optimization).

Prizes per special track: 
- 1st Place: $2,000 in GCP credits
- 2nd Place: $1,000 in GCP credits 
- 3rd Place: $500 in GCP credits 

## Judging 
- **Rubric 1**: The relevnance and realness of the problem and appeal to enterprises and VCs. 
- **Rubric 2**: Correcness & quality of the MVP and demo. 

## Setup (clone this repo and hollistically follow)
- Step 1: Fork this repo, clone to your Mac, open terminal.
- Step 2: `git clone https://github.com/cactus-compute/cactus`
- Step 3: `cd cactus && source ./setup && cd ..` (re-run in new terminal)
- Step 4: `cactus build --python`
- Step 5: `cactus download google/functiongemma-270m-it --reconvert`
- Step 6: Get cactus key from the [cactus website](https://cactuscompute.com/dashboard/api-keys)
- Sept 7: Run `cactus auth` and enter your token when prompted.
- Step 8: `pip install google-genai` (if using cloud fallback) 
- Step 9: Obtain Gemini API key from [Google AI Studio](https://aistudio.google.com/api-keys) (if using cloud fallback) 
- Step 10: `export GEMINI_API_KEY="your-key"` (if using cloud fallback) 

## Next steps
1. Read Cactus docs carefully: [Link](https://docs.cactuscompute.com/latest/)
2. Read Gemma 4 on Cactus walkthrough carefully: [Link](https://docs.cactuscompute.com/latest/blog/gemma4/)
3. Cactus & DeepMind team would be available on-site. 