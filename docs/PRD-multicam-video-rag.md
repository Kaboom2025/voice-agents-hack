# PRD: Multi-Camera Video RAG System

**Status:** Draft v0.1
**Owner:** TBD
**Stack anchor:** Cactus engine (on-device inference) + Gemma 3 4B multimodal for embedding, ChromaDB for vector storage, **Rust + [iroh](https://www.iroh.computer/) for all node-to-node networking** (peer discovery, authenticated QUIC transport, NAT traversal, and resumable bulk transfer of raw clips).

---

## 1. Problem & Vision

Surveillance, robotics, and "smart space" deployments increasingly involve **multiple concurrent video feeds** (e.g. several rooms, several robots, several cameras on a vehicle). Operators need to ask natural-language questions across **all** of those feeds:

- *"When did the package get delivered to the front door today?"*
- *"Which camera last saw the red backpack?"*
- *"Did anyone enter the lab between 3pm and 5pm?"*
- *"Summarize what happened on camera 2 in the last hour."*

Current solutions either (a) require sending raw video to the cloud (expensive, privacy-hostile, bandwidth-bound), or (b) are single-camera only.

**We will build a distributed multi-camera Video RAG system** where each camera node runs **on-device multimodal embedding** via Cactus + Gemma 3 4B, ships only embeddings + lightweight metadata to a central client server, and lets users issue natural-language queries over the combined index.

---

## 2. Goals & Non-Goals

### 2.1 Goals (v1)

- Support **N ≥ 4** concurrent camera/microphone follower nodes per client server.
- Each follower performs **on-device** video frame + audio frame embedding using Cactus + Gemma 3 4B (no raw media leaves the follower by default).
- Embeddings are streamed to the client server and persisted in **ChromaDB**.
- Users can run **natural-language queries** that span any subset of cameras and any time range, and receive ranked clips/timestamps/snippets with source camera attribution.
- End-to-end embedding-to-queryable latency target: **< 60 seconds** (soft real-time).
- Run on commodity hardware: follower = Apple Silicon Mac mini / Jetson-class / mid-range ARM SBC; client = single Linux box.

### 2.2 Non-Goals (v1)

- Live alerting / streaming triggers ("notify me when X happens"). Defer to v2.
- Cloud-hosted multi-tenant SaaS. v1 is single-tenant, self-hosted.
- Re-identification / face recognition / person tracking across cameras as a first-class feature.
- Mobile/phone follower client. v1 followers are stationary edge boxes.
- Sub-second live monitoring lag.

---

## 3. Users & Use Cases


| Persona                      | Use case                                                                                |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| Home/small-business operator | Ask "what happened today" across 4–8 cameras without cloud upload.                      |
| Lab / robotics researcher    | Query multi-robot video logs to find specific events for debugging.                     |
| Security analyst             | Time-bounded multi-feed forensic search ("show me everyone who entered between 2–4pm"). |


**Primary user flow:**

1. Operator deploys 1 client server + N follower nodes (each with camera + mic).
2. Followers continuously record, chunk, embed, and ship embeddings to the client.
3. Operator opens the client UI/CLI, types a query, gets ranked results with thumbnails + camera ID + timestamp + jump-to-clip.

---

## 4. System Architecture

```
┌───────────────────────────┐       ┌───────────────────────────┐
│  Follower Node 1          │       │  Follower Node N          │
│  ┌─────────────────────┐  │       │  ┌─────────────────────┐  │
│  │ Camera + Mic capture│  │  ...  │  │ Camera + Mic capture│  │
│  └─────────┬───────────┘  │       │  └─────────┬───────────┘  │
│            ▼              │       │            ▼              │
│  ┌─────────────────────┐  │       │  ┌─────────────────────┐  │
│  │ Chunker (video+audio)│  │      │  │ Chunker             │  │
│  └─────────┬───────────┘  │       │  └─────────┬───────────┘  │
│            ▼              │       │            ▼              │
│  ┌─────────────────────┐  │       │  ┌─────────────────────┐  │
│  │ Cactus + Gemma 3 4B │  │       │  │ Cactus + Gemma 3 4B │  │
│  │  → embeddings       │  │       │  │  → embeddings       │  │
│  └─────────┬───────────┘  │       │  └─────────┬───────────┘  │
│            ▼              │       │            ▼              │
│  ┌─────────────────────┐  │       │  ┌─────────────────────┐  │
│  │ Local raw clip cache│  │       │  │ Local raw clip cache│  │
│  │ (rolling, optional) │  │       │  │                     │  │
│  └─────────┬───────────┘  │       │  └─────────┬───────────┘  │
│            ▼              │       │            ▼              │
│   iroh QUIC stream push   │       │   iroh QUIC stream push   │
└────────────┬──────────────┘       └────────────┬──────────────┘
             │                                   │
             └───────────────┬───────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │     Client Server             │
              │  ┌─────────────────────────┐ │
              │  │ iroh Endpoint (QUIC)    │ │
              │  │  + ALPN: cactus/ingest  │ │
              │  │  + ALPN: cactus/clip    │ │
              │  └────────────┬────────────┘ │
              │               ▼              │
              │  ┌─────────────────────────┐ │
              │  │ ChromaDB (vector store) │ │
              │  │  + SQLite (metadata)    │ │
              │  └────────────┬────────────┘ │
              │               ▼              │
              │  ┌─────────────────────────┐ │
              │  │ Query/RAG service       │ │
              │  │ (Cactus + Gemma for LLM │ │
              │  │  reasoning over hits)   │ │
              │  └────────────┬────────────┘ │
              │               ▼              │
              │  ┌─────────────────────────┐ │
              │  │ Web UI / CLI            │ │
              │  └─────────────────────────┘ │
              └──────────────────────────────┘
```

---

## 5. Component Specs

### 5.1 Follower Node

**Responsibilities**

- Capture video (default 1080p @ 15 fps) and audio (16 kHz mono) from local devices.
- Chunk the stream into fixed windows (default **5 s clips**, configurable 2–15 s).
- For each chunk:
  - Sample K frames (default K=4, evenly spaced).
  - Pass frames + audio segment to Gemma 3 4B via Cactus to produce a **single multimodal embedding vector** per chunk.
  - Optionally generate a short **caption** ("two people walking past a doorway") via the same model — used as a sparse text channel for hybrid retrieval.
- Push `{embedding, caption, camera_id, start_ts, end_ts, chunk_uri}` to the client server.
- Maintain a **rolling local raw-clip cache** (default 24h, FIFO) so the client can fetch the actual video on demand for playback.

**Interfaces (all over iroh QUIC, identified by ALPN)**

- `cactus/ingest/v1` — long-lived bidirectional stream, follower → client. Length-prefixed CBOR `EmbeddingChunk` frames pushed as produced; client acks chunk_ids for at-least-once delivery + dedupe.
- `cactus/control/v1` — bidirectional, client → follower. RPC-style: `GetConfig`, `SetConfig`, `Ping`, `RotateKeys`.
- `cactus/clip/v1` — on-demand, client → follower. Request `chunk_id` → follower streams the raw MP4 segment back. Backed by **iroh-blobs** so transfers are resumable, content-addressed (BLAKE3), and dedup-friendly across followers.
- Discovery: followers publish their `NodeId` via iroh's built-in **pkarr / DNS discovery**, plus a static client-side allowlist of permitted follower `NodeId`s.

**Hardware target**

- Apple Silicon (M-series), or ARM64 Linux with NEON; 8 GB+ RAM.
- Cactus chosen specifically because it runs Gemma 3 4B on ARM CPU efficiently with low RAM (zero-copy mmap).

**Implementation language**

- Rust binary (`cactus-follower`). Cactus is invoked via its C FFI through a thin `cactus-sys` crate (autogenerated bindings against `cactus/ffi`). Audio/video capture via `nokhwa` + `cpal`, encoding via `ffmpeg-next` or `gstreamer-rs`.

**Failure handling**

- If client unreachable: iroh's QUIC stream errors trigger a switch to **local spool** (bounded ring buffer on disk, default 4 GB). On reconnect, drain spool oldest-first, deduped by `chunk_id`.
- iroh handles NAT traversal (relay fallback), connection migration on network change, and 0-RTT resumption — followers on flaky Wi-Fi or LTE recover transparently.
- If embedding falls behind real-time: drop frame samples first, drop captions second, drop entire chunks last (with metric).

### 5.2 Embedding Strategy (Cactus + Gemma 3 4B)

- Use Gemma 3 4B's **vision tower output** (pre-LM pooled representation) as the embedding for each sampled frame; mean-pool across the K frames in a chunk to get a chunk vector.
- Audio: feed the audio segment through the same multimodal pipeline (Cactus exposes audio inputs per `engine.h`'s `ChatMessage.images / audio` fields). Produce an audio embedding per chunk.
- Final chunk vector = concatenation of `[video_emb || audio_emb]` then L2-normalize. Store dimension separately in metadata so the query side can re-route if the model changes.
- Captions are generated by prompting Gemma with the K frames + audio, asking for a one-sentence description. Captions are stored verbatim and also embedded into a **text-only** collection for hybrid (dense+sparse+text-vector) retrieval.

**Open question:** whether to keep video and audio as separate Chroma collections vs. one fused vector. v1 default = **two collections** (`video_clips`, `audio_clips`) plus a `captions` collection. Query merges across them.

### 5.3 Client Server

**Responsibilities**

- Run a single iroh `Endpoint` accepting connections on the ALPNs above. One Tokio task per accepted bidi stream.
- Validate, dedupe (by `chunk_id`, secondary on `camera_id + start_ts`), and write to:
  - **ChromaDB** collections (`video_clips`, `audio_clips`, `captions`).
  - **SQLite** (`sqlx`) metadata table (chunk_id, camera_id, follower_node_id, start_ts, end_ts, has_raw_clip, caption, blob_hash).
- Expose Query API (see §5.4) over a local HTTP server (`axum`) for the UI/CLI; the UI is the only thing that talks HTTP. Everything camera-facing is iroh.
- Manage follower registry: pin allowlist of follower `NodeId`s; iroh `Ping` over `cactus/control/v1` every 10 s for health; push config updates.

**Implementation language**

- Rust binary (`cactus-client`). Crates: `iroh`, `iroh-blobs`, `tokio`, `axum`, `sqlx`, `chromadb` (HTTP client to a sidecar Chroma instance, or `arroy`/`qdrant-client` if we move off Chroma later).

**Storage estimates (rule-of-thumb)**

- Per chunk: ~ (4 KB embedding) + (200 B caption) + (200 B metadata) ≈ 4.5 KB.
- 8 cameras × 5 s chunks × 24 h ≈ 138 k chunks/day ≈ **~620 MB/day** of embeddings + metadata. Easily fits on a single SSD.
- Raw video stays on followers; only fetched on demand.

### 5.4 Query / RAG Pipeline

**API:** `POST /query`

```json
{
  "query": "when did anyone enter the lab today?",
  "cameras": ["cam-lab-1", "cam-lab-2"],          // optional, default = all
  "time_range": {"from": "2026-04-18T00:00", "to": "2026-04-18T23:59"},
  "top_k": 20,
  "modalities": ["video", "audio", "caption"]      // default all
}
```

**Pipeline**

1. Embed the query with the **same** Gemma/Cactus stack (text-only mode) → query vector(s).
2. Run filtered ANN search in each requested Chroma collection (filter on `camera_id ∈ …`, `start_ts ∈ range`).
3. Reciprocal-rank-fuse results across modalities.
4. Take top-K fused chunks, fetch their captions + metadata.
5. Pass `{query, [captions + camera + timestamp]}` to Gemma (LLM mode via Cactus) to produce a natural-language answer with citations to specific `(camera, timestamp)` pairs.
6. Return answer + ranked clip list. UI fetches raw clips lazily from the originating follower.

**Latency budget (target)**

- Query embed: < 200 ms
- Chroma ANN: < 300 ms (8 cameras, 1 week of data)
- LLM synthesis (4B on client GPU/CPU): < 4 s
- Total p95: **< 5 s**

### 5.5 UI (v1: minimal)

- Web SPA + CLI.
- Query box, camera multi-select, time-range picker.
- Result list: thumbnail, camera, timestamp, caption, "play clip" button (streams from follower), confidence score.
- LLM-generated synthesis at top with inline `[cam-2 @ 14:03]` citations that jump to the clip.

---

## 6. Data Model

```
chunk_id        TEXT PRIMARY KEY        -- uuid
camera_id       TEXT
follower_url    TEXT
start_ts        TIMESTAMP (UTC)
end_ts          TIMESTAMP (UTC)
duration_s      REAL
video_dim       INT
audio_dim       INT
caption         TEXT
has_raw_clip    BOOL
created_at      TIMESTAMP
```

Chroma collections store the embedding + the same `chunk_id` as document ID, with metadata mirror of `{camera_id, start_ts, end_ts}` to enable filtered search.

---

## 7. Security & Privacy

- **Default: raw video never leaves the follower.** Only embeddings + short captions are transmitted; raw MP4 segments are only sent over `cactus/clip/v1` when the operator explicitly requests playback.
- **Transport security comes from iroh:** every connection is QUIC + TLS 1.3 with each peer authenticated by its Ed25519 `NodeId` (no CA infrastructure required). The client maintains an allowlist of follower `NodeId`s; the follower pins the client's `NodeId`. Mutual auth, no shared secrets in flight.
- Pre-shared bootstrap token used **once** at follower install time to register its `NodeId` with the client out-of-band.
- Captions may inadvertently include sensitive content — provide a config flag `disable_captions=true` for high-privacy deployments (search then falls back to embedding-only).
- Local raw-clip cache on followers is encrypted at rest (configurable; off by default for performance).
- No telemetry to cactus-compute servers from this product.

---

## 8. Configuration

Single YAML on each follower:

```yaml
camera_id: cam-front-door
client_node_id: k51qzi5uqu5dh...     # iroh NodeId (Ed25519 pubkey, z-base-32)
# Optional: explicit relay/discovery overrides; otherwise use iroh defaults.
iroh:
  relay_mode: default                 # default | disabled | custom
  discovery: [pkarr, dns, local]
chunk_seconds: 5
frames_per_chunk: 4
fps: 15
resolution: 1080p
audio_sample_rate: 16000
generate_captions: true
raw_clip_retention_hours: 24
spool_max_bytes: 4_000_000_000
model:
  backend: cactus
  weights: gemma-3-4b-multimodal-q4_0
```

The follower's own `NodeId` is generated on first run and stored in `~/.cactus-follower/secret.key` (chmod 600). It is printed once for the operator to paste into the client's allowlist.

---

## 9. Metrics & Observability

Per follower:

- `embedding_lag_seconds` (now − latest committed chunk end_ts)
- `dropped_frames_total`, `dropped_chunks_total`
- `embed_duration_ms` p50/p95
- `spool_queue_bytes`
- `iroh_conn_state` (direct | relayed | disconnected), `iroh_rtt_ms`, `iroh_relay_url`

Per client:

- `ingest_chunks_per_minute{camera}`
- `query_latency_ms` p50/p95
- `chroma_collection_size`
- `follower_health{node_id, camera_id}` (up/down, last_seen)
- `iroh_active_connections`, `iroh_holepunch_success_ratio`

---

## 10. Milestones


| Milestone | Scope                                                                                                                                                                                  |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M0        | Rust skeleton: `cactus-follower` + `cactus-client` exchanging dummy embeddings over iroh `cactus/ingest/v1`. Single follower → client → Chroma → CLI text query against captions only. |
| M1        | Real Gemma/Cactus video+audio embeddings via `cactus-sys` FFI. Multi-follower (≥4). On-demand raw-clip fetch via `iroh-blobs` + `cactus/clip/v1`. Web UI with playback.                |
| M2        | Hybrid retrieval (dense+caption+audio) with reciprocal rank fusion + LLM synthesis with citations.                                                                                     |
| M3        | Hardening: NodeId allowlist UX, follower spool/backfill, encrypted clip cache, metrics dashboard, iroh relay self-hosting option.                                                      |
| M4 (v2)   | Live alerts ("notify when X happens"), cross-camera entity linking, mobile follower app.                                                                                               |


---

## 10a. Rust Workspace Layout

```
cactus-multicam/
├── Cargo.toml                   # workspace
├── crates/
│   ├── cactus-sys/              # bindgen wrapper around cactus/ffi
│   ├── cactus-embed/            # safe Rust API: frames+audio -> Vec<f32>
│   ├── multicam-proto/          # CBOR/serde types: EmbeddingChunk, ControlMsg
│   ├── multicam-net/            # iroh Endpoint setup, ALPN handlers, blob store
│   ├── cactus-follower/         # binary: capture + embed + push
│   └── cactus-client/           # binary: iroh server + Chroma + axum HTTP API
└── ui/                          # SPA (separate, talks HTTP to cactus-client)
```

Key external crates: `iroh`, `iroh-blobs`, `tokio`, `serde`, `ciborium`, `axum`, `sqlx`, `tracing`, `nokhwa`, `cpal`, `ffmpeg-next` (or `gstreamer`).

---

## 11. Open Questions

1. **Embedding source.** Use Gemma 3 4B's vision-tower pooled output, or run a dedicated CLIP-style head? Vision-tower keeps "one model fits all" but quality for retrieval is unproven; benchmark on M0.
2. **Chunk size.** 5 s default vs adaptive (scene-change detection). Adaptive is better quality but adds complexity; defer.
3. **Single fused vector vs. per-modality collections.** PRD currently picks per-modality; revisit after M2 retrieval-quality eval.
4. **Where does the LLM synthesis run** — on the client server CPU, or shell out to a GPU box / cloud? Default: client CPU via Cactus, cloud-fallback flag available (Cactus already supports cloud handoff).
5. **Backpressure** when one follower massively outpaces another — do we throttle ingest per-camera or globally?
6. **Retention policy** for embeddings (forever? 90d? per-camera?). v1 default: keep forever, document the per-day storage cost.
7. **iroh relay strategy.** Default ships using the public n0 relays. For fully air-gapped deployments we'll need to document self-hosting an `iroh-relay` (or operating in `relay_mode: disabled` + `discovery: [local]` mDNS-only). Decide whether to bundle a relay binary with the client.
8. **Should the UI also be an iroh node** (so a remote operator's laptop can connect to the client server peer-to-peer with no port forwarding) or stay plain HTTPS? Leaning iroh for v2; HTTPS-on-localhost for v1.

---

## 12. Risks


| Risk                                                                     | Mitigation                                                                                                                      |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Gemma 3 4B vision embeddings are not discriminative enough for retrieval | Spike in M0; if poor, swap embedding model (e.g. SigLIP) but keep Gemma for caption + LLM synthesis.                            |
| Followers can't keep up with real-time embedding                         | Reduce frames_per_chunk, increase chunk_seconds, or add a smaller embed-only model path.                                        |
| Network partition causes embedding loss                                  | iroh auto-reconnect + local spool + backfill (bounded by disk).                                                                 |
| iroh hole-punching fails on hostile NATs                                 | Falls back to relay automatically; document expected throughput hit (~10–30 Mbps via public relay) and offer self-hosted relay. |
| `cactus` C FFI ABI changes break `cactus-sys`                            | Pin `cactus` to a specific tag in `Cargo.toml`; CI builds `cactus-sys` against the pinned version.                              |
| Captions leak sensitive info                                             | `disable_captions` flag; per-camera content-class allowlist (v2).                                                               |
| Storage growth on client                                                 | Configurable retention; document per-camera/day cost; downsample old data to coarser chunks (v2).                               |


---

## 13. Success Criteria (v1 Acceptance)

- 4 followers running for 24 h continuously without dropped chunks > 1%.
- Operator can issue 10 representative natural-language queries and 8/10 return the correct camera + within ±10 s of the actual event.
- p95 query latency < 5 s on a 1-week, 4-camera index.
- Zero raw video bytes egress observed at the follower NIC during normal operation (verified by tcpdump).

