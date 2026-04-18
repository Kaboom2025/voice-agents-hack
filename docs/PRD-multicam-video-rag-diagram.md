# Multi-Camera Video RAG — Architecture Diagrams

Visual companion to [`PRD-multicam-video-rag.md`](./PRD-multicam-video-rag.md). All diagrams are Mermaid; they render natively on GitHub and in most Markdown previewers.

---

## 1. System Architecture

End-to-end view: N follower nodes embed locally, push only vectors + captions to the client over iroh QUIC, and raw clips stay on the edge until explicitly requested.

```mermaid
flowchart TB
    subgraph F1["Follower Node 1"]
        direction TB
        C1["Camera + Mic Capture<br/>1080p @ 15fps, 16kHz mono"]
        CH1["Chunker<br/>5s windows, K=4 frames"]
        E1["Cactus + Gemma 3 4B<br/>video + audio embedding<br/>+ optional caption"]
        RC1["Rolling Raw Clip Cache<br/>24h FIFO"]
        SP1["Local Spool<br/>4GB ring buffer<br/>(on disconnect)"]
        C1 --> CH1 --> E1
        CH1 --> RC1
        E1 -.->|"on client unreachable"| SP1
    end

    subgraph FN["Follower Node N"]
        direction TB
        CN["Camera + Mic Capture"]
        CHN["Chunker"]
        EN["Cactus + Gemma 3 4B"]
        RCN["Rolling Raw Clip Cache"]
        CN --> CHN --> EN
        CHN --> RCN
    end

    subgraph IROH["iroh QUIC Transport (TLS 1.3, Ed25519 NodeId auth)"]
        ING["ALPN: cactus/ingest/v1<br/>bidi, long-lived"]
        CTL["ALPN: cactus/control/v1<br/>RPC: Ping / GetConfig / SetConfig"]
        CLIP["ALPN: cactus/clip/v1<br/>iroh-blobs, BLAKE3, resumable"]
    end

    subgraph CLIENT["Client Server"]
        direction TB
        EP["iroh Endpoint<br/>accepts ALPNs above"]
        DEDUP["Validate + Dedupe<br/>by chunk_id"]
        subgraph STORE["Storage"]
            CHROMA[("ChromaDB<br/>video_clips<br/>audio_clips<br/>captions")]
            SQL[("SQLite<br/>chunk metadata<br/>follower registry")]
        end
        RAG["Query / RAG Service<br/>embed → ANN → RRF fuse → LLM synth"]
        API["axum HTTP API<br/>POST /query"]
        UI["Web SPA + CLI<br/>results, playback, citations"]
        EP --> DEDUP --> STORE
        STORE --> RAG --> API --> UI
    end

    E1 --> ING
    EN --> ING
    ING --> EP
    EP <--> CTL
    CTL --> E1
    CTL --> EN
    UI -.->|"1. request clip by chunk_id"| CLIP
    CLIP -->|"2. fetch from originating follower"| RC1
    CLIP -->|"2. fetch from originating follower"| RCN
    RC1 ==>|"3. raw MP4 stream<br/>(BLAKE3, resumable)"| CLIP
    RCN ==>|"3. raw MP4 stream"| CLIP
    CLIP ==>|"4. stream to browser<br/>via client server"| UI

    classDef edge fill:#e8f4f8,stroke:#2a7a9a,color:#000
    classDef transport fill:#fff4e6,stroke:#c07000,color:#000
    classDef server fill:#f0e8f8,stroke:#6a2a9a,color:#000
    class F1,FN edge
    class IROH transport
    class CLIENT server
```

---

## 2. Ingest Pipeline (per chunk)

What one follower does with every 5-second window.

```mermaid
sequenceDiagram
    autonumber
    participant Cam as Camera + Mic
    participant Chk as Chunker
    participant Cac as Cactus + Gemma 3 4B
    participant Cache as Raw Clip Cache
    participant Net as iroh QUIC
    participant Srv as Client Server
    participant Chr as ChromaDB
    participant Sql as SQLite

    Cam->>Chk: frames + audio stream
    Chk->>Chk: buffer 5s window
    Chk->>Cac: K=4 sampled frames + audio segment
    Cac->>Cac: vision tower pool → video_emb
    Cac->>Cac: audio path → audio_emb
    Cac->>Cac: (optional) caption prompt → "two people walking..."
    Cac-->>Chk: {video_emb, audio_emb, caption}
    Chk->>Cache: write raw MP4 (24h FIFO)
    Chk->>Net: EmbeddingChunk (CBOR, length-prefixed)
    Net->>Srv: stream over cactus/ingest/v1
    Srv->>Srv: validate + dedupe by chunk_id
    Srv->>Chr: upsert into video_clips, audio_clips, captions
    Srv->>Sql: insert chunk metadata row
    Srv-->>Net: ack(chunk_id)
    Net-->>Chk: delivery confirmed
```

---

## 3. Query / RAG Flow

What happens when the operator types a natural-language question.

```mermaid
sequenceDiagram
    autonumber
    participant User as Operator (UI/CLI)
    participant API as axum /query
    participant Emb as Cactus (text mode)
    participant Chr as ChromaDB
    participant Fuse as RRF Fuser
    participant LLM as Cactus + Gemma (LLM)
    participant Fol as Originating Follower

    User->>API: {query, cameras?, time_range?, top_k, modalities}
    API->>Emb: embed query text
    Emb-->>API: query vector
    par ANN per modality
        API->>Chr: ANN search video_clips (filtered)
        API->>Chr: ANN search audio_clips (filtered)
        API->>Chr: ANN search captions (filtered)
    end
    Chr-->>Fuse: ranked hits per collection
    Fuse->>Fuse: reciprocal rank fusion
    Fuse->>API: top-K fused chunks
    API->>LLM: {query, [captions + camera + timestamp]}
    LLM-->>API: NL answer w/ [cam-2 @ 14:03] citations
    API-->>User: answer + ranked clip list
    User->>Fol: lazy fetch raw clip via cactus/clip/v1 (iroh-blobs)
    Fol-->>User: MP4 segment (resumable, BLAKE3-addressed)
```

---

## 4. Failure & Backpressure

How the follower keeps producing when the network or the CPU gets in the way.

```mermaid
flowchart LR
    Start([chunk produced]) --> Q{client<br/>reachable?}
    Q -->|yes| Push[push via<br/>cactus/ingest/v1]
    Q -->|no| Spool[append to<br/>local spool<br/>≤ 4GB ring]
    Spool --> Wait[wait for<br/>iroh reconnect]
    Wait --> Drain[drain oldest-first<br/>dedupe by chunk_id]
    Drain --> Push
    Push --> Ack{ack<br/>received?}
    Ack -->|yes| Done([done])
    Ack -->|no, retry| Push

    Behind{embedding<br/>behind real-time?} -.-> Drop1[drop frame samples]
    Drop1 -.-> Drop2[drop captions]
    Drop2 -.-> Drop3[drop whole chunk<br/>+ increment metric]

    classDef warn fill:#fff0f0,stroke:#a02020,color:#000
    class Spool,Drop1,Drop2,Drop3 warn
```

---

## 5. Rust Workspace Layout

How the code is organized across crates.

```mermaid
flowchart TB
    subgraph WS["cactus-multicam (workspace)"]
        direction TB
        SYS["cactus-sys<br/>bindgen over cactus/ffi"]
        EMB["cactus-embed<br/>safe Rust API<br/>frames+audio → Vec&lt;f32&gt;"]
        PROTO["multicam-proto<br/>CBOR/serde types<br/>EmbeddingChunk, ControlMsg"]
        NET["multicam-net<br/>iroh Endpoint, ALPN handlers<br/>iroh-blobs store"]
        FOL["cactus-follower (bin)<br/>capture + embed + push"]
        CLI["cactus-client (bin)<br/>iroh server + Chroma<br/>+ axum HTTP"]
        UI["ui/ (SPA)<br/>HTTP → cactus-client"]
    end

    SYS --> EMB
    EMB --> FOL
    PROTO --> NET
    PROTO --> FOL
    PROTO --> CLI
    NET --> FOL
    NET --> CLI
    CLI -.HTTP.-> UI

    classDef lib fill:#eef5ee,stroke:#3a7a3a,color:#000
    classDef bin fill:#f8f0e8,stroke:#a0602a,color:#000
    class SYS,EMB,PROTO,NET lib
    class FOL,CLI,UI bin
```

---

## 6. Milestone Progression

```mermaid
flowchart LR
    M0["M0<br/>Rust skeleton<br/>dummy embeddings<br/>1 follower → CLI"]
    M1["M1<br/>real Gemma embeddings<br/>≥4 followers<br/>clip fetch + web UI"]
    M2["M2<br/>hybrid retrieval<br/>RRF + LLM synth<br/>citations"]
    M3["M3<br/>hardening<br/>allowlist UX, spool,<br/>encryption, metrics"]
    M4["M4 (v2)<br/>live alerts<br/>entity linking<br/>mobile follower"]

    M0 --> M1 --> M2 --> M3 --> M4

    classDef done fill:#e8f5e8,stroke:#2a7a2a,color:#000
    classDef next fill:#fff8e0,stroke:#a07000,color:#000
    classDef future fill:#f0f0f0,stroke:#707070,color:#000
    class M0 done
    class M1,M2,M3 next
    class M4 future
```
