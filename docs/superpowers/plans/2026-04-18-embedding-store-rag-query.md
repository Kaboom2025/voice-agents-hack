# Embedding Store & RAG Query System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an in-memory embedding store on the leader that persists received EmbeddingChunks with JPEG thumbnails, and a natural-language query pipeline backed by Gemini that answers security coordinator questions via `POST /api/query`.

**Architecture:** Three components added to the leader: (1) `EmbeddingStore` — an in-memory `Vec<StoredChunk>` behind `RwLock`, filtered by time range and camera ID; (2) `GeminiQueryClient` — parses NL queries into structured filters via `gemini-2.0-flash` and synthesizes answers using retrieved JPEG frames via vision; (3) Both wired into `AppState` with `FollowerMsg::Chunk` storing every incoming chunk. The follower is extended to include a representative JPEG thumbnail (middle frame of the embedding window, quality 60) in each `EmbeddingChunk`.

**Tech Stack:** Rust, reqwest 0.12 (Gemini REST API), base64 0.22, axum (existing HTTP layer), postcard (existing wire protocol), Gemini API (`gemini-2.0-flash` for NL parse + vision synthesis).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `common/src/lib.rs` | Modify | Add `representative_jpeg: Option<Vec<u8>>` to `EmbeddingChunk` |
| `follower/src/main.rs` | Modify | Populate `representative_jpeg` in push loop (middle frame, quality 60) |
| `follower/tests/e2e.rs` | Modify | Add `representative_jpeg: None` to `EmbeddingChunk` literal at line 96 |
| `leader/Cargo.toml` | Modify | Add `reqwest` and `base64` dependencies |
| `leader/src/store.rs` | Create | `EmbeddingStore` + `StoredChunk` + `QueryFilter` with push/query |
| `leader/src/query.rs` | Create | `GeminiQueryClient` — NL parse + vision synthesis |
| `leader/src/main.rs` | Modify | Add `store`/`query_client` to `AppState`, wire chunk handler, replace query endpoint |

---

### Task 1: Add `representative_jpeg` to `EmbeddingChunk` and populate in follower

**Files:**
- Modify: `common/src/lib.rs`
- Modify: `follower/src/main.rs:241-268`
- Modify: `follower/tests/e2e.rs:96-103`

- [ ] **Step 1: Write the failing test**

Add to the bottom of `common/src/lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_chunk_round_trips_with_jpeg() {
        let chunk = EmbeddingChunk {
            chunk_id: "test-0".into(),
            camera_id: "cam-0".into(),
            start_ts_ms: 1000,
            end_ts_ms: 6000,
            embedding: vec![0.1, 0.2, 0.3],
            caption: Some("test".into()),
            representative_jpeg: Some(vec![0xFF, 0xD8, 0xFF, 0xD9]),
        };
        let bytes = postcard::to_allocvec(&chunk).unwrap();
        let decoded: EmbeddingChunk = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.representative_jpeg, Some(vec![0xFF, 0xD8, 0xFF, 0xD9]));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p common
```

Expected: FAIL — `representative_jpeg` field does not exist yet.

- [ ] **Step 3: Add `representative_jpeg` field to `EmbeddingChunk`**

In `common/src/lib.rs`, replace the `EmbeddingChunk` struct definition (field order is critical — postcard encodes sequentially; new field goes last):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingChunk {
    /// Stable id (e.g. `blake3(camera_id || start_ts)`). Used for dedupe.
    pub chunk_id: String,
    pub camera_id: String,
    /// Unix epoch milliseconds.
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
    /// Concatenated `[video || audio]` embedding, L2-normalized upstream.
    pub embedding: Vec<f32>,
    /// Optional one-sentence caption for hybrid retrieval.
    pub caption: Option<String>,
    /// Middle frame of the embedding window, JPEG-encoded at quality 60.
    pub representative_jpeg: Option<Vec<u8>>,
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test -p common
```

Expected: PASS.

- [ ] **Step 5: Fix the e2e test — add new field to literal**

In `follower/tests/e2e.rs`, update the `EmbeddingChunk` literal at line 96:

```rust
let chunk = common::EmbeddingChunk {
    chunk_id: format!("cam-test-{seq}"),
    camera_id: "cam-test".into(),
    start_ts_ms: 0,
    end_ts_ms: 0,
    embedding: out.embedding,
    caption: out.caption,
    representative_jpeg: None,
};
```

- [ ] **Step 6: Populate `representative_jpeg` in the follower push loop**

In `follower/src/main.rs`, the push loop around line 241 builds `EmbeddingChunk`. Add the JPEG extraction between the `embed_window` call and the chunk construction. The full block should look like:

```rust
_ = interval.tick() => {
    let buf = frame_buffer.clone();
    let emb = video_embedder.clone();
    let out: VideoEmbeddingOutput = match emb.embed_window(&buf).await {
        Ok(o) => o,
        Err(e) => { warn!(error = %e, "embed_window failed, skipping chunk"); continue; }
    };

    let representative_jpeg = {
        let window = frame_buffer.window();
        window
            .get(window.len() / 2)
            .and_then(|(_, mid_frame)| {
                GeminiVideoEmbedder::encode_jpeg_bytes(mid_frame, 60).ok()
            })
    };

    let chunk = EmbeddingChunk {
        chunk_id: format!("{}-{}", args.camera_id, sent),
        camera_id: args.camera_id.clone(),
        start_ts_ms: out.start_ts_ms,
        end_ts_ms: out.end_ts_ms,
        embedding: out.embedding,
        caption: out.caption,
        representative_jpeg,
    };
    // ... rest of the loop is unchanged
```

- [ ] **Step 7: Run full workspace build and tests**

```bash
cargo build --workspace && cargo test --workspace
```

Expected: PASS. Confirm no `missing field 'representative_jpeg'` errors.

- [ ] **Step 8: Commit**

```bash
git add common/src/lib.rs follower/src/main.rs follower/tests/e2e.rs
git commit -m "feat: add representative_jpeg thumbnail to EmbeddingChunk wire protocol"
```

---

### Task 2: Add `reqwest` and `base64` dependencies to leader

**Files:**
- Modify: `leader/Cargo.toml`

- [ ] **Step 1: Verify the dependencies are missing**

```bash
cargo build -p leader 2>&1 | grep -c "error"
```

Expected: 0 errors (clean build before changes).

- [ ] **Step 2: Add dependencies to `leader/Cargo.toml`**

In `leader/Cargo.toml`, add under `[dependencies]`:

```toml
base64 = "0.22"
reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }
```

- [ ] **Step 3: Verify build succeeds**

```bash
cargo build -p leader
```

Expected: PASS. The crates download and compile.

- [ ] **Step 4: Commit**

```bash
git add leader/Cargo.toml
git commit -m "chore: add reqwest and base64 dependencies to leader"
```

---

### Task 3: Create `EmbeddingStore` with time/camera filtering

**Files:**
- Create: `leader/src/store.rs`
- Modify: `leader/src/main.rs` — add `mod store;`

- [ ] **Step 1: Add `mod store;` to `leader/src/main.rs`**

After the existing `#[cfg(feature = "cactus")] mod cactus;` line at the top of `leader/src/main.rs`, add:

```rust
mod store;
```

- [ ] **Step 2: Write the failing tests — create `leader/src/store.rs` with stubs**

```rust
use std::sync::RwLock;
use common::EmbeddingChunk;

pub struct StoredChunk {
    pub chunk: EmbeddingChunk,
}

impl Clone for StoredChunk {
    fn clone(&self) -> Self {
        Self { chunk: self.chunk.clone() }
    }
}

pub struct QueryFilter {
    pub time_start_ms: Option<u64>,
    pub time_end_ms: Option<u64>,
    pub camera_ids: Option<Vec<String>>,
    pub top_k: usize,
}

pub struct EmbeddingStore {
    chunks: RwLock<Vec<StoredChunk>>,
    max_size: usize,
}

impl EmbeddingStore {
    pub fn new(max_size: usize) -> Self {
        Self {
            chunks: RwLock::new(Vec::new()),
            max_size,
        }
    }

    pub fn push(&self, chunk: EmbeddingChunk) {
        todo!()
    }

    pub fn query(&self, filter: &QueryFilter) -> Vec<StoredChunk> {
        todo!()
    }

    pub fn len(&self) -> usize {
        self.chunks.read().expect("store poisoned").len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(camera_id: &str, ts_ms: u64) -> EmbeddingChunk {
        EmbeddingChunk {
            chunk_id: format!("{camera_id}-{ts_ms}"),
            camera_id: camera_id.into(),
            start_ts_ms: ts_ms,
            end_ts_ms: ts_ms + 5000,
            embedding: vec![0.1, 0.2, 0.3],
            caption: None,
            representative_jpeg: None,
        }
    }

    #[test]
    fn push_stores_chunk() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn query_no_filter_returns_top_k() {
        let store = EmbeddingStore::new(100);
        for i in 0..10u64 {
            store.push(make_chunk("cam-0", i * 1000));
        }
        let results = store.query(&QueryFilter {
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 5,
        });
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn query_time_range_filters_correctly() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 5000));
        store.push(make_chunk("cam-0", 10000));
        let results = store.query(&QueryFilter {
            time_start_ms: Some(3000),
            time_end_ms: Some(8000),
            camera_ids: None,
            top_k: 10,
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.start_ts_ms, 5000);
    }

    #[test]
    fn query_camera_filter_excludes_other_cameras() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-1", 2000));
        store.push(make_chunk("cam-0", 3000));
        let results = store.query(&QueryFilter {
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: Some(vec!["cam-1".into()]),
            top_k: 10,
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.camera_id, "cam-1");
    }

    #[test]
    fn push_evicts_oldest_when_over_max_size() {
        let store = EmbeddingStore::new(3);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 2000));
        store.push(make_chunk("cam-0", 3000));
        store.push(make_chunk("cam-0", 4000)); // should evict ts=1000
        assert_eq!(store.len(), 3);
        let results = store.query(&QueryFilter {
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
        });
        assert!(results.iter().all(|c| c.chunk.start_ts_ms >= 2000));
    }

    #[test]
    fn query_returns_most_recent_chunks_first() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 5000));
        store.push(make_chunk("cam-0", 3000));
        let results = store.query(&QueryFilter {
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
        });
        assert_eq!(results[0].chunk.start_ts_ms, 5000);
        assert_eq!(results[2].chunk.start_ts_ms, 1000);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cargo test -p leader -- store
```

Expected: FAIL with `not yet implemented` panics.

- [ ] **Step 4: Implement `push` and `query`**

Replace the `todo!()` bodies in `leader/src/store.rs`:

```rust
pub fn push(&self, chunk: EmbeddingChunk) {
    let mut chunks = self.chunks.write().expect("store poisoned");
    chunks.push(StoredChunk { chunk });
    if chunks.len() > self.max_size {
        chunks.remove(0);
    }
}

pub fn query(&self, filter: &QueryFilter) -> Vec<StoredChunk> {
    let chunks = self.chunks.read().expect("store poisoned");
    let mut matched: Vec<StoredChunk> = chunks
        .iter()
        .filter(|sc| {
            if let Some(start) = filter.time_start_ms {
                if sc.chunk.end_ts_ms < start {
                    return false;
                }
            }
            if let Some(end) = filter.time_end_ms {
                if sc.chunk.start_ts_ms > end {
                    return false;
                }
            }
            if let Some(ref cam_ids) = filter.camera_ids {
                if !cam_ids.contains(&sc.chunk.camera_id) {
                    return false;
                }
            }
            true
        })
        .cloned()
        .collect();
    matched.sort_by(|a, b| b.chunk.start_ts_ms.cmp(&a.chunk.start_ts_ms));
    matched.truncate(filter.top_k);
    matched
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cargo test -p leader -- store
```

Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add leader/src/store.rs leader/src/main.rs
git commit -m "feat: add in-memory EmbeddingStore with time/camera filtering"
```

---

### Task 4: Create `GeminiQueryClient` for NL parse and vision synthesis

**Files:**
- Create: `leader/src/query.rs`
- Modify: `leader/src/main.rs` — add `mod query;`

- [ ] **Step 1: Add `mod query;` to `leader/src/main.rs`**

After `mod store;`, add:

```rust
mod query;
```

- [ ] **Step 2: Write failing tests — create `leader/src/query.rs` with stubs**

```rust
use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde::Deserialize;
use serde_json::json;

use crate::store::StoredChunk;

pub struct ParsedQuery {
    pub time_start_ms: Option<u64>,
    pub time_end_ms: Option<u64>,
    pub camera_ids: Option<Vec<String>>,
    pub top_k: usize,
}

pub struct GeminiQueryClient {
    api_key: String,
    http: reqwest::Client,
}

const GEMINI_BASE: &str =
    "https://generativelanguage.googleapis.com/v1beta/models";

impl GeminiQueryClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            http: reqwest::Client::new(),
        }
    }

    pub async fn parse_nl_query(
        &self,
        query: &str,
        now_ms: u64,
        available_cameras: &[String],
    ) -> Result<ParsedQuery> {
        todo!()
    }

    pub async fn synthesize_answer(
        &self,
        query: &str,
        chunks: &[StoredChunk],
    ) -> Result<String> {
        todo!()
    }
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Content,
}

#[derive(Deserialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Deserialize)]
struct Part {
    text: String,
}

fn extract_text(resp: &GeminiResponse) -> &str {
    resp.candidates
        .first()
        .and_then(|c| c.content.parts.first())
        .map(|p| p.text.as_str())
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::EmbeddingChunk;

    fn stub_chunk(camera_id: &str, ts_ms: u64, jpeg: Option<Vec<u8>>) -> StoredChunk {
        StoredChunk {
            chunk: EmbeddingChunk {
                chunk_id: format!("{camera_id}-{ts_ms}"),
                camera_id: camera_id.into(),
                start_ts_ms: ts_ms,
                end_ts_ms: ts_ms + 5000,
                embedding: vec![],
                caption: Some(format!("scene at {ts_ms}")),
                representative_jpeg: jpeg,
            },
        }
    }

    #[test]
    fn parse_gemini_response_extracts_text() {
        let raw = r#"{"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}"#;
        let resp: GeminiResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(extract_text(&resp), "hello");
    }

    #[test]
    fn parse_gemini_response_returns_empty_on_no_candidates() {
        let raw = r#"{"candidates":[]}"#;
        let resp: GeminiResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(extract_text(&resp), "");
    }

    #[test]
    fn empty_chunks_returns_no_footage_message() {
        // synthesize_answer with zero chunks should return a canned message
        // without making a network call. We test this via the early-return path.
        let client = GeminiQueryClient::new("dummy");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(client.synthesize_answer("anything?", &[]));
        assert!(result.is_ok());
        assert!(result.unwrap().contains("No camera footage"));
    }
}
```

- [ ] **Step 3: Run tests to verify the stubs compile and two tests pass**

```bash
cargo test -p leader -- query
```

Expected: `parse_gemini_response_extracts_text` PASS, `parse_gemini_response_returns_empty_on_no_candidates` PASS, `empty_chunks_returns_no_footage_message` FAIL (hits `todo!()`).

- [ ] **Step 4: Implement `synthesize_answer`**

Replace the `todo!()` in `synthesize_answer`:

```rust
pub async fn synthesize_answer(
    &self,
    query: &str,
    chunks: &[StoredChunk],
) -> Result<String> {
    if chunks.is_empty() {
        return Ok("No camera footage found matching your query.".into());
    }

    let mut parts: Vec<serde_json::Value> = vec![json!({
        "text": "You are a security monitoring AI. Review the camera footage frames below and answer the security question."
    })];

    // Include up to 10 JPEG frames (most recent first, already sorted by caller)
    for sc in chunks.iter().filter(|sc| sc.chunk.representative_jpeg.is_some()).take(10) {
        let jpeg = sc.chunk.representative_jpeg.as_ref().unwrap();
        parts.push(json!({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": B64.encode(jpeg)
            }
        }));
        parts.push(json!({
            "text": format!(
                "[Camera: {} | {}ms – {}ms]",
                sc.chunk.camera_id, sc.chunk.start_ts_ms, sc.chunk.end_ts_ms
            )
        }));
    }

    // Text-only fallback for chunks without JPEGs
    let text_context: Vec<String> = chunks
        .iter()
        .filter(|sc| sc.chunk.representative_jpeg.is_none())
        .map(|sc| {
            format!(
                "[{} at {}ms] {}",
                sc.chunk.camera_id,
                sc.chunk.start_ts_ms,
                sc.chunk.caption.as_deref().unwrap_or("no description")
            )
        })
        .collect();
    if !text_context.is_empty() {
        parts.push(json!({
            "text": format!("Additional observations (no frame):\n{}", text_context.join("\n"))
        }));
    }

    parts.push(json!({
        "text": format!(
            "Security question: {query}\n\nAnswer concisely based only on the footage above."
        )
    }));

    let body = json!({ "contents": [{"parts": parts}] });
    let url = format!(
        "{GEMINI_BASE}/gemini-2.0-flash:generateContent?key={}",
        self.api_key
    );
    let resp: GeminiResponse = self
        .http
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("gemini synthesis request")?
        .error_for_status()
        .context("gemini synthesis status")?
        .json()
        .await
        .context("gemini synthesis decode")?;

    Ok(extract_text(&resp).to_string())
}
```

- [ ] **Step 5: Run tests — `empty_chunks_returns_no_footage_message` should now pass**

```bash
cargo test -p leader -- query
```

Expected: all 3 unit tests PASS.

- [ ] **Step 6: Implement `parse_nl_query`**

Replace the `todo!()` in `parse_nl_query`:

```rust
pub async fn parse_nl_query(
    &self,
    query: &str,
    now_ms: u64,
    available_cameras: &[String],
) -> Result<ParsedQuery> {
    let camera_list = available_cameras.join(", ");
    let thirty_min_ago = now_ms.saturating_sub(30 * 60 * 1000);
    let prompt = format!(
        "Parse this security monitoring query into a JSON object.\n\
        Current time: {now_ms} ms since epoch.\n\
        Available cameras: [{camera_list}].\n\n\
        Query: \"{query}\"\n\n\
        Return JSON with exactly these fields:\n\
        - \"time_start_ms\": integer or null (null = no lower bound; \
          'last 30 minutes' → {thirty_min_ago})\n\
        - \"time_end_ms\": integer or null (null = use current time {now_ms})\n\
        - \"camera_ids\": array of strings or null (null = all cameras; \
          map location words to camera IDs from the available list)\n\
        - \"top_k\": integer, default 20, max 50"
    );

    let body = json!({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    });
    let url = format!(
        "{GEMINI_BASE}/gemini-2.0-flash:generateContent?key={}",
        self.api_key
    );
    let resp: GeminiResponse = self
        .http
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("gemini parse request")?
        .error_for_status()
        .context("gemini parse status")?
        .json()
        .await
        .context("gemini parse decode")?;

    let text = extract_text(&resp);
    let parsed: serde_json::Value = serde_json::from_str(text)
        .with_context(|| format!("gemini returned non-JSON for parse: {text}"))?;

    Ok(ParsedQuery {
        time_start_ms: parsed["time_start_ms"].as_u64(),
        time_end_ms: parsed["time_end_ms"].as_u64(),
        camera_ids: parsed["camera_ids"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()),
        top_k: parsed["top_k"].as_u64().unwrap_or(20) as usize,
    })
}
```

- [ ] **Step 7: Run full build**

```bash
cargo build -p leader
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add leader/src/query.rs leader/src/main.rs
git commit -m "feat: add GeminiQueryClient for NL query parsing and vision synthesis"
```

---

### Task 5: Wire store and query client into leader `AppState` and `/api/query`

**Files:**
- Modify: `leader/src/main.rs` — AppState, Args, `serve_stream`, `query` handler

- [ ] **Step 1: Write a compile-time test that the store is accessible from main**

Add at the bottom of `leader/src/main.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{EmbeddingStore, QueryFilter};
    use common::EmbeddingChunk;
    use std::sync::Arc;

    fn make_chunk(camera_id: &str, ts_ms: u64) -> EmbeddingChunk {
        EmbeddingChunk {
            chunk_id: format!("{camera_id}-{ts_ms}"),
            camera_id: camera_id.into(),
            start_ts_ms: ts_ms,
            end_ts_ms: ts_ms + 5000,
            embedding: vec![0.1, 0.2],
            caption: None,
            representative_jpeg: None,
        }
    }

    #[test]
    fn store_accessible_and_chunks_persist() {
        let store = Arc::new(EmbeddingStore::new(1000));
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 6000));
        let results = store.query(&QueryFilter {
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
        });
        assert_eq!(results.len(), 2);
    }
}
```

Run: `cargo test -p leader -- tests`
Expected: PASS.

- [ ] **Step 2: Add imports to `leader/src/main.rs`**

Add after the existing `use` block at the top:

```rust
use crate::query::GeminiQueryClient;
use crate::store::{EmbeddingStore, QueryFilter};
```

- [ ] **Step 3: Update `AppState` to include `store` and `query_client`**

Replace the `AppState` struct definition:

```rust
#[derive(Clone)]
struct AppState {
    registry: Arc<RwLock<HashMap<String, CameraEntry>>>,
    next_req_id: Arc<AtomicU64>,
    chunks_total: Arc<AtomicU64>,
    frame_timeout: Duration,
    store: Arc<EmbeddingStore>,
    query_client: Option<Arc<GeminiQueryClient>>,
    #[cfg(feature = "cactus")]
    llm: Option<Arc<CactusModel>>,
}
```

- [ ] **Step 4: Add `gemini_api_key` and `store_max_size` args**

Add to the `Args` struct:

```rust
/// Gemini API key for RAG query parsing and answer synthesis.
#[arg(long, env = "GEMINI_API_KEY")]
gemini_api_key: Option<String>,

/// Max embedding chunks to retain in memory (older chunks evicted first).
#[arg(long, env = "STORE_MAX_SIZE", default_value_t = 10_000)]
store_max_size: usize,
```

- [ ] **Step 5: Initialize `store` and `query_client` in `main()`**

In `main()`, directly after the ticket-write block (before the `#[cfg(feature = "cactus")]` Gemma load), add:

```rust
let store = Arc::new(EmbeddingStore::new(args.store_max_size));
let query_client = args.gemini_api_key.as_ref().map(|key| {
    info!("gemini query client ready");
    Arc::new(GeminiQueryClient::new(key.clone()))
});
if query_client.is_none() {
    warn!("GEMINI_API_KEY not set — /api/query will return stub answers");
}
```

- [ ] **Step 6: Update `AppState` initialization in `main()`**

```rust
let app_state = AppState {
    registry: Arc::new(RwLock::new(HashMap::new())),
    next_req_id: Arc::new(AtomicU64::new(1)),
    chunks_total: Arc::new(AtomicU64::new(0)),
    frame_timeout: Duration::from_millis(args.frame_timeout_ms),
    store,
    query_client,
    #[cfg(feature = "cactus")]
    llm,
};
```

- [ ] **Step 7: Wire store into the `FollowerMsg::Chunk` handler in `serve_stream()`**

Find the `FollowerMsg::Chunk(chunk)` arm (around line 385). After the two `fetch_add` calls and before the `info!` log, add `state.store.push(chunk.clone());`:

```rust
FollowerMsg::Chunk(chunk) => {
    let n = state.chunks_total.fetch_add(1, Ordering::Relaxed) + 1;
    if let Some(e) = entry.as_ref() {
        e.chunks_total.fetch_add(1, Ordering::Relaxed);
        e.last_seen_ms.store(now_ms(), Ordering::Relaxed);
    }
    state.store.push(chunk.clone());  // <-- new line
    info!(
        total = n,
        camera = %chunk.camera_id,
        chunk = %chunk.chunk_id,
        dim = chunk.embedding.len(),
        caption = chunk.caption.as_deref().unwrap_or(""),
        "recv chunk",
    );
    let ack = LeaderMsg::Ack { chunk_id: chunk.chunk_id };
    if let Err(e) = write_frame(&mut send, &ack).await {
        error!(%e, "ack write failed");
        break;
    }
}
```

- [ ] **Step 8: Replace the `#[cfg(not(feature = "cactus"))]` `query` handler with the RAG pipeline**

Replace the entire `#[cfg(not(feature = "cactus"))] async fn query(...)` function:

```rust
#[cfg(not(feature = "cactus"))]
async fn query(
    State(state): State<AppState>,
    Json(req): Json<QueryReq>,
) -> Response {
    if req.query.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "empty query").into_response();
    }

    let Some(ref client) = state.query_client else {
        return Json(QueryResp {
            answer: format!(
                "(GEMINI_API_KEY not configured — echoing) {}",
                req.query
            ),
            citations: Vec::new(),
            hits: Vec::new(),
        })
        .into_response();
    };

    let now = now_ms();
    let available_cameras: Vec<String> = state
        .registry
        .read()
        .expect("registry poisoned")
        .keys()
        .cloned()
        .collect();

    // Step 1: Parse NL query into structured filters
    let parsed = match client
        .parse_nl_query(&req.query, now, &available_cameras)
        .await
    {
        Ok(p) => p,
        Err(e) => {
            error!(%e, "parse_nl_query failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("parse error: {e}"),
            )
                .into_response();
        }
    };

    // Step 2: Retrieve matching chunks
    let filter = QueryFilter {
        time_start_ms: parsed.time_start_ms,
        time_end_ms: parsed.time_end_ms.or(Some(now)),
        camera_ids: req.cameras.clone().or(parsed.camera_ids),
        top_k: req.top_k.map(|k| k as usize).unwrap_or(parsed.top_k),
    };
    let chunks = state.store.query(&filter);
    let n_chunks = chunks.len();

    // Step 3: Synthesize answer with Gemini vision
    let answer = match client.synthesize_answer(&req.query, &chunks).await {
        Ok(a) => a,
        Err(e) => {
            error!(%e, "synthesize_answer failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("synthesis error: {e}"),
            )
                .into_response();
        }
    };

    info!(n_chunks, "query answered");
    Json(QueryResp {
        answer,
        citations: Vec::new(),
        hits: Vec::new(),
    })
    .into_response()
}
```

- [ ] **Step 9: Run build and all tests**

```bash
cargo build --workspace && cargo test --workspace
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add leader/src/main.rs
git commit -m "feat: wire EmbeddingStore and GeminiQueryClient into leader /api/query RAG pipeline"
```

---

### Task 6: Smoke test the full end-to-end pipeline

**Prerequisites:** `GEMINI_API_KEY` set in environment or `.env` file at project root.

- [ ] **Step 1: Start the leader**

```bash
GEMINI_API_KEY=<your_key> cargo run --bin leader
```

Expected output:
```
leader ready
http on: http://127.0.0.1:8080
gemini query client ready
ticket (share with remote followers): <TICKET>
```

- [ ] **Step 2: Start a follower with synthetic embeddings**

In a second terminal (no camera needed):

```bash
GEMINI_API_KEY=<your_key> cargo run --bin follower -- --synthetic --camera-id cam-parking --count 4
```

Expected: 4 chunks sent. Leader logs `recv chunk` 4 times with `dim=3072`.

- [ ] **Step 3: Verify cameras are registered**

```bash
curl -s http://127.0.0.1:8080/api/cameras | python3 -m json.tool
```

Expected: `cam-parking` appears with `status: "online"` and `chunks_per_min > 0`.

- [ ] **Step 4: Query real-time status**

```bash
curl -s -X POST http://127.0.0.1:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Is anyone in the parking lot right now?"}' \
  | python3 -m json.tool
```

Expected: `answer` field contains a Gemini-generated response that references the camera footage, not an echo.

- [ ] **Step 5: Query a time range**

```bash
curl -s -X POST http://127.0.0.1:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What happened in the last 5 minutes?"}' \
  | python3 -m json.tool
```

Expected: Gemini synthesizes an answer from recent chunks.

- [ ] **Step 6: Start a second follower and verify multi-camera query**

```bash
GEMINI_API_KEY=<your_key> cargo run --bin follower -- --synthetic --camera-id cam-checkout --count 4
```

Then query:
```bash
curl -s -X POST http://127.0.0.1:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Which area is busiest right now?"}' \
  | python3 -m json.tool
```

Expected: Gemini references both `cam-parking` and `cam-checkout` in its answer.

- [ ] **Step 7: Commit any fixes found during smoke test**

```bash
git add -p
git commit -m "fix: <describe anything that needed patching>"
```

---

## Self-Review

### Spec Coverage

| Query pattern from spec | Handled by |
|---|---|
| "How many people in the store right now?" | parse→last 10s, all cameras; Gemini vision counts |
| "Is anyone in the parking lot?" | parse→camera_ids=[cam-parking]; Gemini synthesizes |
| "How long has person in red hoodie been outside?" | parse→cam-parking, large time window; Gemini reasons over frames |
| "Has anyone been lingering in one spot?" | parse→all cameras, 30min window; Gemini flags low-movement |
| "What happened at checkout between 2:00-2:15?" | parse→cam-checkout, explicit time range; Gemini reviews |
| "Is anyone still in the store after closing?" | parse→all cameras, post-close timestamps; Gemini answers |

### Placeholder Scan

No TODOs, no "TBD", no "similar to Task N" references. All code is complete. ✓

### Type Consistency

- `StoredChunk` defined Task 3 → used in Task 4 (`&[StoredChunk]`) and Task 5 (store returns `Vec<StoredChunk>`) ✓
- `QueryFilter` defined Task 3 → constructed in Task 5 handler ✓
- `ParsedQuery` defined Task 4 → consumed in Task 5 handler ✓
- `GeminiQueryClient::parse_nl_query(&str, u64, &[String]) -> Result<ParsedQuery>` consistent across Tasks 4 and 5 ✓
- `GeminiQueryClient::synthesize_answer(&str, &[StoredChunk]) -> Result<String>` consistent across Tasks 4 and 5 ✓
- `representative_jpeg: Option<Vec<u8>>` added Task 1, read in Task 4 via `sc.chunk.representative_jpeg` ✓
- `EmbeddingStore::push(EmbeddingChunk)` defined Task 3, called in Task 5 with `chunk.clone()` (chunk is `EmbeddingChunk`) ✓
