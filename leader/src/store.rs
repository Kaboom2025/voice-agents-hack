use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

use anyhow::{Context, Result};
use common::EmbeddingChunk;
use tracing::{info, warn};

use crate::chroma::ChromaCollections;

/// In-memory representation — stores everything EXCEPT the thumbnail
/// bytes (those live on disk to keep RAM bounded).
#[derive(Clone)]
pub struct StoredChunk {
    pub chunk: EmbeddingChunk,
    /// Whether a thumbnail JPEG exists on disk for this chunk.
    pub has_thumbnail: bool,
}

/// A scored result from a similarity query.
#[derive(Clone)]
pub struct ScoredChunk {
    pub chunk: StoredChunk,
    /// Cosine similarity score (−1..1). Higher is more relevant.
    pub score: f32,
}

/// Which modalities to match against when computing similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchModality {
    /// Compare query against the full [video || audio] embedding.
    All,
    /// Compare query against only the video portion of stored embeddings.
    Video,
    /// Compare query against only the audio portion of stored embeddings.
    Audio,
}

impl Default for SearchModality {
    fn default() -> Self {
        Self::All
    }
}

pub struct QueryFilter {
    /// When set, chunks are ranked by cosine similarity against this
    /// vector. When `None`, results are ranked by recency.
    pub query_embedding: Option<Vec<f32>>,
    pub time_start_ms: Option<u64>,
    pub time_end_ms: Option<u64>,
    pub camera_ids: Option<Vec<String>>,
    pub top_k: usize,
    /// Which portion of the stored embedding to compare against.
    pub modality: SearchModality,
    /// Optional text query for caption-based (sparse) matching. Results
    /// matching the caption text get a score boost.
    pub caption_query: Option<String>,
}

/// Persistent embedding store backed by an append-only file on disk.
///
/// Layout on disk:
/// ```text
/// <store_dir>/
///   chunks.bin          — append-only, length-prefixed postcard frames
///   thumbnails/
///     <chunk_id>.jpg    — representative JPEG per chunk
/// ```
pub struct EmbeddingStore {
    chunks: RwLock<Vec<StoredChunk>>,
    /// O(1) lookup by chunk_id → index into `chunks` vec.
    index: RwLock<HashMap<String, usize>>,
    max_size: usize,
    /// Path to the store directory. `None` for in-memory-only (tests).
    dir: Option<PathBuf>,
    /// Serialized append writer — one at a time.
    writer: Mutex<Option<BufWriter<fs::File>>>,
    /// ChromaDB collections for per-modality ANN search (PRD §5.2).
    /// `None` when ChromaDB is not configured (falls back to brute-force).
    chroma: Option<ChromaCollections>,
}

impl EmbeddingStore {
    /// Create an in-memory-only store (for tests).
    pub fn new(max_size: usize) -> Self {
        Self {
            chunks: RwLock::new(Vec::new()),
            index: RwLock::new(HashMap::new()),
            max_size,
            dir: None,
            writer: Mutex::new(None),
            chroma: None,
        }
    }

    /// Open (or create) a persistent store at `dir`.
    pub fn open(dir: &Path, max_size: usize, chroma: Option<ChromaCollections>) -> Result<Self> {
        fs::create_dir_all(dir)
            .with_context(|| format!("create store dir {}", dir.display()))?;
        fs::create_dir_all(dir.join("thumbnails"))
            .with_context(|| format!("create thumbnails dir"))?;

        let chunks_path = dir.join("chunks.bin");
        let mut loaded: Vec<StoredChunk> = Vec::new();

        if chunks_path.exists() {
            let data = fs::read(&chunks_path)
                .with_context(|| format!("read {}", chunks_path.display()))?;
            let mut cursor = &data[..];
            while cursor.len() >= 4 {
                let len = u32::from_le_bytes([cursor[0], cursor[1], cursor[2], cursor[3]]) as usize;
                cursor = &cursor[4..];
                if cursor.len() < len {
                    warn!("truncated chunk at end of store file, skipping");
                    break;
                }
                match postcard::from_bytes::<EmbeddingChunk>(&cursor[..len]) {
                    Ok(mut chunk) => {
                        // Check if thumbnail exists on disk.
                        let thumb_path = dir.join("thumbnails").join(format!("{}.jpg", chunk.chunk_id));
                        let has_thumbnail = thumb_path.exists();
                        // Don't keep thumbnail bytes in memory — they're on disk.
                        chunk.representative_jpeg = None;
                        loaded.push(StoredChunk { chunk, has_thumbnail });
                    }
                    Err(e) => {
                        warn!(%e, "corrupt chunk in store file, skipping");
                    }
                }
                cursor = &cursor[len..];
            }

            // Evict oldest if over max.
            if loaded.len() > max_size {
                let excess = loaded.len() - max_size;
                loaded.drain(..excess);
            }

            info!(chunks = loaded.len(), path = %chunks_path.display(), "loaded store from disk");
        }

        // Build the hash index.
        let mut idx = HashMap::with_capacity(loaded.len());
        for (i, sc) in loaded.iter().enumerate() {
            idx.insert(sc.chunk.chunk_id.clone(), i);
        }

        // Open the file for appending new chunks.
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&chunks_path)
            .with_context(|| format!("open {} for append", chunks_path.display()))?;

        Ok(Self {
            chunks: RwLock::new(loaded),
            index: RwLock::new(idx),
            max_size,
            dir: Some(dir.to_path_buf()),
            writer: Mutex::new(Some(BufWriter::new(file))),
            chroma,
        })
    }

    pub fn push(&self, chunk: EmbeddingChunk) {
        // P1-6: chunk_id is now follower-side deterministic
        // (`<camera_id>:<start_ts_ms>`). If we already have this id a
        // restart is replaying the append log — skip silently rather
        // than double-counting.
        {
            let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
            if idx.contains_key(&chunk.chunk_id) {
                return;
            }
        }

        // Write thumbnail to disk (if present).
        let has_thumbnail = if let (Some(dir), Some(ref jpeg)) = (&self.dir, &chunk.representative_jpeg) {
            let thumb_path = dir.join("thumbnails").join(format!("{}.jpg", chunk.chunk_id));
            if let Err(e) = fs::write(&thumb_path, jpeg) {
                warn!(%e, "failed to write thumbnail {}", thumb_path.display());
                false
            } else {
                true
            }
        } else {
            chunk.representative_jpeg.is_some()
        };

        // Persist the chunk (without thumbnail bytes) to the append log.
        if let Some(ref dir) = self.dir {
            let mut disk_chunk = chunk.clone();
            disk_chunk.representative_jpeg = None;
            if let Ok(encoded) = postcard::to_allocvec(&disk_chunk) {
                let len = (encoded.len() as u32).to_le_bytes();
                let mut writer = self.writer.lock().unwrap();
                if let Some(ref mut w) = *writer {
                    if let Err(e) = w.write_all(&len).and_then(|_| w.write_all(&encoded)).and_then(|_| w.flush()) {
                        warn!(%e, "failed to persist chunk to {}", dir.display());
                    }
                }
            }
        }

        // Insert into in-memory index (without thumbnail bytes).
        let mut mem_chunk = chunk;
        mem_chunk.representative_jpeg = None;
        let chunk_id = mem_chunk.chunk_id.clone();
        let mut chunks = self.chunks.write().unwrap_or_else(|e| e.into_inner());
        let mut idx = self.index.write().unwrap_or_else(|e| e.into_inner());
        if chunks.len() >= self.max_size && !chunks.is_empty() {
            // Remove oldest from index.
            let evicted_id = chunks[0].chunk.chunk_id.clone();
            idx.remove(&evicted_id);
            chunks.remove(0);
            // Rebuild index offsets (shifted by one).
            idx.clear();
            for (i, sc) in chunks.iter().enumerate() {
                idx.insert(sc.chunk.chunk_id.clone(), i);
            }
        }
        let pos = chunks.len();
        chunks.push(StoredChunk { chunk: mem_chunk, has_thumbnail });
        idx.insert(chunk_id, pos);
    }

    /// Upsert the chunk into ChromaDB per-modality collections (PRD §5.2).
    /// Video portion → `video-clips`, audio portion → `audio-clips`.
    pub async fn sync_to_chroma(&self, chunk: &EmbeddingChunk) {
        let Some(ref collections) = self.chroma else { return };

        let mut metadata = HashMap::new();
        metadata.insert("camera_id".into(), serde_json::json!(chunk.camera_id));
        metadata.insert("start_ts_ms".into(), serde_json::json!(chunk.start_ts_ms));
        metadata.insert("end_ts_ms".into(), serde_json::json!(chunk.end_ts_ms));

        let doc = chunk.caption.as_deref();

        // Video portion → video-clips collection.
        let video_emb = if chunk.video_dim > 0 && chunk.video_dim <= chunk.embedding.len() {
            &chunk.embedding[..chunk.video_dim]
        } else {
            &chunk.embedding[..]
        };
        if let Err(e) = collections
            .video
            .add(&chunk.chunk_id, video_emb, metadata.clone(), doc)
            .await
        {
            warn!(%e, chunk_id = %chunk.chunk_id, "failed to sync video to chromadb");
        }

        // Audio portion → audio-clips collection (if audio present).
        if chunk.audio_dim > 0 && chunk.video_dim + chunk.audio_dim <= chunk.embedding.len() {
            if let Some(ref audio_client) = collections.audio {
                let audio_emb = &chunk.embedding[chunk.video_dim..chunk.video_dim + chunk.audio_dim];
                if let Err(e) = audio_client
                    .add(&chunk.chunk_id, audio_emb, metadata, doc)
                    .await
                {
                    warn!(%e, chunk_id = %chunk.chunk_id, "failed to sync audio to chromadb");
                }
            }
        }
    }

    /// Search for chunks matching the filter. Uses per-modality ChromaDB
    /// ANN search with RRF merge when available (PRD §5.4), otherwise
    /// falls back to brute-force cosine scan.
    pub async fn query_async(&self, filter: &QueryFilter) -> Vec<ScoredChunk> {
        if let (Some(ref collections), Some(ref query_vec)) = (&self.chroma, &filter.query_embedding) {
            match self.query_via_chroma(collections, query_vec, filter).await {
                Ok(results) => return results,
                Err(e) => {
                    warn!(%e, "chromadb query failed, falling back to brute-force");
                }
            }
        }
        self.query_brute_force(filter)
    }

    /// PRD §5.4: search each modality collection independently, then
    /// reciprocal-rank-fuse the results.
    async fn query_via_chroma(
        &self,
        collections: &ChromaCollections,
        query_vec: &[f32],
        filter: &QueryFilter,
    ) -> Result<Vec<ScoredChunk>> {
        let where_filter = build_chroma_filter(filter);

        // 1) Search video collection — query dim should match video dim.
        let video_result = collections
            .video
            .query(query_vec, filter.top_k * 2, where_filter.clone())
            .await?;

        // 2) Search audio collection (if present and requested).
        let audio_result = if matches!(filter.modality, SearchModality::All | SearchModality::Audio) {
            if let Some(ref audio_client) = collections.audio {
                // Audio query only works if query dim matches audio dim.
                // For text queries this often won't match; we surface that
                // loudly so the UI can tell the user audio is unavailable.
                match audio_client.query(query_vec, filter.top_k * 2, where_filter).await {
                    Ok(r) => Some(r),
                    Err(e) => {
                        warn_audio_query_failed_once(query_vec.len(), &e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // 3) Reciprocal Rank Fusion (PRD §5.4 step 3).
        let mut rrf_scores: HashMap<String, f32> = HashMap::new();
        const RRF_K: f32 = 60.0; // standard RRF constant

        // Video hits.
        if let Some(ids) = video_result.ids.first() {
            for (rank, chunk_id) in ids.iter().enumerate() {
                *rrf_scores.entry(chunk_id.clone()).or_default() +=
                    1.0 / (RRF_K + rank as f32 + 1.0);
            }
        }

        // Audio hits.
        if let Some(ref ar) = audio_result {
            if let Some(ids) = ar.ids.first() {
                for (rank, chunk_id) in ids.iter().enumerate() {
                    *rrf_scores.entry(chunk_id.clone()).or_default() +=
                        1.0 / (RRF_K + rank as f32 + 1.0);
                }
            }
        }

        // 4) Map RRF scores back to StoredChunks + compute final similarity.
        let chunks = self.chunks.read().unwrap_or_else(|e| e.into_inner());
        let idx = self.index.read().unwrap_or_else(|e| e.into_inner());

        let mut scored: Vec<ScoredChunk> = Vec::with_capacity(rrf_scores.len());
        let rrf_weight = rrf_weight();
        for (chunk_id, rrf_score) in &rrf_scores {
            if let Some(&pos) = idx.get(chunk_id) {
                if let Some(sc) = chunks.get(pos) {
                    // Use modality-aware cosine similarity as the primary
                    // score, with RRF as a tiebreaker for multi-modality hits.
                    let cosine = modality_aware_similarity(
                        query_vec,
                        &sc.chunk.embedding,
                        sc.chunk.video_dim,
                        sc.chunk.audio_dim,
                        filter.modality,
                    );
                    let caption = filter.caption_query.as_ref()
                        .map(|cq| caption_boost(&sc.chunk.caption, cq))
                        .unwrap_or(0.0);
                    // Combine: cosine dominates, RRF adds up to
                    // ~rrf_weight·0.016 for chunks that appear in
                    // multiple collections. Both weights are tunable via
                    // LEADER_RRF_WEIGHT / LEADER_CAPTION_BOOST.
                    let score = cosine + rrf_score * rrf_weight + caption;
                    scored.push(ScoredChunk { chunk: sc.clone(), score });
                }
            }
        }

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(filter.top_k);
        Ok(scored)
    }

    /// Search for chunks matching the filter (sync, brute-force).
    pub fn query(&self, filter: &QueryFilter) -> Vec<ScoredChunk> {
        self.query_brute_force(filter)
    }

    fn query_brute_force(&self, filter: &QueryFilter) -> Vec<ScoredChunk> {
        let chunks = self.chunks.read().unwrap_or_else(|e| e.into_inner());

        let filtered: Vec<&StoredChunk> = chunks
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
            .collect();

        let mut scored: Vec<ScoredChunk> = match &filter.query_embedding {
            Some(query_vec) => {
                filtered
                    .into_iter()
                    .map(|sc| {
                        let score = modality_aware_similarity(
                            query_vec,
                            &sc.chunk.embedding,
                            sc.chunk.video_dim,
                            sc.chunk.audio_dim,
                            filter.modality,
                        );
                        // Add caption boost for hybrid retrieval.
                        let boost = filter.caption_query.as_ref()
                            .map(|cq| caption_boost(&sc.chunk.caption, cq))
                            .unwrap_or(0.0);
                        ScoredChunk {
                            chunk: sc.clone(),
                            score: score + boost,
                        }
                    })
                    .collect()
            }
            None => {
                // No embedding — still apply caption boost if text query given.
                filtered
                    .into_iter()
                    .map(|sc| {
                        let boost = filter.caption_query.as_ref()
                            .map(|cq| caption_boost(&sc.chunk.caption, cq))
                            .unwrap_or(0.0);
                        ScoredChunk {
                            chunk: sc.clone(),
                            score: boost,
                        }
                    })
                    .collect()
            }
        };

        if filter.query_embedding.is_some() || filter.caption_query.is_some() {
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scored.sort_by(|a, b| b.chunk.chunk.start_ts_ms.cmp(&a.chunk.chunk.start_ts_ms));
        }

        scored.truncate(filter.top_k);
        scored
    }

    /// Look up a single chunk by its `chunk_id` (O(1) via hash index).
    pub fn get_by_id(&self, chunk_id: &str) -> Option<StoredChunk> {
        let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
        let chunks = self.chunks.read().unwrap_or_else(|e| e.into_inner());
        idx.get(chunk_id).and_then(|&pos| chunks.get(pos).cloned())
    }

    /// Read a thumbnail JPEG from disk. Returns `None` if the chunk
    /// doesn't exist or has no thumbnail.
    pub fn get_thumbnail(&self, chunk_id: &str) -> Option<Vec<u8>> {
        let dir = self.dir.as_ref()?;
        let path = dir.join("thumbnails").join(format!("{}.jpg", chunk_id));
        fs::read(&path).ok()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.chunks.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Return the list of distinct camera IDs that have sent chunks.
    #[allow(dead_code)]
    pub fn camera_ids(&self) -> Vec<String> {
        let chunks = self.chunks.read().unwrap_or_else(|e| e.into_inner());
        let mut ids: Vec<String> = chunks
            .iter()
            .map(|sc| sc.chunk.camera_id.clone())
            .collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// Compact the store: rewrite the chunks file with only the chunks
    /// currently in memory (drops evicted entries from disk).
    #[allow(dead_code)]
    pub fn compact(&self) -> Result<()> {
        let Some(ref dir) = self.dir else { return Ok(()) };
        let chunks_path = dir.join("chunks.bin");
        let tmp_path = dir.join("chunks.bin.tmp");

        let chunks = self.chunks.read().unwrap_or_else(|e| e.into_inner());
        let mut f = BufWriter::new(
            fs::File::create(&tmp_path)
                .with_context(|| format!("create {}", tmp_path.display()))?,
        );
        for sc in chunks.iter() {
            let encoded = postcard::to_allocvec(&sc.chunk)
                .context("postcard encode during compact")?;
            let len = (encoded.len() as u32).to_le_bytes();
            f.write_all(&len)?;
            f.write_all(&encoded)?;
        }
        f.flush()?;
        drop(f);
        drop(chunks);

        fs::rename(&tmp_path, &chunks_path)
            .with_context(|| format!("rename {} → {}", tmp_path.display(), chunks_path.display()))?;

        // Re-open for appending.
        let file = fs::OpenOptions::new()
            .append(true)
            .open(&chunks_path)
            .context("reopen after compact")?;
        let mut writer = self.writer.lock().unwrap();
        *writer = Some(BufWriter::new(file));

        Ok(())
    }
}

/// Build a ChromaDB `$and` where-filter from our `QueryFilter`.
fn build_chroma_filter(filter: &QueryFilter) -> Option<serde_json::Value> {
    let mut conditions: Vec<serde_json::Value> = Vec::new();

    if let Some(start) = filter.time_start_ms {
        conditions.push(serde_json::json!({"end_ts_ms": {"$gte": start as i64}}));
    }
    if let Some(end) = filter.time_end_ms {
        conditions.push(serde_json::json!({"start_ts_ms": {"$lte": end as i64}}));
    }
    if let Some(ref cam_ids) = filter.camera_ids {
        if cam_ids.len() == 1 {
            conditions.push(serde_json::json!({"camera_id": {"$eq": cam_ids[0]}}));
        } else if cam_ids.len() > 1 {
            conditions.push(serde_json::json!({"camera_id": {"$in": cam_ids}}));
        }
    }

    match conditions.len() {
        0 => None,
        1 => Some(conditions.into_iter().next().unwrap()),
        _ => Some(serde_json::json!({"$and": conditions})),
    }
}

/// Cosine similarity between two vectors. Both are expected to be
/// L2-normalized upstream (follower does this), in which case this is
/// equivalent to the dot product. We still compute the full formula to
/// be safe against un-normalized vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Modality-aware similarity: extracts the relevant portion of the stored
/// embedding (video-only, audio-only, or full) and compares against the
/// query vector. Returns `0.0` on a genuine dimension mismatch — silently
/// truncating to a common prefix produces meaningless scores, so we'd
/// rather miss the hit loudly than rank garbage. Each unique
/// `(query_dim, target_dim)` mismatch is warned about exactly once.
fn modality_aware_similarity(
    query: &[f32],
    stored: &[f32],
    video_dim: usize,
    audio_dim: usize,
    modality: SearchModality,
) -> f32 {
    let target_slice = match modality {
        SearchModality::Video => {
            if video_dim > 0 && video_dim <= stored.len() {
                &stored[..video_dim]
            } else {
                stored
            }
        }
        SearchModality::Audio => {
            if audio_dim > 0 && video_dim + audio_dim <= stored.len() {
                &stored[video_dim..video_dim + audio_dim]
            } else {
                // No audio portion — return zero similarity.
                return 0.0;
            }
        }
        SearchModality::All => stored,
    };

    if query.len() == target_slice.len() {
        return cosine_similarity(query, target_slice);
    }

    warn_dim_mismatch_once(query.len(), target_slice.len(), modality);
    0.0
}

/// Tracks `(query_dim, target_dim, modality)` pairs we've already warned
/// about so log volume stays bounded when every stored chunk triggers the
/// same mismatch.
fn warn_dim_mismatch_once(query_dim: usize, target_dim: usize, modality: SearchModality) {
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(usize, usize, &'static str)>>> =
        OnceLock::new();
    let mod_label: &'static str = match modality {
        SearchModality::All => "all",
        SearchModality::Video => "video",
        SearchModality::Audio => "audio",
    };
    let set = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut guard = set.lock().unwrap_or_else(|e| e.into_inner());
    if guard.insert((query_dim, target_dim, mod_label)) {
        warn!(
            query_dim,
            target_dim,
            modality = mod_label,
            "dimension mismatch — similarity set to 0.0. Check that the \
             query backend matches the backend that produced stored chunks."
        );
    }
}

/// Warn once per unique query dim about the audio Chroma collection
/// failing. The most common cause is a text query whose dim does not
/// match the audio collection's stored dim.
fn warn_audio_query_failed_once(query_dim: usize, err: &anyhow::Error) {
    static SEEN: OnceLock<Mutex<std::collections::HashSet<usize>>> = OnceLock::new();
    let set = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut guard = set.lock().unwrap_or_else(|e| e.into_inner());
    if guard.insert(query_dim) {
        warn!(
            query_dim,
            error = %err,
            "audio-clips Chroma query failed — audio modality unavailable for \
             this query (most likely a dim mismatch)."
        );
    }
}

/// Compute a caption-text relevance boost via simple keyword overlap.
/// Returns a small additive score (0.0–`LEADER_CAPTION_BOOST`) so dense
/// similarity still dominates but keyword matches break ties. The cap is
/// configurable via the `LEADER_CAPTION_BOOST` env var (default `0.15`).
fn caption_boost(caption: &Option<String>, query: &str) -> f32 {
    let caption = match caption {
        Some(c) if !c.is_empty() => c.to_lowercase(),
        _ => return 0.0,
    };
    let query_lower = query.to_lowercase();
    let query_words: Vec<&str> = query_lower
        .split_whitespace()
        .filter(|w| w.len() > 2) // skip short stop-words
        .collect();
    if query_words.is_empty() {
        return 0.0;
    }
    let matches = query_words
        .iter()
        .filter(|w| caption.contains(**w))
        .count();
    let ratio = matches as f32 / query_words.len() as f32;
    let cap = caption_boost_cap();
    (ratio * cap).min(cap)
}

/// RRF mixing weight (`score = cosine + rrf * W + caption`). Defaults to
/// `2.0`; override via `LEADER_RRF_WEIGHT`.
fn rrf_weight() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LEADER_RRF_WEIGHT")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(2.0)
    })
}

/// Max additive caption-boost score. Defaults to `0.15`; override via
/// `LEADER_CAPTION_BOOST`.
fn caption_boost_cap() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LEADER_CAPTION_BOOST")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.15)
    })
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
            video_dim: 3,
            audio_dim: 0,
            caption: None,
            representative_jpeg: None,
        }
    }

    /// Create a chunk with a known embedding direction for similarity tests.
    fn make_vec_chunk(camera_id: &str, ts_ms: u64, emb: Vec<f32>) -> EmbeddingChunk {
        EmbeddingChunk {
            chunk_id: format!("{camera_id}-{ts_ms}"),
            camera_id: camera_id.into(),
            start_ts_ms: ts_ms,
            end_ts_ms: ts_ms + 5000,
            embedding: emb,
            video_dim: 3,
            audio_dim: 0,
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
            query_embedding: None,
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 5,
            modality: SearchModality::default(),
            caption_query: None,
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
            query_embedding: None,
            time_start_ms: Some(3000),
            time_end_ms: Some(8000),
            camera_ids: None,
            top_k: 10,
            modality: SearchModality::default(),
            caption_query: None,
        });
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.chunk.start_ts_ms, 5000);
        assert_eq!(results[1].chunk.chunk.start_ts_ms, 1000);
    }

    #[test]
    fn query_camera_filter_excludes_other_cameras() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-1", 2000));
        store.push(make_chunk("cam-0", 3000));
        let results = store.query(&QueryFilter {
            query_embedding: None,
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: Some(vec!["cam-1".into()]),
            top_k: 10,
            modality: SearchModality::default(),
            caption_query: None,
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.chunk.camera_id, "cam-1");
    }

    #[test]
    fn push_evicts_oldest_when_over_max_size() {
        let store = EmbeddingStore::new(3);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 2000));
        store.push(make_chunk("cam-0", 3000));
        store.push(make_chunk("cam-0", 4000));
        assert_eq!(store.len(), 3);
        let results = store.query(&QueryFilter {
            query_embedding: None,
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
            modality: SearchModality::default(),
            caption_query: None,
        });
        assert!(results.iter().all(|c| c.chunk.chunk.start_ts_ms >= 2000));
    }

    #[test]
    fn query_returns_most_recent_chunks_first() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        store.push(make_chunk("cam-0", 5000));
        store.push(make_chunk("cam-0", 3000));
        let results = store.query(&QueryFilter {
            query_embedding: None,
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
            modality: SearchModality::default(),
            caption_query: None,
        });
        assert_eq!(results[0].chunk.chunk.start_ts_ms, 5000);
        assert_eq!(results[2].chunk.chunk.start_ts_ms, 1000);
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_mismatched_dims_returns_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn query_with_embedding_ranks_by_similarity() {
        let store = EmbeddingStore::new(100);
        // Close to query direction
        store.push(make_vec_chunk("cam-0", 1000, vec![0.9, 0.1, 0.0]));
        // Opposite direction
        store.push(make_vec_chunk("cam-0", 2000, vec![-0.9, -0.1, 0.0]));
        // Orthogonal
        store.push(make_vec_chunk("cam-0", 3000, vec![0.0, 0.0, 1.0]));

        let query_vec = vec![1.0, 0.0, 0.0];
        let results = store.query(&QueryFilter {
            query_embedding: Some(query_vec),
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
            modality: SearchModality::Video,
            caption_query: None,
        });

        assert_eq!(results.len(), 3);
        // Most similar (aligned with query) should be first
        assert_eq!(results[0].chunk.chunk.start_ts_ms, 1000);
        assert!(results[0].score > 0.9);
        // Orthogonal in the middle
        assert_eq!(results[1].chunk.chunk.start_ts_ms, 3000);
        assert!(results[1].score.abs() < 0.1);
        // Opposite last
        assert_eq!(results[2].chunk.chunk.start_ts_ms, 2000);
        assert!(results[2].score < -0.9);
    }

    #[test]
    fn get_by_id_finds_existing_chunk() {
        let store = EmbeddingStore::new(100);
        store.push(make_chunk("cam-0", 1000));
        let found = store.get_by_id("cam-0-1000");
        assert!(found.is_some());
        assert_eq!(found.unwrap().chunk.camera_id, "cam-0");
    }

    #[test]
    fn get_by_id_returns_none_for_missing() {
        let store = EmbeddingStore::new(100);
        assert!(store.get_by_id("nonexistent").is_none());
    }

    #[test]
    fn modality_aware_video_only_search() {
        let store = EmbeddingStore::new(100);
        // Chunk with [video=3d || audio=3d] = 6d total
        let mut emb = vec![0.9, 0.1, 0.0, 0.0, 0.0, 1.0]; // video aligned with query, audio orthogonal
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut emb { *x /= norm; }
        store.push(EmbeddingChunk {
            chunk_id: "c1".into(),
            camera_id: "cam-0".into(),
            start_ts_ms: 1000,
            end_ts_ms: 6000,
            embedding: emb,
            video_dim: 3,
            audio_dim: 3,
            caption: None,
            representative_jpeg: None,
        });

        // Query with 3-d vector (matching video portion only)
        let results = store.query(&QueryFilter {
            query_embedding: Some(vec![1.0, 0.0, 0.0]),
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
            modality: SearchModality::Video,
            caption_query: None,
        });
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.9, "video-only search should score high");
    }

    #[test]
    fn caption_boost_adds_tiebreaker_score() {
        let store = EmbeddingStore::new(100);
        // Two chunks with identical embeddings but different captions
        store.push(EmbeddingChunk {
            chunk_id: "c-match".into(),
            camera_id: "cam-0".into(),
            start_ts_ms: 1000,
            end_ts_ms: 6000,
            embedding: vec![0.6, 0.8, 0.0],
            video_dim: 3,
            audio_dim: 0,
            caption: Some("person walking through door".into()),
            representative_jpeg: None,
        });
        store.push(EmbeddingChunk {
            chunk_id: "c-nomatch".into(),
            camera_id: "cam-0".into(),
            start_ts_ms: 2000,
            end_ts_ms: 7000,
            embedding: vec![0.6, 0.8, 0.0],
            video_dim: 3,
            audio_dim: 0,
            caption: Some("empty hallway".into()),
            representative_jpeg: None,
        });

        let results = store.query(&QueryFilter {
            query_embedding: Some(vec![0.6, 0.8, 0.0]),
            time_start_ms: None,
            time_end_ms: None,
            camera_ids: None,
            top_k: 10,
            modality: SearchModality::Video,
            caption_query: Some("person walking".into()),
        });
        assert_eq!(results.len(), 2);
        // Caption-matching chunk should rank higher due to boost
        assert_eq!(results[0].chunk.chunk.chunk_id, "c-match");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn dimension_mismatch_handled_gracefully() {
        // Text embed (3-d) vs stored chunk (6-d with video=3 + audio=3)
        let sim = modality_aware_similarity(
            &[1.0, 0.0, 0.0],      // 3-d query
            &[0.9, 0.1, 0.0, 0.0, 0.0, 1.0], // 6-d stored
            3, 3,
            SearchModality::Video,
        );
        assert!(sim > 0.9, "should match against video portion only");
    }
}
