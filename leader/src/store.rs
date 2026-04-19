use common::EmbeddingChunk;
use std::sync::RwLock;

#[derive(Clone)]
pub struct StoredChunk {
    pub chunk: EmbeddingChunk,
}

/// A scored result from a similarity query.
#[derive(Clone)]
pub struct ScoredChunk {
    pub chunk: StoredChunk,
    /// Cosine similarity score (−1..1). Higher is more relevant.
    pub score: f32,
}

pub struct QueryFilter {
    /// When set, chunks are ranked by cosine similarity against this
    /// vector. When `None`, results are ranked by recency.
    pub query_embedding: Option<Vec<f32>>,
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
        let mut chunks = self.chunks.write().unwrap_or_else(|e| e.into_inner());
        chunks.push(StoredChunk { chunk });
        if chunks.len() > self.max_size {
            chunks.remove(0);
        }
    }

    /// Search for chunks matching the filter. When `query_embedding` is
    /// provided, results are ranked by cosine similarity (true RAG
    /// retrieval). Otherwise falls back to recency ordering.
    pub fn query(&self, filter: &QueryFilter) -> Vec<ScoredChunk> {
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
                        let score = cosine_similarity(query_vec, &sc.chunk.embedding);
                        ScoredChunk {
                            chunk: sc.clone(),
                            score,
                        }
                    })
                    .collect()
            }
            None => {
                // No query vector → rank by recency, score = 0.0
                filtered
                    .into_iter()
                    .map(|sc| ScoredChunk {
                        chunk: sc.clone(),
                        score: 0.0,
                    })
                    .collect()
            }
        };

        // Sort: similarity descending if we have an embedding, recency otherwise.
        if filter.query_embedding.is_some() {
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scored.sort_by(|a, b| b.chunk.chunk.start_ts_ms.cmp(&a.chunk.chunk.start_ts_ms));
        }

        scored.truncate(filter.top_k);
        scored
    }

    /// Look up a single chunk by its `chunk_id`.
    pub fn get_by_id(&self, chunk_id: &str) -> Option<StoredChunk> {
        let chunks = self.chunks.read().unwrap_or_else(|e| e.into_inner());
        chunks.iter().find(|sc| sc.chunk.chunk_id == chunk_id).cloned()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.chunks.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Return the list of distinct camera IDs that have sent chunks.
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
}
