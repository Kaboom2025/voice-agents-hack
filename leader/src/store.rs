use common::EmbeddingChunk;
use std::sync::RwLock;

#[derive(Clone)]
pub struct StoredChunk {
    pub chunk: EmbeddingChunk,
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
        let mut chunks = self.chunks.write().unwrap_or_else(|e| e.into_inner());
        chunks.push(StoredChunk { chunk });
        if chunks.len() > self.max_size {
            chunks.remove(0);
        }
    }

    pub fn query(&self, filter: &QueryFilter) -> Vec<StoredChunk> {
        let chunks = self.chunks.read().unwrap_or_else(|e| e.into_inner());
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

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.chunks.read().unwrap_or_else(|e| e.into_inner()).len()
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
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.start_ts_ms, 5000);
        assert_eq!(results[1].chunk.start_ts_ms, 1000);
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
        store.push(make_chunk("cam-0", 4000));
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
