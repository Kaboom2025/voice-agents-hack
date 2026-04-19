//! ChromaDB HTTP client for the leader.
//!
//! Talks to a ChromaDB server (started via `uvx --from chromadb chroma run`)
//! over its REST API. Per PRD §5.2, we maintain separate collections for
//! each modality (`video_clips`, `audio_clips`) so that query vectors
//! always match the stored vector dimension.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Tenant/database path prefix for ChromaDB v2 API.
const V2_PREFIX: &str = "api/v2/tenants/default_tenant/databases/default_database";

/// Per-modality ChromaDB collections (PRD §5.2).
///
/// - `video_clips` — stores the video portion of each chunk embedding
/// - `audio_clips` — stores the audio portion (when available)
///
/// Both use cosine distance. The text query vector from `embed_text()`
/// matches the dimensionality of the video collection, so ANN search
/// works without padding or truncation.
pub struct ChromaCollections {
    pub video: ChromaClient,
    pub audio: Option<ChromaClient>,
}

impl ChromaCollections {
    /// Connect to ChromaDB and create/get the per-modality collections.
    pub async fn connect(base_url: &str) -> Result<Self> {
        // Verify server is reachable.
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("reqwest client build with static timeout is infallible");
        let hb_url = format!("{base_url}/api/v2/heartbeat");
        let hb = http
            .get(&hb_url)
            .send()
            .await
            .context("chromadb: heartbeat request failed — is chroma running?")?;
        if !hb.status().is_success() {
            anyhow::bail!("chromadb heartbeat returned {}", hb.status());
        }

        let video = ChromaClient::connect_collection(base_url, "video-clips").await?;
        let audio = match ChromaClient::connect_collection(base_url, "audio-clips").await {
            Ok(c) => Some(c),
            Err(e) => {
                warn!(%e, "failed to create audio-clips collection, audio search disabled");
                None
            }
        };

        let video_count = video.count().await.unwrap_or(0);
        let audio_count = audio.as_ref().map(|a| {
            // Can't await in map, just log 0 for now
            0u64
        }).unwrap_or(0);
        info!(video_count, "chromadb collections ready");

        Ok(Self { video, audio })
    }
}

/// Client for a single ChromaDB collection.
pub struct ChromaClient {
    http: reqwest::Client,
    base_url: String,
    collection_id: String,
    collection_name: String,
}

#[derive(Serialize)]
struct CreateCollectionReq {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    configuration: Option<serde_json::Value>,
    get_or_create: bool,
}

#[derive(Deserialize)]
struct CollectionInfo {
    id: String,
    name: String,
}

#[derive(Serialize)]
struct AddReq {
    ids: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadatas: Option<Vec<HashMap<String, serde_json::Value>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    documents: Option<Vec<String>>,
}

#[derive(Serialize)]
struct QueryReq {
    query_embeddings: Vec<Vec<f32>>,
    n_results: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    r#where: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
}

#[derive(Deserialize, Debug)]
pub struct QueryResult {
    pub ids: Vec<Vec<String>>,
    pub distances: Option<Vec<Vec<f64>>>,
    pub metadatas: Option<Vec<Vec<Option<HashMap<String, serde_json::Value>>>>>,
    pub documents: Option<Vec<Vec<Option<String>>>>,
}

// ─── Implementation ──────────────────────────────────────────────────

impl ChromaClient {
    /// Connect to a ChromaDB server and get-or-create a named collection
    /// with cosine distance. Does NOT check heartbeat — caller should
    /// use `ChromaCollections::connect()` which does.
    async fn connect_collection(base_url: &str, collection_name: &str) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("reqwest client build with static timeout is infallible");
        let url = format!("{base_url}/{V2_PREFIX}/collections");

        let body = CreateCollectionReq {
            name: collection_name.to_string(),
            metadata: None,
            configuration: Some(serde_json::json!({
                "hnsw": { "space": "cosine" }
            })),
            get_or_create: true,
        };

        let resp = http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("chromadb: create/get collection request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("chromadb create collection failed ({status}): {body}");
        }

        let info: CollectionInfo = resp.json().await.context("chromadb: parse collection info")?;
        info!(
            collection = %info.name,
            id = %info.id,
            "chromadb collection ready"
        );

        Ok(Self {
            http,
            base_url: base_url.to_string(),
            collection_id: info.id,
            collection_name: info.name,
        })
    }

    /// Add (upsert) a single embedding with metadata.
    pub async fn add(
        &self,
        id: &str,
        embedding: &[f32],
        metadata: HashMap<String, serde_json::Value>,
        document: Option<&str>,
    ) -> Result<()> {
        let url = format!(
            "{}/{V2_PREFIX}/collections/{}/add",
            self.base_url, self.collection_id
        );

        let body = AddReq {
            ids: vec![id.to_string()],
            embeddings: vec![embedding.to_vec()],
            metadatas: Some(vec![metadata]),
            documents: document.map(|d| vec![d.to_string()]),
        };

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("chromadb add {id}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("chromadb add failed ({status}): {body}");
        }

        debug!(id, collection = %self.collection_name, "chromadb: added");
        Ok(())
    }

    /// Query by embedding vector. Returns up to `n_results` hits sorted
    /// by cosine distance (lower = more similar).
    pub async fn query(
        &self,
        query_embedding: &[f32],
        n_results: usize,
        where_filter: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        let url = format!(
            "{}/{V2_PREFIX}/collections/{}/query",
            self.base_url, self.collection_id
        );

        let body = QueryReq {
            query_embeddings: vec![query_embedding.to_vec()],
            n_results,
            r#where: where_filter,
            include: Some(vec![
                "metadatas".into(),
                "documents".into(),
                "distances".into(),
            ]),
        };

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("chromadb query request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("chromadb query failed ({status}): {body}");
        }

        let result: QueryResult = resp.json().await.context("chromadb: parse query result")?;
        debug!(
            n_hits = result.ids.first().map(|v| v.len()).unwrap_or(0),
            "chromadb: query returned"
        );
        Ok(result)
    }

    /// How many embeddings are in the collection.
    pub async fn count(&self) -> Result<u64> {
        let url = format!(
            "{}/{V2_PREFIX}/collections/{}/count",
            self.base_url, self.collection_id
        );
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .context("chromadb count request")?;
        if !resp.status().is_success() {
            anyhow::bail!("chromadb count failed: {}", resp.status());
        }
        let n: u64 = resp.json().await.context("chromadb count parse")?;
        Ok(n)
    }
}
