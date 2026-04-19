//! Minimal Gemini embedding client for the leader.
//!
//! Embeds the user's query text via `models/gemini-embedding-2-preview`
//! so we can compare against the follower-produced chunk embeddings via
//! cosine similarity.

use anyhow::Result;
use serde::{Deserialize, Serialize};

const GEMINI_EMBED_MODEL: &str = "models/gemini-embedding-2-preview";
const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    content: Content,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: EmbeddingValues,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

pub struct GeminiEmbedClient {
    api_key: String,
    http: reqwest::Client,
}

impl GeminiEmbedClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            http: reqwest::Client::new(),
        }
    }

    /// Embed a text query into the same vector space used by the follower's
    /// `GeminiEmbedClient` for multimodal content.
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let req = EmbedRequest {
            model: GEMINI_EMBED_MODEL.to_string(),
            content: Content {
                parts: vec![Part {
                    text: text.to_string(),
                }],
            },
        };

        let url = format!("{GEMINI_API_BASE}/{GEMINI_EMBED_MODEL}:embedContent");

        let resp = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("gemini embed HTTP failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|e| format!("<error reading body: {e}>"));
            anyhow::bail!("gemini embed API error {status}: {body}");
        }

        let parsed: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("gemini embed parse failed: {e}"))?;

        let mut values = parsed.embedding.values;
        // L2-normalize so it's consistent with the normalized chunk
        // embeddings stored by followers.
        let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in values.iter_mut() {
                *x /= norm;
            }
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialise_embed_response() {
        let json = r#"{"embedding":{"values":[0.1,-0.2,0.3]}}"#;
        let resp: EmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.embedding.values.len(), 3);
        assert!((resp.embedding.values[0] - 0.1).abs() < 1e-6);
    }
}
