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

const GEMINI_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

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
        let url = format!("{GEMINI_BASE}/gemini-2.0-flash:generateContent");
        let resp: GeminiResponse = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
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
            camera_ids: parsed["camera_ids"].as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            }),
            top_k: parsed["top_k"].as_u64().unwrap_or(20) as usize,
        })
    }

    pub async fn synthesize_answer(&self, query: &str, chunks: &[StoredChunk]) -> Result<String> {
        if chunks.is_empty() {
            return Ok("No camera footage found matching your query.".into());
        }

        let mut parts: Vec<serde_json::Value> = vec![json!({
            "text": "You are a security monitoring AI. Review the camera footage frames below and answer the security question."
        })];

        for (sc, jpeg) in chunks
            .iter()
            .filter_map(|sc| sc.chunk.representative_jpeg.as_ref().map(|jpeg| (sc, jpeg)))
            .take(10)
        {
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
        let url = format!("{GEMINI_BASE}/gemini-2.0-flash:generateContent");
        let resp: GeminiResponse = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
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

    #[allow(dead_code)]
    fn stub_chunk(camera_id: &str, ts_ms: u64, jpeg: Option<Vec<u8>>) -> StoredChunk {
        StoredChunk {
            chunk: EmbeddingChunk {
                chunk_id: format!("{camera_id}-{ts_ms}"),
                camera_id: camera_id.into(),
                start_ts_ms: ts_ms,
                end_ts_ms: ts_ms + 5000,
                embedding: vec![],
                video_dim: 0,
                audio_dim: 0,
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
        let client = GeminiQueryClient::new("dummy");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(client.synthesize_answer("anything?", &[]));
        assert!(result.is_ok());
        assert!(result.unwrap().contains("No camera footage"));
    }
}
