use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::json;
use tracing::{debug, info, warn};

use crate::cactus::CactusModel;
use crate::store::{EmbeddingStore, StoredChunk};

pub struct ParsedQuery {
    pub time_start_ms: Option<u64>,
    pub time_end_ms: Option<u64>,
    pub camera_ids: Option<Vec<String>>,
    pub top_k: usize,
}

pub struct CactusQueryHandler {
    model: Arc<CactusModel>,
}

impl CactusQueryHandler {
    pub fn new(model: Arc<CactusModel>) -> Self {
        Self { model }
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
            "Parse this security monitoring query and return ONLY a JSON object, nothing else.\n\
            Do NOT think out loud, do NOT explain, do NOT use any reasoning channel.\n\
            Current time: {now_ms} ms since epoch.\n\
            Available cameras: [{camera_list}].\n\n\
            Query: \"{query}\"\n\n\
            Return JSON with exactly these fields:\n\
            - \"time_start_ms\": integer or null (null = no lower bound; 'last 30 minutes' → {thirty_min_ago})\n\
            - \"time_end_ms\": integer or null (null = use current time {now_ms})\n\
            - \"camera_ids\": array of strings or null (null = all cameras)\n\
            - \"top_k\": integer, default 20, max 50\n\n\
            Example: {{\"time_start_ms\":{thirty_min_ago},\"time_end_ms\":null,\"camera_ids\":null,\"top_k\":20}}\n\
            Output only the JSON object:"
        );

        let messages = text_messages(&prompt);
        let model = Arc::clone(&self.model);
        // Gemma 4 emits a thinking channel first; give it room to think and
        // still produce the final JSON. 2048 tokens covers both.
        info!(query = %query, "parse_nl_query: invoking gemma");
        let t0 = Instant::now();
        let raw = tokio::task::spawn_blocking(move || {
            model.complete(&messages, Some(r#"{"max_tokens":2048}"#))
        })
        .await
        .context("parse task panicked")?
        .context("cactus parse failed")?;
        info!(elapsed_ms = t0.elapsed().as_millis() as u64, "parse_nl_query: gemma returned");

        let response = extract_response(&raw);
        debug!(response = %response, "parse_nl_query: raw response");
        let json_str = find_json(&response).unwrap_or(&response);
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .with_context(|| format!("gemma returned non-JSON: {response}"))?;
        info!("parse_nl_query: parsed JSON ok");

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

    pub async fn synthesize_answer(&self, query: &str, chunks: &[StoredChunk], store: &EmbeddingStore) -> Result<String> {
        if chunks.is_empty() {
            return Ok("No camera footage found matching your query.".into());
        }

        // Load representative JPEGs from disk and write to temp files;
        // Cactus reads them by path.
        let mut temp_files: Vec<tempfile::NamedTempFile> = Vec::new();
        let mut image_paths: Vec<String> = Vec::new();
        for sc in chunks.iter().take(10) {
            if let Some(jpeg) = store.get_thumbnail(&sc.chunk.chunk_id) {
                let mut tmp = tempfile::Builder::new()
                    .suffix(".jpg")
                    .tempfile()
                    .context("create temp jpeg")?;
                tmp.write_all(&jpeg).context("write temp jpeg")?;
                image_paths.push(tmp.path().to_string_lossy().into_owned());
                temp_files.push(tmp);
            }
        }

        let observations: Vec<String> = chunks
            .iter()
            .map(|sc| {
                format!(
                    "[{} {}ms–{}ms] {}",
                    sc.chunk.camera_id,
                    sc.chunk.start_ts_ms,
                    sc.chunk.end_ts_ms,
                    sc.chunk.caption.as_deref().unwrap_or("no description"),
                )
            })
            .collect();

        let content = format!(
            "You are a security camera monitoring assistant. A user asked a question about camera footage.\n\n\
            User question: \"{query}\"\n\n\
            You are given {n_frames} camera frames from the relevant time periods, plus these metadata observations:\n\
            {obs}\n\n\
            Instructions:\n\
            - Answer the user's SPECIFIC question directly. Do NOT describe the images.\n\
            - Start with YES or NO if it's a yes/no question.\n\
            - Cite the camera ID and time when relevant, e.g. [cam-local 2:03 PM].\n\
            - Keep your answer to 1-3 sentences maximum.\n\
            - Do NOT think out loud or use any reasoning channel.\n\n\
            Answer:",
            n_frames = image_paths.len(),
            obs = observations.join("\n"),
        );

        let messages = if image_paths.is_empty() {
            text_messages(&content)
        } else {
            vision_messages(&content, &image_paths)
        };

        let model = Arc::clone(&self.model);
        let n_images = image_paths.len();
        info!(n_chunks = chunks.len(), n_images, "synthesize_answer: invoking gemma");
        let t0 = Instant::now();
        let raw = tokio::task::spawn_blocking(move || {
            // Keep temp files alive until Cactus finishes reading them.
            let _keep = temp_files;
            model.complete(&messages, Some(r#"{"max_tokens":1024}"#))
        })
        .await
        .context("synthesis task panicked")?
        .context("cactus synthesis failed")?;
        info!(elapsed_ms = t0.elapsed().as_millis() as u64, "synthesize_answer: gemma returned");

        let response = extract_response(&raw);
        if response.trim().is_empty() {
            warn!("synthesize_answer: empty response from gemma");
        }
        Ok(response)
    }
}

fn text_messages(content: &str) -> String {
    let escaped = content
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!(r#"[{{"role":"user","content":"{escaped}"}}]"#)
}

fn vision_messages(content: &str, image_paths: &[String]) -> String {
    json!([{
        "role": "user",
        "content": content,
        "images": image_paths,
    }])
    .to_string()
}

fn extract_response(raw: &str) -> String {
    let body = serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|v| v["response"].as_str().map(String::from))
        .unwrap_or_else(|| raw.to_string());
    strip_thinking(&body).to_string()
}

/// Gemma 4 emits `<|channel>thought ... <|channel>final` style preambles.
/// Return the substring after the last `<|channel>` marker so callers see
/// only the final-answer text.
fn strip_thinking(text: &str) -> &str {
    if let Some(idx) = text.rfind("<|channel>") {
        let rest = &text[idx..];
        if let Some(nl) = rest.find('\n') {
            return rest[nl + 1..].trim_start();
        }
    }
    text
}

fn find_json(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end > start).then(|| &text[start..=end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_json_extracts_embedded_object() {
        let text = r#"Sure! Here is the JSON: {"top_k":20} done."#;
        assert_eq!(find_json(text), Some(r#"{"top_k":20}"#));
    }

    #[test]
    fn find_json_returns_none_on_no_braces() {
        assert_eq!(find_json("no json here"), None);
    }

    #[test]
    fn extract_response_unwraps_cactus_json() {
        let raw = r#"{"response":"hello world","timings":{}}"#;
        assert_eq!(extract_response(raw), "hello world");
    }

    #[test]
    fn extract_response_falls_back_to_raw() {
        assert_eq!(extract_response("plain text"), "plain text");
    }
}
