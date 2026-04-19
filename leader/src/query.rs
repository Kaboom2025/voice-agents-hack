use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::json;

use crate::cactus::CactusModel;
use crate::store::StoredChunk;

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
        let raw = tokio::task::spawn_blocking(move || {
            model.complete(&messages, Some(r#"{"max_tokens":256}"#))
        })
        .await
        .context("parse task panicked")?
        .context("cactus parse failed")?;

        let response = extract_response(&raw);
        let json_str = find_json(&response).unwrap_or(&response);
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .with_context(|| format!("gemma returned non-JSON: {response}"))?;

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

        // Write representative JPEGs to temp files; Cactus reads them by path.
        let mut temp_files: Vec<tempfile::NamedTempFile> = Vec::new();
        let mut image_paths: Vec<String> = Vec::new();
        for (_sc, jpeg) in chunks
            .iter()
            .filter_map(|sc| sc.chunk.representative_jpeg.as_ref().map(|j| (sc, j)))
            .take(10)
        {
            let mut tmp = tempfile::Builder::new()
                .suffix(".jpg")
                .tempfile()
                .context("create temp jpeg")?;
            tmp.write_all(jpeg).context("write temp jpeg")?;
            image_paths.push(tmp.path().to_string_lossy().into_owned());
            temp_files.push(tmp);
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
            "You are a security monitoring AI. Review the footage and answer concisely.\n\n\
            Observations:\n{}\n\n\
            Question: {query}",
            observations.join("\n")
        );

        let messages = if image_paths.is_empty() {
            text_messages(&content)
        } else {
            vision_messages(&content, &image_paths)
        };

        let model = Arc::clone(&self.model);
        let raw = tokio::task::spawn_blocking(move || {
            // Keep temp files alive until Cactus finishes reading them.
            let _keep = temp_files;
            model.complete(&messages, Some(r#"{"max_tokens":1024}"#))
        })
        .await
        .context("synthesis task panicked")?
        .context("cactus synthesis failed")?;

        Ok(extract_response(&raw).to_string())
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
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|v| v["response"].as_str().map(String::from))
        .unwrap_or_else(|| raw.to_string())
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
