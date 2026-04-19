use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
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
        // Defense-in-depth: re-strip thinking markers in case Cactus wrapped
        // the reasoning preamble inside the `response` field itself.
        let cleaned = strip_thinking(&response);
        let json_str = find_json(cleaned).unwrap_or(cleaned);
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

    pub async fn synthesize_answer(&self, query: &str, chunks: &[StoredChunk], _store: &EmbeddingStore) -> Result<String> {
        if chunks.is_empty() {
            return Ok("No camera footage found matching your query.".into());
        }

        // Answer from **captions only** — now that the follower produces
        // real Gemma-4 captions for every chunk (P0-1), we no longer
        // need to re-show the JPEGs here. Feeding the images back to
        // Gemma makes it ignore the question and just describe the
        // scene (e.g. "The image shows a man with glasses..."), which
        // is the failure mode we were debugging. Captions already
        // contain the semantic signal; trust them.
        let observations: Vec<String> = chunks
            .iter()
            .take(20)
            .enumerate()
            .map(|(i, sc)| {
                format!(
                    "{idx}. [{cam} {start}ms–{end}ms] {cap}",
                    idx = i + 1,
                    cam = sc.chunk.camera_id,
                    start = sc.chunk.start_ts_ms,
                    end = sc.chunk.end_ts_ms,
                    cap = sc.chunk.caption.as_deref().unwrap_or("(no caption)"),
                )
            })
            .collect();

        // NB: Cactus + Gemma-4 silently drops the user turn when a `system`
        // role precedes it (the model replies "please provide a question"),
        // so we inline the instructions into a single user message instead.
        let prompt = format!(
            "You are a security-camera monitoring assistant. Answer the user's \
            question using ONLY the provided captions (one per 5-second video \
            chunk). You never see images; trust the captions. If the captions \
            do not contain enough information to answer, say so plainly. Do \
            NOT think out loud or use any reasoning channel. Start yes/no \
            questions with YES or NO.\n\n\
            User question: \"{query}\"\n\n\
            Relevant camera captions ({n} chunks, most relevant first):\n\
            {obs}\n\n\
            Answer the question in 1–3 sentences. Cite chunk index and camera \
            id when relevant, e.g. \"(#2, cam-local)\". If the captions don't \
            mention anything matching the question, say \"No, the captured \
            footage does not show that.\"\n\n\
            Question again: \"{query}\"\n\
            Answer:",
            n = observations.len(),
            obs = observations.join("\n"),
        );

        let messages = text_messages(&prompt);

        let model = Arc::clone(&self.model);
        info!(n_chunks = chunks.len(), "synthesize_answer: invoking gemma (captions-only)");
        let t0 = Instant::now();
        let raw = tokio::task::spawn_blocking(move || {
            model.complete(&messages, Some(r#"{"max_tokens":512,"temperature":0.2}"#))
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

fn extract_response(raw: &str) -> String {
    let body = serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|v| v["response"].as_str().map(String::from))
        .unwrap_or_else(|| raw.to_string());
    strip_thinking(&body).to_string()
}

/// Gemma 4 emits Harmony-style reasoning preambles like
/// `<|channel|>analysis<|message|>...thought...<|end|><|start|>assistant<|channel|>final<|message|>...answer...`.
/// Return the substring after the last `<|channel|>final<|message|>`
/// (or the last `<|message|>` if that sentinel is missing), stripping any
/// trailing `<|end|>` / `<|return|>` markers. Falls back to the legacy
/// `<|channel>` (no trailing pipe) form we used to emit, and finally to
/// the input unchanged when no marker is present.
fn strip_thinking(text: &str) -> &str {
    // Prefer the Harmony `<|channel|>final<|message|>` anchor so we skip
    // any `analysis`/`commentary` channels Gemma emits first.
    let body: &str = if let Some(start) = text.rfind("<|channel|>final<|message|>") {
        let rest = &text[start + "<|channel|>final<|message|>".len()..];
        rest
    } else if let Some(start) = text.rfind("<|message|>") {
        &text[start + "<|message|>".len()..]
    } else if let Some(idx) = text.rfind("<|channel|>") {
        // No explicit message marker — fall back to "after the last channel".
        let rest = &text[idx..];
        match rest.find('\n') {
            Some(nl) => &rest[nl + 1..],
            None => rest,
        }
    } else if let Some(idx) = text.rfind("<|channel>") {
        // Legacy (no trailing pipe) form from earlier cactus builds.
        let rest = &text[idx..];
        match rest.find('\n') {
            Some(nl) => &rest[nl + 1..],
            None => rest,
        }
    } else {
        text
    };

    // Trim trailing end-of-message sentinels so JSON parsing works.
    let mut out = body.trim_start();
    for marker in ["<|end|>", "<|return|>", "<|endoftext|>"] {
        if let Some(i) = out.find(marker) {
            out = &out[..i];
        }
    }
    out.trim()
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

    #[test]
    fn strip_thinking_handles_harmony_final_channel() {
        let raw = "<|channel|>analysis<|message|>hmm, let me think...<|end|>\
                   <|start|>assistant<|channel|>final<|message|>\
                   {\"top_k\":20}<|return|>";
        assert_eq!(strip_thinking(raw), r#"{"top_k":20}"#);
    }

    #[test]
    fn strip_thinking_handles_legacy_no_pipe_marker() {
        let raw = "<|channel>final\n{\"top_k\":20}";
        assert_eq!(strip_thinking(raw), r#"{"top_k":20}"#);
    }

    #[test]
    fn strip_thinking_passthrough_when_no_markers() {
        assert_eq!(strip_thinking("hello world"), "hello world");
    }

    #[test]
    fn extract_response_unwraps_and_strips_thinking() {
        let raw = r#"{"response":"<|channel|>final<|message|>{\"top_k\":20}<|return|>"}"#;
        assert_eq!(extract_response(raw), r#"{"top_k":20}"#);
    }
}
