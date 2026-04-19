# Audio Capture + Joint Audio+Video Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture microphone audio in the follower and include it alongside JPEG video frames in a single Gemini `embedContent` request, producing one joint 3072-dim embedding per window.

**Architecture:** A new `AudioBuffer` (sliding-window PCM accumulator, same pattern as `FrameBuffer`) is populated by a `cpal` audio capture task. `GeminiEmbedClient::embed` is generalised to accept mixed `(mime_type, b64_data)` parts. `GeminiVideoEmbedder::embed_window` receives an optional `AudioBuffer` reference, encodes its PCM to an in-memory WAV, base64-encodes it, and appends it as an `audio/wav` part alongside the JPEG parts. The `VideoEmbedder` trait gains an `audio: Option<&AudioBuffer>` parameter. The main loop spawns a cpal input-stream task and passes the buffer into the embed loop.

**Tech Stack:** Rust, cpal 0.15 (cross-platform audio), base64 0.22 (already present), tokio mpsc, existing `FrameBuffer` / `GeminiEmbedClient` patterns.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `follower/Cargo.toml` | Modify | Add `cpal` dependency |
| `follower/src/audio_buffer.rs` | Create | Sliding-window PCM accumulator |
| `follower/src/lib.rs` | Modify | Export `audio_buffer` module |
| `follower/src/gemini_client.rs` | Modify | Generalise `embed()` to accept mixed media parts |
| `follower/src/gemini_embedder.rs` | Modify | WAV encoding helper + update trait + `GeminiVideoEmbedder` + `SyntheticVideoEmbedder` |
| `follower/src/main.rs` | Modify | Spawn audio capture task; pass `AudioBuffer` into embed loop |

---

### Task 1: Add `cpal` dependency

**Files:**
- Modify: `follower/Cargo.toml`

- [ ] **Step 1: Add the dependency**

In `follower/Cargo.toml`, add to `[dependencies]`:

```toml
cpal = "0.15"
```

- [ ] **Step 2: Verify it resolves**

```bash
cd /Users/saalik/Documents/Projects/voice-agents-hack
cargo check -p follower 2>&1 | head -30
```

Expected: no errors (cpal downloads and compiles).

- [ ] **Step 3: Commit**

```bash
git add follower/Cargo.toml Cargo.lock
git commit -m "feat: add cpal dependency for audio capture"
```

---

### Task 2: Create `AudioBuffer`

**Files:**
- Create: `follower/src/audio_buffer.rs`
- Modify: `follower/src/lib.rs`

- [ ] **Step 1: Write the failing test first**

Create `follower/src/audio_buffer.rs` with tests only (no impl yet):

```rust
use std::collections::VecDeque;
use std::sync::Mutex;

pub struct AudioBuffer {
    inner: Mutex<VecDeque<(u64, Vec<f32>)>>,
    window_ms: u64,
    sample_rate: Mutex<Option<u32>>,
}

impl AudioBuffer {
    pub fn new(window_ms: u64) -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
            window_ms,
            sample_rate: Mutex::new(None),
        }
    }

    /// Append a chunk of PCM samples captured at `ts_ms`.
    pub fn push(&self, ts_ms: u64, samples: Vec<f32>, sample_rate: u32) {
        *self.sample_rate.lock().unwrap() = Some(sample_rate);
        let mut q = self.inner.lock().unwrap();
        q.push_back((ts_ms, samples));
        if let Some(&(newest_ts, _)) = q.back() {
            let cutoff = newest_ts.saturating_sub(self.window_ms);
            while q.front().is_some_and(|&(ts, _)| ts < cutoff) {
                q.pop_front();
            }
        }
    }

    /// Return all PCM samples in the current window and the sample rate.
    /// Returns `None` if no samples have been pushed yet.
    pub fn drain(&self) -> Option<(Vec<f32>, u32)> {
        let q = self.inner.lock().unwrap();
        let sr = *self.sample_rate.lock().unwrap();
        let sr = sr?;
        if q.is_empty() {
            return None;
        }
        let samples: Vec<f32> = q.iter().flat_map(|(_, chunk)| chunk.iter().copied()).collect();
        Some((samples, sr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_returns_none() {
        let buf = AudioBuffer::new(5_000);
        assert!(buf.drain().is_none());
    }

    #[test]
    fn push_and_drain_returns_samples_and_rate() {
        let buf = AudioBuffer::new(5_000);
        buf.push(1000, vec![0.1, 0.2, 0.3], 16_000);
        let (samples, sr) = buf.drain().unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn chunks_older_than_window_are_pruned() {
        let buf = AudioBuffer::new(5_000);
        buf.push(0, vec![0.1], 16_000);
        buf.push(3_000, vec![0.2], 16_000);
        buf.push(6_000, vec![0.3], 16_000);
        let (samples, _) = buf.drain().unwrap();
        // chunk at t=0 is >5000ms before newest (t=6000), so pruned
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.2).abs() < 1e-6);
        assert!((samples[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn multiple_chunks_are_concatenated() {
        let buf = AudioBuffer::new(10_000);
        buf.push(1000, vec![0.1, 0.2], 44_100);
        buf.push(2000, vec![0.3, 0.4], 44_100);
        let (samples, sr) = buf.drain().unwrap();
        assert_eq!(sr, 44_100);
        assert_eq!(samples.len(), 4);
    }
}
```

- [ ] **Step 2: Export the module in `lib.rs`**

In `follower/src/lib.rs`, add:

```rust
pub mod audio_buffer;
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
cd /Users/saalik/Documents/Projects/voice-agents-hack
cargo test -p follower audio_buffer 2>&1
```

Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add follower/src/audio_buffer.rs follower/src/lib.rs
git commit -m "feat: add AudioBuffer sliding-window PCM accumulator"
```

---

### Task 3: Generalise `GeminiEmbedClient::embed` for mixed media parts

**Files:**
- Modify: `follower/src/gemini_client.rs`

The current signature is `embed(jpeg_b64_frames: Vec<String>)` which hard-codes `image/jpeg` as the MIME type. We need to support arbitrary `(mime_type, b64_data)` pairs so audio parts can be appended alongside images.

- [ ] **Step 1: Update `gemini_client.rs`**

Replace the entire `embed` method and add a new `MediaPart` type. The existing `Part` / `InlineData` structs are already correct — we just need to expose a public type and change the signature:

```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub const GEMINI_EMBED_MODEL: &str = "models/gemini-embedding-2-preview";
pub const GEMINI_EMBED_DIM: usize = 3072;
const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// A single content part for a Gemini embedContent request.
pub struct MediaPart {
    pub mime_type: String,
    pub data_b64: String,
}

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
    inline_data: InlineData,
}

#[derive(Serialize)]
struct InlineData {
    mime_type: String,
    data: String,
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

    pub async fn embed(&self, media_parts: Vec<MediaPart>) -> Result<Vec<f32>> {
        let parts: Vec<Part> = media_parts
            .into_iter()
            .map(|p| Part {
                inline_data: InlineData {
                    mime_type: p.mime_type,
                    data: p.data_b64,
                },
            })
            .collect();

        let req = EmbedRequest {
            model: GEMINI_EMBED_MODEL.to_string(),
            content: Content { parts },
        };

        let url = format!("{GEMINI_API_BASE}/{GEMINI_EMBED_MODEL}:embedContent");

        let resp = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("gemini HTTP request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|e| format!("<error reading body: {e}>"));
            anyhow::bail!("gemini API error {status}: {body}");
        }

        let embed_resp: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("gemini response parse failed: {e}"))?;

        Ok(embed_resp.embedding.values)
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

    #[test]
    fn serialise_embed_request_has_parts() {
        let req = EmbedRequest {
            model: GEMINI_EMBED_MODEL.to_string(),
            content: Content {
                parts: vec![
                    Part {
                        inline_data: InlineData {
                            mime_type: "image/jpeg".into(),
                            data: "abc".into(),
                        },
                    },
                    Part {
                        inline_data: InlineData {
                            mime_type: "audio/wav".into(),
                            data: "def".into(),
                        },
                    },
                ],
            },
        };
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains("\"parts\""));
        assert!(s.contains("image/jpeg"));
        assert!(s.contains("audio/wav"));
        assert!(s.contains("abc"));
        assert!(s.contains("def"));
    }

    #[tokio::test]
    async fn live_embed_single_frame() {
        let Ok(key) = std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("gemini_api_key"))
        else {
            eprintln!("GEMINI_API_KEY not set — skipping live test");
            return;
        };
        let jpeg_b64 = minimal_red_jpeg_b64();
        let client = GeminiEmbedClient::new(key);
        let result = client
            .embed(vec![MediaPart {
                mime_type: "image/jpeg".into(),
                data_b64: jpeg_b64,
            }])
            .await;
        match result {
            Ok(v) => {
                println!("embedding dim = {}", v.len());
                assert!(!v.is_empty());
            }
            Err(e) => {
                eprintln!("live embed error (may be expected if model not available): {e}");
            }
        }
    }

    fn minimal_red_jpeg_b64() -> String {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let bytes: &[u8] = &[
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06,
            0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D,
            0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D,
            0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28,
            0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01,
            0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
            0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10,
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42,
            0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
            0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
            0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73,
            0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
            0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
            0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
            0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
            0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08,
            0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD2, 0x8A, 0x28, 0x03, 0xFF, 0xD9,
        ];
        STANDARD.encode(bytes)
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/saalik/Documents/Projects/voice-agents-hack
cargo test -p follower gemini_client 2>&1
```

Expected: `deserialise_embed_response`, `serialise_embed_request_has_parts` pass. Live test skipped if no key.

- [ ] **Step 3: Commit**

```bash
git add follower/src/gemini_client.rs
git commit -m "feat: generalise GeminiEmbedClient::embed to accept mixed media parts"
```

---

### Task 4: Add WAV encoding and update `VideoEmbedder` trait + implementations

**Files:**
- Modify: `follower/src/gemini_embedder.rs`

This task:
1. Adds `encode_wav_b64(samples: &[f32], sample_rate: u32) -> String` — pure Rust, no extra crate
2. Changes `VideoEmbedder::embed_window` to accept `audio: Option<&AudioBuffer>`
3. Updates `GeminiVideoEmbedder::embed_window` to append the WAV part when audio is present
4. Updates `SyntheticVideoEmbedder::embed_window` to accept (and ignore) audio

- [ ] **Step 1: Replace `follower/src/gemini_embedder.rs` with the full updated file**

```rust
use anyhow::Result;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use crate::audio_buffer::AudioBuffer;
use crate::camera::CapturedFrame;
use crate::frame_buffer::FrameBuffer;
use crate::gemini_client::{GeminiEmbedClient, MediaPart};

pub struct VideoEmbeddingOutput {
    pub embedding: Vec<f32>,
    pub caption: Option<String>,
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
}

#[async_trait]
pub trait VideoEmbedder: Send + Sync {
    async fn embed_window(
        &self,
        frames: &FrameBuffer,
        audio: Option<&AudioBuffer>,
    ) -> Result<VideoEmbeddingOutput>;
}

pub struct GeminiVideoEmbedder {
    client: GeminiEmbedClient,
    target_fps: f32,
    jpeg_quality: u8,
}

pub struct SyntheticVideoEmbedder {
    dim: usize,
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Encode mono f32 PCM samples to a WAV byte vector (no extra crate).
fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let pcm: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();
    let data_size = pcm.len() * 2;
    let mut wav = Vec::with_capacity(44 + data_size);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&((36 + data_size) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_size as u32).to_le_bytes());
    for s in pcm {
        wav.extend_from_slice(&s.to_le_bytes());
    }
    wav
}

pub fn encode_wav_b64(samples: &[f32], sample_rate: u32) -> String {
    B64.encode(encode_wav(samples, sample_rate))
}

impl GeminiVideoEmbedder {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: GeminiEmbedClient::new(api_key),
            target_fps: 2.0,
            jpeg_quality: 80,
        }
    }

    pub fn encode_jpeg_bytes(frame: &CapturedFrame, quality: u8) -> Result<Vec<u8>> {
        use image::{codecs::jpeg::JpegEncoder, ExtendedColorType};
        let capacity = (frame.width as usize * frame.height as usize / 4).max(4096);
        let mut buf = Vec::with_capacity(capacity);
        JpegEncoder::new_with_quality(&mut buf, quality)
            .encode(
                frame.rgb.as_slice(),
                frame.width,
                frame.height,
                ExtendedColorType::Rgb8,
            )
            .map_err(|e| anyhow::anyhow!("jpeg encode: {e}"))?;
        Ok(buf)
    }

    pub fn encode_jpeg_b64(frame: &CapturedFrame, quality: u8) -> Result<String> {
        let bytes = Self::encode_jpeg_bytes(frame, quality)?;
        Ok(B64.encode(&bytes))
    }

    pub fn window_timestamps(buffer: &FrameBuffer) -> (u64, u64) {
        let w = buffer.window();
        match (w.first(), w.last()) {
            (Some((s, _)), Some((e, _))) => (*s, *e),
            _ => (0, 0),
        }
    }
}

#[async_trait]
impl VideoEmbedder for GeminiVideoEmbedder {
    async fn embed_window(
        &self,
        buffer: &FrameBuffer,
        audio: Option<&AudioBuffer>,
    ) -> Result<VideoEmbeddingOutput> {
        let (start_ts_ms, end_ts_ms) = Self::window_timestamps(buffer);
        let sampled = buffer.sample(self.target_fps);

        anyhow::ensure!(!sampled.is_empty(), "frame buffer is empty — cannot embed");

        let quality = self.jpeg_quality;
        let mut parts: Vec<MediaPart> = sampled
            .iter()
            .map(|f| {
                Self::encode_jpeg_b64(f, quality).map(|data_b64| MediaPart {
                    mime_type: "image/jpeg".into(),
                    data_b64,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let n_frames = parts.len();
        let mut has_audio = false;

        if let Some(audio_buf) = audio {
            if let Some((samples, sr)) = audio_buf.drain() {
                parts.push(MediaPart {
                    mime_type: "audio/wav".into(),
                    data_b64: encode_wav_b64(&samples, sr),
                });
                has_audio = true;
            }
        }

        let mut values = self.client.embed(parts).await?;
        l2_normalize(&mut values);

        Ok(VideoEmbeddingOutput {
            embedding: values,
            caption: Some(format!(
                "gemini-video window={}-{}ms frames={} audio={}",
                start_ts_ms, end_ts_ms, n_frames, has_audio
            )),
            start_ts_ms,
            end_ts_ms,
        })
    }
}

impl SyntheticVideoEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[async_trait]
impl VideoEmbedder for SyntheticVideoEmbedder {
    async fn embed_window(
        &self,
        buffer: &FrameBuffer,
        _audio: Option<&AudioBuffer>,
    ) -> Result<VideoEmbeddingOutput> {
        let (start_ts_ms, end_ts_ms) = GeminiVideoEmbedder::window_timestamps(buffer);
        let seed = start_ts_ms.wrapping_mul(0x9E3779B97F4A7C15);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v = vec![0f32; self.dim];
        for slot in &mut v {
            *slot = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        l2_normalize(&mut v);
        Ok(VideoEmbeddingOutput {
            embedding: v,
            caption: Some(format!("synthetic-video {start_ts_ms}-{end_ts_ms}ms")),
            start_ts_ms,
            end_ts_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn grey_frame(w: u32, h: u32, v: u8) -> CapturedFrame {
        CapturedFrame {
            width: w,
            height: h,
            rgb: Arc::new(vec![v; (w * h * 3) as usize]),
        }
    }

    #[test]
    fn encode_jpeg_bytes_produces_nonempty_output() {
        let frame = grey_frame(64, 64, 128);
        let bytes = GeminiVideoEmbedder::encode_jpeg_bytes(&frame, 85).unwrap();
        assert!(bytes.starts_with(&[0xFF, 0xD8]));
        assert!(bytes.len() > 100);
    }

    #[test]
    fn encode_jpeg_b64_is_valid_base64() {
        let frame = grey_frame(32, 32, 200);
        let b64 = GeminiVideoEmbedder::encode_jpeg_b64(&frame, 85).unwrap();
        let decoded = B64.decode(&b64).unwrap();
        assert!(decoded.starts_with(&[0xFF, 0xD8]));
    }

    #[test]
    fn window_timestamps_span_buffer_content() {
        let buf = FrameBuffer::new(10_000);
        buf.push(1000, grey_frame(4, 4, 10));
        buf.push(4000, grey_frame(4, 4, 20));
        buf.push(8000, grey_frame(4, 4, 30));
        let (start, end) = GeminiVideoEmbedder::window_timestamps(&buf);
        assert_eq!(start, 1000);
        assert_eq!(end, 8000);
    }

    #[test]
    fn window_timestamps_both_zero_on_empty_buffer() {
        let buf = FrameBuffer::new(10_000);
        let (start, end) = GeminiVideoEmbedder::window_timestamps(&buf);
        assert_eq!(start, 0);
        assert_eq!(end, 0);
    }

    #[test]
    fn encode_wav_starts_with_riff_header() {
        let samples = vec![0.0f32; 16_000]; // 1 second of silence at 16kHz
        let wav = encode_wav(&samples, 16_000);
        assert!(wav.starts_with(b"RIFF"));
        assert!(&wav[8..12] == b"WAVE");
        assert_eq!(wav.len(), 44 + samples.len() * 2);
    }

    #[test]
    fn encode_wav_b64_is_decodable() {
        let samples = vec![0.5f32, -0.5f32, 0.0f32];
        let b64 = encode_wav_b64(&samples, 44_100);
        let decoded = B64.decode(&b64).unwrap();
        assert!(decoded.starts_with(b"RIFF"));
    }

    #[tokio::test]
    async fn synthetic_embed_ignores_audio_buffer() {
        let buf = FrameBuffer::new(10_000);
        buf.push(0, grey_frame(4, 4, 10));
        buf.push(1000, grey_frame(4, 4, 20));
        let audio_buf = AudioBuffer::new(10_000);
        audio_buf.push(0, vec![0.1, 0.2], 16_000);
        let embedder = SyntheticVideoEmbedder::new(64);
        let out = embedder.embed_window(&buf, Some(&audio_buf)).await.unwrap();
        assert_eq!(out.embedding.len(), 64);
    }

    #[tokio::test]
    async fn live_embed_window_returns_3072_dim_embedding() {
        let Ok(key) = std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("gemini_api_key"))
        else {
            eprintln!("GEMINI_API_KEY not set — skipping live test");
            return;
        };
        let buf = FrameBuffer::new(10_000);
        for i in 0u64..3 {
            buf.push(i * 1000, grey_frame(64, 64, (50 + i * 60) as u8));
        }
        let embedder = GeminiVideoEmbedder::new(key);
        match embedder.embed_window(&buf, None).await {
            Ok(out) => {
                println!(
                    "dim={} start={} end={}",
                    out.embedding.len(),
                    out.start_ts_ms,
                    out.end_ts_ms
                );
                assert!(!out.embedding.is_empty());
                let norm: f32 = out.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-3,
                    "embedding should be L2-normalised, norm={norm}"
                );
            }
            Err(e) => eprintln!("live embed error: {e}"),
        }
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/saalik/Documents/Projects/voice-agents-hack
cargo test -p follower gemini_embedder 2>&1
```

Expected: `encode_jpeg_bytes_produces_nonempty_output`, `encode_jpeg_b64_is_valid_base64`, `window_timestamps_*`, `encode_wav_*`, `synthetic_embed_ignores_audio_buffer` all pass.

- [ ] **Step 3: Commit**

```bash
git add follower/src/gemini_embedder.rs
git commit -m "feat: add WAV encoding and update VideoEmbedder trait to accept optional AudioBuffer"
```

---

### Task 5: Update `main.rs` — fix call sites, spawn audio capture, pass AudioBuffer

**Files:**
- Modify: `follower/src/main.rs`

Changes needed:
1. `embed_window(&buf)` → `embed_window(&buf, Some(&audio_buffer))` (or `None` if `--no-audio`)
2. Add `--no-audio` CLI flag
3. Spawn cpal audio capture task that pushes chunks into `Arc<AudioBuffer>`

- [ ] **Step 1: Replace `follower/src/main.rs` with the updated version**

```rust
//! Follower CLI: webcam + microphone → Gemini multimodal embedding → iroh QUIC push.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::Parser;
use common::{
    read_frame, write_frame, EmbeddingChunk, FollowerMsg, LeaderMsg, Ticket, INGEST_ALPN,
};
use follower::audio_buffer::AudioBuffer;
use follower::camera::{self, CapturedFrame};
use follower::embedder::GEMINI_EMBED_DIM;
use follower::frame_buffer::FrameBuffer;
use follower::gemini_embedder::{
    GeminiVideoEmbedder, SyntheticVideoEmbedder, VideoEmbedder, VideoEmbeddingOutput,
};
use iroh::Endpoint;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

#[derive(Parser, Debug)]
#[command(about = "iroh follower: capture webcam + mic, embed via Gemini, push to leader")]
struct Args {
    /// Ticket string. If omitted, the follower reads it from `--ticket-file`.
    ticket: Option<String>,

    /// Path to a ticket file (the leader writes one on startup).
    #[arg(long, env = "LEADER_TICKET_FILE", default_value = ".leader.ticket")]
    ticket_file: PathBuf,

    /// Logical camera id (unique per follower).
    #[arg(long, default_value = "cam-0")]
    camera_id: String,

    /// Milliseconds between embedding steps.
    #[arg(long, default_value_t = 5000)]
    step_ms: u64,

    /// Sliding video window size in milliseconds.
    #[arg(long, default_value_t = 10_000)]
    window_ms: u64,

    /// Gemini API key. If set, uses GeminiVideoEmbedder. Falls back to synthetic.
    #[arg(long, env = "GEMINI_API_KEY")]
    gemini_api_key: Option<String>,

    /// Force synthetic random vectors regardless of API key availability.
    #[arg(long, default_value_t = false)]
    synthetic: bool,

    /// Stop after this many chunks. 0 = run forever.
    #[arg(long, default_value_t = 0)]
    count: u64,

    /// OS camera index (0 = default webcam).
    #[arg(long, default_value_t = 0)]
    device_index: u32,

    /// Skip the webcam and use a solid-color placeholder frame.
    #[arg(long, default_value_t = false)]
    no_camera: bool,

    /// Skip microphone capture (embed video only).
    #[arg(long, default_value_t = false)]
    no_audio: bool,

    /// Directory where captured JPEG frames are written.
    #[arg(long, env = "FOLLOWER_FRAME_DIR", default_value = "./frames")]
    frame_dir: PathBuf,

    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(&args.log)
        .with_target(false)
        .init();

    let ticket_str = match args.ticket.clone() {
        Some(t) => t,
        None => std::fs::read_to_string(&args.ticket_file)
            .with_context(|| {
                format!(
                    "no ticket given and ticket file {} not readable (is the leader running?)",
                    args.ticket_file.display()
                )
            })?
            .trim()
            .to_string(),
    };
    if ticket_str.is_empty() {
        bail!("ticket is empty");
    }

    let ticket: Ticket = ticket_str.parse().context("parse ticket")?;
    info!(leader = %ticket.leader.node_id, "dialing leader");

    let endpoint = Endpoint::builder().discovery_n0().bind().await?;
    let conn = endpoint.connect(ticket.leader, INGEST_ALPN).await?;
    info!("connected");
    let (mut send, mut recv) = conn.open_bi().await?;

    let (writer_tx, mut writer_rx) = mpsc::channel::<FollowerMsg>(128);
    let writer_task = tokio::spawn(async move {
        while let Some(msg) = writer_rx.recv().await {
            if let Err(e) = write_frame(&mut send, &msg).await {
                warn!(%e, "writer: send failed");
                break;
            }
        }
        let _ = send.finish();
    });

    writer_tx
        .send(FollowerMsg::Hello {
            camera_id: args.camera_id.clone(),
        })
        .await
        .context("send hello")?;

    std::fs::create_dir_all(&args.frame_dir)
        .with_context(|| format!("create frame dir {}", args.frame_dir.display()))?;
    info!(dir = %args.frame_dir.display(), "saving frames");

    let video_embedder = build_video_embedder(&args);
    let frame_buffer = Arc::new(FrameBuffer::new(args.window_ms));

    // --- Audio buffer (None if --no-audio) ---
    let audio_buffer: Option<Arc<AudioBuffer>> = if args.no_audio {
        info!("audio: disabled (--no-audio)");
        None
    } else {
        let buf = Arc::new(AudioBuffer::new(args.window_ms));
        let buf_clone = buf.clone();
        match spawn_audio_capture(buf_clone) {
            Ok(()) => {
                info!("audio: microphone capture started");
                Some(buf)
            }
            Err(e) => {
                warn!(error = %e, "audio capture failed to start, running video-only");
                None
            }
        }
    };

    let frames = if args.no_camera {
        info!("frame source: solid placeholder");
        FrameSource::Still(solid_placeholder())
    } else {
        match camera::spawn(args.device_index) {
            Ok(handle) => {
                info!(device = args.device_index, "webcam opened");
                FrameSource::Cam(handle.rx)
            }
            Err(e) => {
                warn!(error = %e, "camera open failed, using placeholder frame");
                FrameSource::Still(solid_placeholder())
            }
        }
    };

    let frames_for_reader = frames.clone();
    let writer_for_reader = writer_tx.clone();
    let reader_task = tokio::spawn(async move {
        loop {
            let msg: Option<LeaderMsg> = match read_frame(&mut recv).await {
                Ok(m) => m,
                Err(e) => {
                    warn!(%e, "reader: read failed");
                    break;
                }
            };
            let Some(msg) = msg else { break };
            match msg {
                LeaderMsg::Ack { chunk_id } => debug!(%chunk_id, "ack"),
                LeaderMsg::FrameRequest { req_id } => {
                    let frame = frames_for_reader.current();
                    let writer = writer_for_reader.clone();
                    tokio::spawn(async move {
                        let resp = match frame {
                            Some(f) => {
                                match tokio::task::spawn_blocking(move || encode_jpeg(&f, 85)).await
                                {
                                    Ok(Ok((jpeg, w, h))) => FollowerMsg::FrameResponse {
                                        req_id,
                                        ts_ms: now_ms(),
                                        width: w,
                                        height: h,
                                        jpeg,
                                    },
                                    Ok(Err(e)) => FollowerMsg::FrameError {
                                        req_id,
                                        message: format!("encode failed: {e}"),
                                    },
                                    Err(e) => FollowerMsg::FrameError {
                                        req_id,
                                        message: format!("encode task panicked: {e}"),
                                    },
                                }
                            }
                            None => FollowerMsg::FrameError {
                                req_id,
                                message: "no frame available yet".into(),
                            },
                        };
                        let _ = writer.send(resp).await;
                    });
                }
            }
        }
    });

    {
        let buf = frame_buffer.clone();
        let frames_clone = frames.clone();
        tokio::spawn(async move {
            match frames_clone {
                FrameSource::Still(f) => {
                    buf.push(now_ms(), f);
                }
                FrameSource::Cam(mut rx) => loop {
                    if rx.changed().await.is_err() {
                        break;
                    }
                    if let Some(frame) = rx.borrow().clone() {
                        buf.push(now_ms(), frame);
                    }
                },
            }
        });
    }

    let mut interval = tokio::time::interval(Duration::from_millis(args.step_ms));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut sent: u64 = 0;
    let mut stop = std::pin::pin!(tokio::signal::ctrl_c());

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let buf = frame_buffer.clone();
                let emb = video_embedder.clone();
                let audio_ref = audio_buffer.as_deref();
                let out: VideoEmbeddingOutput = match emb.embed_window(&buf, audio_ref).await {
                    Ok(o) => o,
                    Err(e) => { warn!(error = %e, "embed_window failed, skipping chunk"); continue; }
                };

                let chunk = EmbeddingChunk {
                    chunk_id: format!("{}-{}", args.camera_id, sent),
                    camera_id: args.camera_id.clone(),
                    start_ts_ms: out.start_ts_ms,
                    end_ts_ms: out.end_ts_ms,
                    embedding: out.embedding,
                    caption: out.caption,
                };
                let dim = chunk.embedding.len();
                if writer_tx.send(FollowerMsg::Chunk(chunk)).await.is_err() {
                    warn!("writer channel closed; exiting");
                    break;
                }
                sent += 1;
                info!(sent, dim, "chunk sent");
                if args.count != 0 && sent >= args.count {
                    break;
                }
            }
            _ = &mut stop => {
                info!("ctrl-c, stopping");
                break;
            }
        }
    }

    let _ = writer_tx.send(FollowerMsg::Bye).await;
    drop(writer_tx);
    let _ = writer_task.await;
    reader_task.abort();
    endpoint.close().await;
    Ok(())
}

/// Spawn a cpal audio input stream. Captured f32 samples are pushed into `buf`.
/// The stream is kept alive by leaking a `Box<dyn Any>` — it runs for the process lifetime.
fn spawn_audio_capture(buf: Arc<AudioBuffer>) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no default audio input device found"))?;
    let config = device
        .default_input_config()
        .context("get default input config")?;
    let sample_rate = config.sample_rate().0;

    let stream = device
        .build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                buf.push(now_ms(), data.to_vec(), sample_rate);
            },
            |e| warn!(error = %e, "audio input error"),
            None,
        )
        .context("build audio input stream")?;

    stream.play().context("start audio stream")?;

    // Leak the stream so it keeps running without needing an owner.
    Box::leak(Box::new(stream));
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn build_video_embedder(args: &Args) -> Arc<dyn VideoEmbedder> {
    if args.synthetic {
        info!("video embedder: synthetic (flag)");
        return Arc::new(SyntheticVideoEmbedder::new(GEMINI_EMBED_DIM));
    }
    if let Some(ref key) = args.gemini_api_key {
        info!("video embedder: GeminiVideoEmbedder (gemini-embedding-2-preview)");
        return Arc::new(GeminiVideoEmbedder::new(key.clone()));
    }
    info!("video embedder: synthetic (no GEMINI_API_KEY)");
    Arc::new(SyntheticVideoEmbedder::new(GEMINI_EMBED_DIM))
}

fn encode_jpeg(frame: &CapturedFrame, quality: u8) -> Result<(Vec<u8>, u32, u32)> {
    use image::{codecs::jpeg::JpegEncoder, ExtendedColorType};
    let cap = (frame.width as usize) * (frame.height as usize) / 4;
    let mut buf = Vec::with_capacity(cap.max(64 * 1024));
    let mut encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .encode(
            frame.rgb.as_slice(),
            frame.width,
            frame.height,
            ExtendedColorType::Rgb8,
        )
        .context("jpeg encode")?;
    Ok((buf, frame.width, frame.height))
}

#[derive(Clone)]
enum FrameSource {
    Cam(tokio::sync::watch::Receiver<Option<CapturedFrame>>),
    Still(CapturedFrame),
}

impl FrameSource {
    fn current(&self) -> Option<CapturedFrame> {
        match self {
            FrameSource::Cam(rx) => rx.borrow().clone(),
            FrameSource::Still(f) => Some(f.clone()),
        }
    }
}

fn solid_placeholder() -> CapturedFrame {
    CapturedFrame {
        width: 64,
        height: 64,
        rgb: Arc::new(vec![128u8; 64 * 64 * 3]),
    }
}
```

- [ ] **Step 2: Run `cargo check` to verify it compiles**

```bash
cd /Users/saalik/Documents/Projects/voice-agents-hack
cargo check -p follower 2>&1
```

Expected: no errors. (cpal will compile for macOS using CoreAudio.)

- [ ] **Step 3: Run all follower tests**

```bash
cargo test -p follower 2>&1
```

Expected: all existing tests pass. Live tests skipped if no API key.

- [ ] **Step 4: Commit**

```bash
git add follower/src/main.rs
git commit -m "feat: spawn cpal audio capture and pass AudioBuffer into embed_window"
```

---

### Task 6: End-to-end smoke test

This task validates the full pipeline compiles and the logic runs correctly without hitting live APIs.

**Files:**
- No new files — run existing commands.

- [ ] **Step 1: Full test suite**

```bash
cd /Users/saalik/Documents/Projects/voice-agents-hack
cargo test -p follower -- --nocapture 2>&1 | grep -E "(test .* ok|FAILED|running)"
```

Expected: all unit tests pass (audio_buffer, frame_buffer, gemini_client, gemini_embedder). Live tests skipped.

- [ ] **Step 2: Dry-run follower with synthetic embedder and no audio**

In terminal 1, start leader:
```bash
cargo run -p leader 2>&1 &
sleep 2
```

In terminal 2, run follower with `--synthetic --no-audio --count 2`:
```bash
TICKET=$(cat .leader.ticket)
cargo run -p follower -- --synthetic --no-audio --count 2 "$TICKET"
```

Expected output includes:
```
chunk sent sent=1 dim=3072
chunk sent sent=2 dim=3072
```

- [ ] **Step 3: Dry-run follower with synthetic embedder AND audio**

```bash
cargo run -p follower -- --synthetic --count 2 "$TICKET"
```

Expected output: same `chunk sent` lines (audio is captured but synthetic embedder ignores it — verifies the audio capture task starts without crashing).

- [ ] **Step 4: Commit (final)**

```bash
git add -p  # review any stray changes
git commit -m "test: validate audio+video pipeline smoke test"
```

---

## Self-Review

**Spec coverage:**
- [x] `cpal` audio capture → `AudioBuffer` (Task 1 & 2)
- [x] WAV encoding inline, no extra crate (Task 4)
- [x] Mixed `image/jpeg` + `audio/wav` parts in single Gemini request (Task 3 & 4)
- [x] Single 3072-dim embedding output unchanged (existing `l2_normalize` + response parsing)
- [x] `--no-audio` flag for graceful degradation (Task 5)
- [x] Audio capture failure is non-fatal — falls back to video-only (Task 5)
- [x] Wire protocol unchanged — `EmbeddingChunk` struct is the same (Task 5)

**Placeholder scan:** None found — all steps include actual code.

**Type consistency:**
- `AudioBuffer` used in both `gemini_embedder.rs` (imported as `crate::audio_buffer::AudioBuffer`) and `main.rs` (imported as `follower::audio_buffer::AudioBuffer`) ✓
- `MediaPart` defined in `gemini_client.rs`, imported in `gemini_embedder.rs` as `crate::gemini_client::MediaPart` ✓
- `VideoEmbedder::embed_window` signature updated in both `GeminiVideoEmbedder` and `SyntheticVideoEmbedder` ✓
- `audio_ref = audio_buffer.as_deref()` gives `Option<&AudioBuffer>` matching the trait parameter ✓
