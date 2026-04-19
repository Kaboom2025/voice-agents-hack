use anyhow::Result;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use crate::camera::CapturedFrame;
use crate::embedder::{ChunkInput, EmbeddingOutput};
use crate::frame_buffer::FrameBuffer;
use crate::gemini_client::GeminiEmbedClient;

pub struct VideoEmbeddingOutput {
    pub embedding: Vec<f32>,
    pub caption: Option<String>,
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
}

#[async_trait]
pub trait VideoEmbedder: Send + Sync {
    async fn embed_window(&self, buffer: &FrameBuffer) -> Result<VideoEmbeddingOutput>;
}

pub struct GeminiVideoEmbedder {
    client: GeminiEmbedClient,
    target_fps: f32,
    jpeg_quality: u8,
}

pub struct SyntheticVideoEmbedder {
    dim: usize,
}

/// Implements the synchronous `Embedder` trait using the Gemini REST API.
/// Designed to be called from `tokio::task::spawn_blocking`.
pub struct GeminiEmbedder {
    client: GeminiEmbedClient,
    jpeg_quality: u8,
}

impl GeminiEmbedder {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: GeminiEmbedClient::new(api_key),
            jpeg_quality: 80,
        }
    }
}

impl crate::embedder::Embedder for GeminiEmbedder {
    fn dim(&self) -> usize {
        0
    }

    fn embed_chunk(&self, input: &ChunkInput, _seq: u64) -> Result<EmbeddingOutput> {
        let b64_frames: Vec<String> = input
            .frames
            .iter()
            .map(|f| GeminiVideoEmbedder::encode_jpeg_b64(f, self.jpeg_quality))
            .collect::<Result<Vec<_>>>()?;

        anyhow::ensure!(!b64_frames.is_empty(), "no frames to embed");

        let n = b64_frames.len();
        let handle = tokio::runtime::Handle::current();
        let mut embedding = handle.block_on(self.client.embed(b64_frames))?;

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }

        let video_dim = embedding.len();
        Ok(EmbeddingOutput {
            embedding,
            video_dim,
            audio_dim: 0,
            caption: Some(format!("gemini frames={n}")),
        })
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
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
    async fn embed_window(&self, buffer: &FrameBuffer) -> Result<VideoEmbeddingOutput> {
        let (start_ts_ms, end_ts_ms) = Self::window_timestamps(buffer);
        let sampled = buffer.sample(self.target_fps);

        anyhow::ensure!(!sampled.is_empty(), "frame buffer is empty — cannot embed");

        let quality = self.jpeg_quality;
        let b64_frames: Vec<String> = sampled
            .iter()
            .map(|f| Self::encode_jpeg_b64(f, quality))
            .collect::<Result<Vec<_>>>()?;

        let n_frames = b64_frames.len();
        let mut values = self.client.embed(b64_frames).await?;
        l2_normalize(&mut values);

        Ok(VideoEmbeddingOutput {
            embedding: values,
            caption: Some(format!(
                "gemini-video window={}-{}ms frames={}",
                start_ts_ms, end_ts_ms, n_frames
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
    async fn embed_window(&self, buffer: &FrameBuffer) -> Result<VideoEmbeddingOutput> {
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
        match embedder.embed_window(&buf).await {
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
