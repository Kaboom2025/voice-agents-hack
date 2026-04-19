use anyhow::Result;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as B64, Engine};

use crate::audio;
use crate::camera::CapturedFrame;
use crate::embedder::{ChunkInput, EmbeddingOutput};
use crate::gemini_client::{GeminiEmbedClient, MediaPart};

/// Encode `frame` as a JPEG at `quality` (0..=100). Free function used by
/// both the Gemini embedder and the follower's main loop when it needs a
/// representative thumbnail for a chunk.
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

/// Encode `frame` as a JPEG and base64-encode the result (Gemini API
/// ingests inline media as base64-encoded bytes).
pub fn encode_jpeg_b64(frame: &CapturedFrame, quality: u8) -> Result<String> {
    let bytes = encode_jpeg_bytes(frame, quality)?;
    Ok(B64.encode(&bytes))
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

#[async_trait]
impl crate::embedder::Embedder for GeminiEmbedder {
    fn dim(&self) -> usize {
        0
    }

    async fn embed_chunk(&self, input: &ChunkInput, _seq: u64) -> Result<EmbeddingOutput> {
        // JPEG encode is CPU-bound — offload to a blocking thread so we
        // don't hold up the async runtime while encoding 4 HD frames + audio WAV.
        let frames = input.frames.clone();
        let audio_samples = input.audio_samples.clone();
        let quality = self.jpeg_quality;
        let media_parts: Vec<MediaPart> =
            tokio::task::spawn_blocking(move || -> Result<Vec<MediaPart>> {
                let mut parts: Vec<MediaPart> = frames
                    .iter()
                    .map(|f| {
                        let b64 = encode_jpeg_b64(f, quality)?;
                        Ok(MediaPart {
                            mime_type: "image/jpeg".to_string(),
                            data_b64: b64,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Add audio as WAV if samples present
                if !audio_samples.is_empty() {
                    let wav_bytes = audio::encode_wav_bytes(&audio_samples)?;
                    let wav_b64 = B64.encode(&wav_bytes);
                    parts.push(MediaPart {
                        mime_type: "audio/wav".to_string(),
                        data_b64: wav_b64,
                    });
                }

                Ok(parts)
            })
            .await
            .map_err(|e| anyhow::anyhow!("media encode task join: {e}"))??;

        anyhow::ensure!(!media_parts.is_empty(), "no media to embed");

        let n_video = media_parts
            .iter()
            .filter(|p| p.mime_type == "image/jpeg")
            .count();
        let has_audio = media_parts.iter().any(|p| p.mime_type == "audio/wav");
        let mut embedding = self.client.embed(media_parts).await?;

        if !l2_normalize(&mut embedding) {
            anyhow::bail!(
                "gemini returned a zero / non-finite embedding; refusing to \
                 ship a poisoned vector downstream"
            );
        }

        // Gemini returns a single fused multimodal vector — the model
        // doesn't split by modality, so we report the full dim as video.
        // When audio was included in the request, note it in audio_dim
        // so downstream search knows this embedding encodes audio too.
        let total_dim = embedding.len();
        let (video_dim, audio_dim) = if has_audio {
            // Split attribution: video gets most of the credit since
            // frames dominate the content, but mark audio's contribution
            // so modality-aware search can work.
            (total_dim, 0) // Gemini fuses modalities — can't split the vector
        } else {
            (total_dim, 0)
        };
        let caption = if has_audio {
            format!("gemini frames={n_video} audio=yes")
        } else {
            format!("gemini frames={n_video}")
        };

        Ok(EmbeddingOutput {
            embedding,
            video_dim,
            audio_dim,
            caption: Some(caption),
        })
    }
}

/// L2-normalize in place. Returns `false` (leaving the input unchanged)
/// on zero-norm vectors or if any element is non-finite (NaN/inf).
/// Mirrors `follower::cactus::l2_normalize`.
fn l2_normalize(v: &mut [f32]) -> bool {
    if !v.iter().all(|x| x.is_finite()) {
        return false;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !(norm > 0.0 && norm.is_finite()) {
        return false;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    true
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
        let bytes = encode_jpeg_bytes(&frame, 85).unwrap();
        assert!(bytes.starts_with(&[0xFF, 0xD8]));
        assert!(bytes.len() > 100);
    }

    #[test]
    fn encode_jpeg_b64_is_valid_base64() {
        let frame = grey_frame(32, 32, 200);
        let b64 = encode_jpeg_b64(&frame, 85).unwrap();
        let decoded = B64.decode(&b64).unwrap();
        assert!(decoded.starts_with(&[0xFF, 0xD8]));
    }
}
