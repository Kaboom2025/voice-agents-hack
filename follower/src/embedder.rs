//! Embedding strategies. The `Embedder` trait hides whether we're
//! calling real Cactus/Gemma or producing a deterministic synthetic
//! vector — the transport layer doesn't know or care.
//!
//! PRD §5.1: each 5 s chunk samples K=4 evenly-spaced frames and the
//! audio segment. PRD §5.2: the video embedding is mean-pooled across
//! K frames, the audio embedding is produced separately, and the final
//! vector is `[video_emb || audio_emb]`, L2-normalized.

#[cfg(feature = "cactus")]
use std::path::Path;
#[cfg(any(feature = "cactus", test))]
use std::sync::Arc;

#[cfg(feature = "cactus")]
use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use rand::{rngs::StdRng, RngCore, SeedableRng};

#[cfg(feature = "cactus")]
use crate::cactus::{l2_normalize, mean_pool, CactusModel};
use crate::camera::CapturedFrame;

// Local copy of l2_normalize so the synthetic path doesn't depend on the
// cactus module (which is feature-gated and may not be compiled in).
#[cfg(not(feature = "cactus"))]
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Gemma-4-E2B hidden dimension — target for mean-pooled image embeddings.
/// Chosen so image vectors match text embed dim for hybrid retrieval.
pub const GEMMA4_HIDDEN_DIM: usize = 1536;

/// Gemini embedding dimension — output from models/gemini-embedding-2-preview.
pub const GEMINI_EMBED_DIM: usize = 3072;

/// Input to the embedder: K frames from a 5 s window plus the audio
/// segment captured during that window.
pub struct ChunkInput {
    /// K evenly-spaced frames sampled from the chunk window.
    pub frames: Vec<CapturedFrame>,
    /// 16 kHz mono f32 audio samples for the chunk window. Empty if no
    /// microphone is available.
    pub audio_samples: Vec<f32>,
}

/// Turns a chunk (multiple frames + audio) into an embedding vector +
/// an optional caption.
///
/// Async because some embedders (Gemini) call out over HTTP. CPU-bound
/// implementations (Cactus) should offload the heavy work to
/// `tokio::task::spawn_blocking` internally.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Dimensionality this embedder produces. 0 means "variable".
    fn dim(&self) -> usize;

    async fn embed_chunk(&self, input: &ChunkInput, seq: u64) -> Result<EmbeddingOutput>;
}

pub struct EmbeddingOutput {
    pub embedding: Vec<f32>,
    /// Dimensionality of the video portion of the embedding.
    pub video_dim: usize,
    /// Dimensionality of the audio portion (0 when audio is unavailable).
    pub audio_dim: usize,
    pub caption: Option<String>,
}

// --- Cactus-backed embedder -----------------------------------------

#[cfg(feature = "cactus")]
#[derive(Clone)]
pub struct CactusEmbedder {
    model: Arc<CactusModel>,
    tmp_dir: std::path::PathBuf,
    file_prefix: String,
    jpeg_quality: u8,
}

#[cfg(feature = "cactus")]
impl CactusEmbedder {
    pub fn new(model: Arc<CactusModel>) -> Self {
        Self {
            model,
            tmp_dir: std::env::temp_dir(),
            file_prefix: "follower-frame".to_string(),
            jpeg_quality: 85,
        }
    }

    pub fn with_tmp_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.tmp_dir = dir.into();
        self
    }

    pub fn with_file_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.file_prefix = prefix.into();
        self
    }

    fn write_jpeg(&self, frame: &CapturedFrame, path: &Path) -> Result<()> {
        use image::{codecs::jpeg::JpegEncoder, ExtendedColorType};
        use std::fs::File;
        let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
        let mut encoder =
            JpegEncoder::new_with_quality(std::io::BufWriter::new(file), self.jpeg_quality);
        encoder
            .encode(
                frame.rgb.as_slice(),
                frame.width,
                frame.height,
                ExtendedColorType::Rgb8,
            )
            .with_context(|| format!("encode jpeg at {}", path.display()))?;
        Ok(())
    }

    /// Embed K frames: embed each independently, then mean-pool across
    /// all K per-frame embeddings to get a single video vector.
    fn embed_frames(&self, frames: &[CapturedFrame], seq: u64) -> Result<Vec<f32>> {
        let mut per_frame_pooled: Vec<Vec<f32>> = Vec::with_capacity(frames.len());
        for (i, frame) in frames.iter().enumerate() {
            let path = self
                .tmp_dir
                .join(format!("{}-{seq:06}-f{i}.jpg", self.file_prefix));
            self.write_jpeg(frame, &path)?;
            let raw = self
                .model
                .embed_image(&path)
                .with_context(|| format!("cactus image embed frame {i}"))?;
            let pooled = mean_pool(&raw, GEMMA4_HIDDEN_DIM)
                .context("mean pool per-frame image embedding")?;
            per_frame_pooled.push(pooled);
        }
        // Mean-pool across K frames → single [GEMMA4_HIDDEN_DIM] video vec.
        let k = per_frame_pooled.len();
        let mut video_emb = vec![0f32; GEMMA4_HIDDEN_DIM];
        for p in &per_frame_pooled {
            for (v, x) in video_emb.iter_mut().zip(p.iter()) {
                *v += *x;
            }
        }
        let inv = 1.0 / k as f32;
        for v in &mut video_emb {
            *v *= inv;
        }
        Ok(video_emb)
    }

    /// Embed the audio segment. Returns an empty vec if no audio.
    fn embed_audio(&self, samples: &[f32], seq: u64) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let wav_path = self
            .tmp_dir
            .join(format!("{}-{seq:06}.wav", self.file_prefix));
        crate::audio::write_wav(&wav_path, samples)?;
        let raw = self
            .model
            .embed_audio(&wav_path)
            .context("cactus audio embed")?;
        let pooled = mean_pool(&raw, GEMMA4_HIDDEN_DIM).context("mean pool audio embedding")?;
        Ok(pooled)
    }
}

#[cfg(feature = "cactus")]
impl CactusEmbedder {
    fn embed_chunk_blocking(&self, input: &ChunkInput, seq: u64) -> Result<EmbeddingOutput> {
        let mut video_emb = self.embed_frames(&input.frames, seq)?;
        let video_dim = video_emb.len();
        l2_normalize(&mut video_emb);

        let mut audio_emb = self.embed_audio(&input.audio_samples, seq)?;
        let audio_dim = audio_emb.len();
        if !audio_emb.is_empty() {
            l2_normalize(&mut audio_emb);
        }

        // Concatenate [video || audio] per PRD §5.2.
        let mut embedding = video_emb;
        embedding.extend_from_slice(&audio_emb);
        l2_normalize(&mut embedding);

        let k = input.frames.len();
        let audio_secs = if !input.audio_samples.is_empty() {
            input.audio_samples.len() as f32 / crate::audio::SAMPLE_RATE as f32
        } else {
            0.0
        };
        let caption = Some(format!("gemma-4 chunk: {k} frames, {audio_secs:.1}s audio"));

        Ok(EmbeddingOutput {
            embedding,
            video_dim,
            audio_dim,
            caption,
        })
    }
}

#[cfg(feature = "cactus")]
#[async_trait]
impl Embedder for CactusEmbedder {
    fn dim(&self) -> usize {
        0
    }

    async fn embed_chunk(&self, input: &ChunkInput, seq: u64) -> Result<EmbeddingOutput> {
        let this = self.clone();
        let frames = input.frames.clone();
        let audio_samples = input.audio_samples.clone();
        tokio::task::spawn_blocking(move || {
            let input = ChunkInput {
                frames,
                audio_samples,
            };
            this.embed_chunk_blocking(&input, seq)
        })
        .await
        .context("cactus embed task join")?
    }
}

// --- Synthetic (no model / no camera) -------------------------------

pub struct SyntheticEmbedder {
    dim: usize,
}

impl SyntheticEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn seed(seq: u64) -> u64 {
        seq.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xcbf29ce484222325
    }
}

#[async_trait]
impl Embedder for SyntheticEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed_chunk(&self, input: &ChunkInput, seq: u64) -> Result<EmbeddingOutput> {
        let mut rng = StdRng::seed_from_u64(Self::seed(seq));
        // Video portion.
        let video_dim = self.dim;
        let mut v = vec![0f32; video_dim];
        for slot in &mut v {
            *slot = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        // Audio portion — only if audio was provided.
        let audio_dim = if input.audio_samples.is_empty() {
            0
        } else {
            self.dim
        };
        let mut a = vec![0f32; audio_dim];
        for slot in &mut a {
            *slot = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        v.extend_from_slice(&a);
        l2_normalize(&mut v);
        Ok(EmbeddingOutput {
            embedding: v,
            video_dim,
            audio_dim,
            caption: Some(format!("synthetic #{seq}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_chunk(with_audio: bool) -> ChunkInput {
        let frame = CapturedFrame {
            width: 4,
            height: 4,
            rgb: Arc::new(vec![0u8; 4 * 4 * 3]),
        };
        ChunkInput {
            frames: vec![frame.clone(), frame.clone(), frame.clone(), frame],
            audio_samples: if with_audio {
                vec![0.0f32; 16000 * 5] // 5 s at 16 kHz
            } else {
                Vec::new()
            },
        }
    }

    #[tokio::test]
    async fn synthetic_embedding_is_l2_normalized() {
        let e = SyntheticEmbedder::new(128);
        let out = e.embed_chunk(&dummy_chunk(false), 0).await.unwrap();
        assert_eq!(out.embedding.len(), 128);
        assert_eq!(out.video_dim, 128);
        assert_eq!(out.audio_dim, 0);
        let norm: f32 = out.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn synthetic_with_audio_has_both_dims() {
        let e = SyntheticEmbedder::new(128);
        let out = e.embed_chunk(&dummy_chunk(true), 0).await.unwrap();
        assert_eq!(out.embedding.len(), 256);
        assert_eq!(out.video_dim, 128);
        assert_eq!(out.audio_dim, 128);
        let norm: f32 = out.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn synthetic_embedding_is_deterministic() {
        let e = SyntheticEmbedder::new(16);
        let a = e.embed_chunk(&dummy_chunk(false), 42).await.unwrap();
        let b = e.embed_chunk(&dummy_chunk(false), 42).await.unwrap();
        assert_eq!(a.embedding, b.embedding);
    }

    #[tokio::test]
    async fn synthetic_embedding_differs_across_seq() {
        let e = SyntheticEmbedder::new(16);
        let a = e.embed_chunk(&dummy_chunk(false), 0).await.unwrap();
        let b = e.embed_chunk(&dummy_chunk(false), 1).await.unwrap();
        assert_ne!(a.embedding, b.embedding);
    }
}
