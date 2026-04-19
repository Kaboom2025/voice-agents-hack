//! Embedding strategies. The `Embedder` trait hides whether we're
//! calling Cactus/Gemma locally or the Gemini API — the transport layer
//! doesn't know or care.
//!
//! PRD §5.1: each 5 s chunk samples K=4 evenly-spaced frames and the
//! audio segment. PRD §5.2: the video embedding is mean-pooled across
//! K frames, the audio embedding is produced separately, and the final
//! vector is `[video_emb || audio_emb]`, L2-normalized.

#[cfg(feature = "cactus")]
use std::path::Path;
#[cfg(feature = "cactus")]
use std::sync::Arc;

#[cfg(feature = "cactus")]
use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;

#[cfg(feature = "cactus")]
use crate::cactus::{l2_normalize, mean_pool, CactusModel};
use crate::camera::CapturedFrame;

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


