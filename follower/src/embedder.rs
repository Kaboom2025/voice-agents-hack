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
use crate::cactus::{extract_final_text, l2_normalize, CactusModel};
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

    /// Caption the representative frame via Gemma-4 vision, then embed
    /// the caption with the text encoder. This is the **only** vector we
    /// ship — pooled vision-tower / audio-encoder states are not in the
    /// same cosine space as the text encoder, so comparing them against
    /// text queries is near-random (see
    /// `docs/search-infrastructure-review.md` §2.1).
    fn caption_rep_frame(&self, frame: &CapturedFrame, seq: u64) -> Result<String> {
        let path = self
            .tmp_dir
            .join(format!("{}-{seq:06}-caption.jpg", self.file_prefix));
        self.write_jpeg(frame, &path)?;

        let messages = build_caption_messages(&path);
        let raw = self
            .model
            .complete(
                &messages,
                Some(r#"{"max_tokens":80,"temperature":0.2}"#),
            )
            .context("cactus_complete for caption failed")?;
        let text = extract_final_text(&raw);
        let cleaned = text.trim().trim_matches('"').trim().to_string();
        if cleaned.is_empty() {
            anyhow::bail!("gemma returned empty caption");
        }
        Ok(cleaned)
    }
}

/// Build a Cactus vision-message array requesting a one-sentence
/// description of the supplied JPEG. Kept as a free function so it's
/// trivial to unit-test the string shape without owning a model.
#[cfg(feature = "cactus")]
fn build_caption_messages(image_path: &Path) -> String {
    let path_str = image_path.to_string_lossy();
    serde_json::json!([
        {
            "role": "system",
            "content": "You are a security-camera captioner. Describe each scene \
                        in ONE concise sentence. Mention people, objects, actions, \
                        and notable visual details. Do NOT think out loud or use \
                        any reasoning channel. Output the caption only."
        },
        {
            "role": "user",
            "content": "Describe this scene in one sentence.",
            "images": [path_str],
        }
    ])
    .to_string()
}

#[cfg(feature = "cactus")]
impl CactusEmbedder {
    fn embed_chunk_blocking(&self, input: &ChunkInput, seq: u64) -> Result<EmbeddingOutput> {
        anyhow::ensure!(!input.frames.is_empty(), "no frames in chunk");

        // Caption the middle frame via Gemma-4 vision. This is the
        // *only* signal we turn into a retrieval vector — see §2.1 of
        // the search-infrastructure review for why pooled vision-tower
        // states were junk.
        let mid = input.frames.len() / 2;
        let rep_frame = &input.frames[mid];
        let caption = self.caption_rep_frame(rep_frame, seq)?;

        // Text-embed the caption through the same encoder used by the
        // leader at query time. Same space, well-posed cosine.
        let mut embedding = self
            .model
            .embed_text(&caption, /* normalize = */ true)
            .context("cactus_embed for caption failed")?;
        if embedding.is_empty() {
            anyhow::bail!("cactus_embed returned an empty caption vector");
        }
        // Belt-and-braces: Cactus already normalized, but guard against
        // any backend that quietly returns zero / non-finite values.
        if !l2_normalize(&mut embedding) {
            anyhow::bail!(
                "caption embedding was zero / non-finite; refusing to ship \
                 a vector that would poison downstream cosine"
            );
        }
        let video_dim = embedding.len();

        Ok(EmbeddingOutput {
            embedding,
            video_dim,
            audio_dim: 0,
            caption: Some(caption),
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


