//! Embedding strategies. The `Embedder` trait hides whether we're
//! calling real Cactus/Gemma or producing a deterministic synthetic
//! vector — the transport layer doesn't know or care.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use crate::cactus::{l2_normalize, mean_pool, CactusModel};
use crate::camera::CapturedFrame;

/// Gemma-4-E2B hidden dimension — target for mean-pooled image embeddings.
/// Chosen so image vectors match text embed dim for hybrid retrieval.
pub const GEMMA4_HIDDEN_DIM: usize = 1536;

/// Turns a captured frame into an embedding vector + an optional caption.
pub trait Embedder: Send + Sync {
    /// Dimensionality this embedder produces. 0 means "variable" (Cactus
    /// returns ~2048 for gemma-4-E2B images; callers should use
    /// `.len()` on the actual output rather than relying on `dim`).
    fn dim(&self) -> usize;

    fn embed_frame(&self, frame: &CapturedFrame, seq: u64) -> Result<EmbeddingOutput>;
}

pub struct EmbeddingOutput {
    pub embedding: Vec<f32>,
    pub caption: Option<String>,
}

// --- Cactus-backed embedder -----------------------------------------

pub struct CactusEmbedder {
    model: Arc<CactusModel>,
    tmp_dir: std::path::PathBuf,
    jpeg_quality: u8,
}

impl CactusEmbedder {
    /// Create with a loaded Cactus model. JPEG intermediates are written
    /// to `std::env::temp_dir()` by default; override with [`with_tmp_dir`].
    pub fn new(model: Arc<CactusModel>) -> Self {
        Self {
            model,
            tmp_dir: std::env::temp_dir(),
            jpeg_quality: 85,
        }
    }

    pub fn with_tmp_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.tmp_dir = dir.into();
        self
    }

    /// Write an RGB frame as JPEG at `path`. JPEG (not PNG) to keep disk
    /// I/O small on the hot path — Cactus decodes either fine.
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
}

impl Embedder for CactusEmbedder {
    fn dim(&self) -> usize {
        0
    }

    fn embed_frame(&self, frame: &CapturedFrame, seq: u64) -> Result<EmbeddingOutput> {
        // Rotating two paths so we don't blow out tmpfs with long runs.
        let slot = seq % 2;
        let path = self.tmp_dir.join(format!("follower-frame-{slot}.jpg"));
        self.write_jpeg(frame, &path)?;

        let raw = self
            .model
            .embed_image(&path)
            .context("cactus image embed")?;
        // Gemma-4 returns the pre-pooled vision tensor. Mean-pool over
        // patches to get a fixed-size, shippable vector.
        let mut pooled =
            mean_pool(&raw, GEMMA4_HIDDEN_DIM).context("mean pool gemma-4 image embedding")?;
        l2_normalize(&mut pooled);

        Ok(EmbeddingOutput {
            embedding: pooled,
            caption: Some(format!(
                "gemma-4 img {}x{} patches={}",
                frame.width,
                frame.height,
                raw.len() / GEMMA4_HIDDEN_DIM
            )),
        })
    }
}

// --- Synthetic (no model / no camera) -------------------------------

/// Fallback that works without Cactus or a webcam. Used by tests and by
/// the follower binary when `--synthetic` is passed.
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

impl Embedder for SyntheticEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_frame(&self, _frame: &CapturedFrame, seq: u64) -> Result<EmbeddingOutput> {
        let mut rng = StdRng::seed_from_u64(Self::seed(seq));
        let mut v = vec![0f32; self.dim];
        for slot in &mut v {
            *slot = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        l2_normalize(&mut v);
        Ok(EmbeddingOutput {
            embedding: v,
            caption: Some(format!("synthetic #{seq}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_frame() -> CapturedFrame {
        CapturedFrame {
            width: 4,
            height: 4,
            rgb: Arc::new(vec![0u8; 4 * 4 * 3]),
        }
    }

    #[test]
    fn synthetic_embedding_is_l2_normalized() {
        let e = SyntheticEmbedder::new(128);
        let out = e.embed_frame(&dummy_frame(), 0).unwrap();
        assert_eq!(out.embedding.len(), 128);
        let norm: f32 = out.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn synthetic_embedding_is_deterministic() {
        let e = SyntheticEmbedder::new(16);
        let a = e.embed_frame(&dummy_frame(), 42).unwrap();
        let b = e.embed_frame(&dummy_frame(), 42).unwrap();
        assert_eq!(a.embedding, b.embedding);
    }

    #[test]
    fn synthetic_embedding_differs_across_seq() {
        let e = SyntheticEmbedder::new(16);
        let a = e.embed_frame(&dummy_frame(), 0).unwrap();
        let b = e.embed_frame(&dummy_frame(), 1).unwrap();
        assert_ne!(a.embedding, b.embedding);
    }
}
