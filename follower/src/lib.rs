//! Follower library — camera + Cactus/Gemini pipeline that pushes
//! `EmbeddingChunk`s to a leader over iroh QUIC.
//!
//! Subsystems:
//! - [`camera`] — webcam capture thread (nokhwa).
//! - [`cactus`] — safe wrapper over the Cactus C ABI.
//! - [`embedder`] — ties the two together behind a trait.

pub mod audio;
#[cfg(feature = "cactus")]
pub mod cactus;
pub mod camera;
pub mod embedder;
pub mod frame_buffer;
pub mod gemini_client;
pub mod gemini_embedder;
pub mod recorder;
