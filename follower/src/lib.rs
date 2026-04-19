//! Follower library — camera + Cactus pipeline that pushes
//! `EmbeddingChunk`s to a leader over iroh QUIC.
//!
//! Subsystems:
//! - [`camera`] — webcam capture thread (nokhwa).
//! - [`cactus`] — safe wrapper over the Cactus C ABI.
//! - [`embedder`] — ties the two together behind a trait so the
//!   synthetic path stays alive for tests / no-camera environments.

pub mod cactus;
pub mod camera;
pub mod embedder;
