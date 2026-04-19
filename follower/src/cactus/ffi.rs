//! Hand-rolled Cactus C ABI declarations.
//!
//! Signatures cross-referenced against the Python `ctypes` bindings at
//! `/opt/homebrew/opt/cactus/libexec/python/src/cactus.py`. Keep these
//! in lockstep if Cactus upgrades.
//!
//! The brew-installed dylib exposes richer multimodal embedding APIs
//! than the public docs mention — notably `cactus_image_embed` and
//! `cactus_audio_embed`, which let us skip the caption-then-embed hop
//! from PRD §5.2.

use std::os::raw::{c_char, c_float, c_int, c_void};

/// Opaque handle returned by [`cactus_init`]. Pass to every other API.
pub type CactusModel = *mut c_void;

unsafe extern "C" {
    /// Load a model from disk. `model_path` is a directory holding the
    /// per-tensor `.weights` files (e.g. gemma-4-e2b-it). `corpus_dir`
    /// is for RAG; pass an empty C string if unused. `cache_index`
    /// controls whether the RAG index is persisted.
    pub fn cactus_init(
        model_path: *const c_char,
        corpus_dir: *const c_char,
        cache_index: bool,
    ) -> CactusModel;

    /// Free everything the model holds. Safe to pass null.
    pub fn cactus_destroy(model: CactusModel);

    /// Text embedding. `buf` must have room for at least `buf_size`
    /// floats. On success, `*dim_out` is set to the actual dimension.
    /// Returns 0 on success, negative on failure (check
    /// [`cactus_get_last_error`]).
    pub fn cactus_embed(
        model: CactusModel,
        text: *const c_char,
        buf: *mut c_float,
        buf_size: usize,
        dim_out: *mut usize,
        normalize: bool,
    ) -> c_int;

    /// Image embedding. `image_path` is a filesystem path to a PNG / JPEG.
    /// The embedder does its own vision-tower pool; no normalization
    /// knob is exposed — callers L2-normalize downstream if they want it.
    pub fn cactus_image_embed(
        model: CactusModel,
        image_path: *const c_char,
        buf: *mut c_float,
        buf_size: usize,
        dim_out: *mut usize,
    ) -> c_int;

    /// Audio embedding. `audio_path` is a filesystem path to a WAV file
    /// (16 kHz mono PCM expected). Returns the pre-pooled audio
    /// representation; callers mean-pool and L2-normalize downstream.
    pub fn cactus_audio_embed(
        model: CactusModel,
        audio_path: *const c_char,
        buf: *mut c_float,
        buf_size: usize,
        dim_out: *mut usize,
    ) -> c_int;

    /// Last error message set by any of the above calls. Pointer is
    /// owned by the lib; do not free. Valid until the next FFI call.
    pub fn cactus_get_last_error() -> *const c_char;
}
