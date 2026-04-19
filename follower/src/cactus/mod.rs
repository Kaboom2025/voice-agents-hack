//! Safe wrapper over the Cactus C ABI. See [`ffi`] for the raw
//! declarations; this module exposes a poison-safe, Drop-clean handle
//! plus the two embed methods the follower actually calls.
//!
//! Thread model: Cactus's C state is not `Sync`. We guard the handle
//! behind a [`Mutex`] so the async runtime can embed from any task, and
//! we mark the wrapper `Send + Sync` on that basis. Embedding is a
//! long-running CPU operation — callers should invoke
//! [`CactusModel::embed_image`] from a `tokio::task::spawn_blocking`.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};

pub mod ffi;

/// Max embedding dimensionality Cactus can return. The Python bindings
/// hardcode 4096 floats; empirically on Gemma-4-E2B `cactus_image_embed`
/// returns the full pre-pooled vision tensor (~393k f32s) and
/// `cactus_audio_embed` returns ~720k f32s (~2.75 MB). We size to
/// 1M floats (4 MB) so both image and audio embeds fit with headroom.
///
/// Cactus reports the required buffer in **bytes**; we pass
/// `EMBED_BUF_LEN * size_of::<f32>()` as the `buf_size` argument below.
const EMBED_BUF_LEN: usize = 1024 * 1024;

const EMBED_BUF_BYTES: usize = EMBED_BUF_LEN * std::mem::size_of::<f32>();

/// 64 KiB is plenty for a per-chunk caption (Gemma emits ≤80 tokens with
/// our prompt, ~400 bytes worst case). Matches the leader's buffer.
const RESPONSE_BUF_BYTES: usize = 64 * 1024;

/// An initialized Cactus model.
///
/// Internally holds the raw handle and serializes every FFI call to the
/// engine. Drops the model on scope exit.
pub struct CactusModel {
    inner: Mutex<ModelHandle>,
}

struct ModelHandle(ffi::CactusModel);

// SAFETY: the raw pointer is only ever touched through the Mutex, which
// provides exclusive access across threads.
unsafe impl Send for ModelHandle {}
unsafe impl Sync for ModelHandle {}

impl Drop for ModelHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: handle came from cactus_init and is not aliased
            // (Mutex ensures we're the only one touching it).
            unsafe { ffi::cactus_destroy(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }
}

impl CactusModel {
    /// Load the model rooted at `model_path` (directory of `.weights`
    /// files). `corpus_dir` is for Cactus's RAG index; we pass empty
    /// since the follower doesn't use it.
    pub fn new(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            bail!("cactus model path does not exist: {}", model_path.display());
        }
        let model_cstr = CString::new(
            model_path
                .to_str()
                .context("model path is not valid utf-8")?,
        )?;
        let corpus_cstr = CString::new("")?;

        // SAFETY: both pointers are valid for the duration of the call.
        // Cactus copies the path internally before returning.
        let handle = unsafe { ffi::cactus_init(model_cstr.as_ptr(), corpus_cstr.as_ptr(), false) };

        if handle.is_null() {
            return Err(anyhow!(
                "cactus_init failed: {}",
                last_error().unwrap_or_else(|| "<no error message>".into())
            ));
        }
        Ok(Self {
            inner: Mutex::new(ModelHandle(handle)),
        })
    }

    /// Embed `text` into a float vector. If `normalize` is true, Cactus
    /// L2-normalizes before returning.
    pub fn embed_text(&self, text: &str, normalize: bool) -> Result<Vec<f32>> {
        let text_cstr = CString::new(text)?;
        let mut buf = vec![0f32; EMBED_BUF_LEN];
        let mut dim: usize = 0;

        let rc = {
            let guard = self.inner.lock().unwrap();
            // SAFETY: we hold the mutex, the handle is non-null (checked
            // on construction), and `buf` / `dim` are valid for the
            // duration of the call.
            unsafe {
                ffi::cactus_embed(
                    guard.0,
                    text_cstr.as_ptr(),
                    buf.as_mut_ptr(),
                    EMBED_BUF_BYTES,
                    &mut dim,
                    normalize,
                )
            }
        };
        if rc < 0 {
            return Err(anyhow!(
                "cactus_embed rc={rc}: {}",
                last_error().unwrap_or_else(|| "<no error message>".into())
            ));
        }
        buf.truncate(dim);
        Ok(buf)
    }

    /// Embed an image on disk. The path must be a JPEG/PNG Cactus can
    /// decode. For frames we encode to a temp JPEG first.
    pub fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>> {
        let path_cstr = CString::new(
            image_path
                .to_str()
                .context("image path is not valid utf-8")?,
        )?;
        let mut buf = vec![0f32; EMBED_BUF_LEN];
        let mut dim: usize = 0;

        let rc = {
            let guard = self.inner.lock().unwrap();
            // SAFETY: mutex-held, handle non-null, buf/dim valid.
            unsafe {
                ffi::cactus_image_embed(
                    guard.0,
                    path_cstr.as_ptr(),
                    buf.as_mut_ptr(),
                    EMBED_BUF_BYTES,
                    &mut dim,
                )
            }
        };
        if rc < 0 {
            return Err(anyhow!(
                "cactus_image_embed rc={rc}: {}",
                last_error().unwrap_or_else(|| "<no error message>".into())
            ));
        }
        buf.truncate(dim);
        Ok(buf)
    }

    /// Embed an audio file on disk. The path must be a WAV file (16 kHz
    /// mono PCM). Returns the raw pre-pooled audio representation.
    pub fn embed_audio(&self, audio_path: &Path) -> Result<Vec<f32>> {
        let path_cstr = CString::new(
            audio_path
                .to_str()
                .context("audio path is not valid utf-8")?,
        )?;
        let mut buf = vec![0f32; EMBED_BUF_LEN];
        let mut dim: usize = 0;

        let rc = {
            let guard = self.inner.lock().unwrap();
            // SAFETY: mutex-held, handle non-null, buf/dim valid.
            unsafe {
                ffi::cactus_audio_embed(
                    guard.0,
                    path_cstr.as_ptr(),
                    buf.as_mut_ptr(),
                    EMBED_BUF_BYTES,
                    &mut dim,
                )
            }
        };
        if rc < 0 {
            return Err(anyhow!(
                "cactus_audio_embed rc={rc}: {}",
                last_error().unwrap_or_else(|| "<no error message>".into())
            ));
        }
        buf.truncate(dim);
        Ok(buf)
    }

    /// Run a chat completion. `messages_json` is a JSON array of
    /// `{"role":..., "content":..., "images": [...]}` entries. Returns
    /// the raw NUL-terminated JSON response Cactus emits (contains
    /// `response`, timing stats, etc). Used by the follower to produce
    /// per-chunk captions that are then text-embedded into the shared
    /// text-encoder cosine space.
    pub fn complete(&self, messages_json: &str, options_json: Option<&str>) -> Result<String> {
        use std::os::raw::c_char;
        use std::ptr;

        let msgs_c = CString::new(messages_json).context("messages contain NUL byte")?;
        let opts_c = options_json
            .map(CString::new)
            .transpose()
            .context("options contain NUL byte")?;
        let opts_ptr = opts_c.as_ref().map_or(ptr::null(), |c| c.as_ptr());

        let mut buf = vec![0 as c_char; RESPONSE_BUF_BYTES];

        let rc = {
            let guard = self.inner.lock().unwrap();
            // SAFETY: mutex-held, handle non-null, buf is valid for
            // the duration of the call.
            unsafe {
                ffi::cactus_complete(
                    guard.0,
                    msgs_c.as_ptr(),
                    buf.as_mut_ptr(),
                    buf.len(),
                    opts_ptr,
                    ptr::null(),
                    None,
                    ptr::null_mut(),
                    ptr::null(),
                    0,
                )
            }
        };

        if rc < 0 {
            return Err(anyhow!(
                "cactus_complete rc={rc}: {}",
                last_error().unwrap_or_else(|| "<no error message>".into())
            ));
        }

        // Buffer is NUL-terminated by Cactus.
        let response = unsafe { CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned() };
        Ok(response)
    }
}

fn last_error() -> Option<String> {
    // SAFETY: Cactus returns either null or a pointer to a static/owned
    // C string valid until the next FFI call. We copy it immediately.
    unsafe {
        let p = ffi::cactus_get_last_error();
        if p.is_null() {
            None
        } else {
            Some(CStr::from_ptr(p).to_string_lossy().into_owned())
        }
    }
}

/// Extract the final answer text from a raw `cactus_complete` JSON
/// response. Unwraps the `{"response":"..."}` envelope (when present)
/// and strips Harmony-style `<|channel|>analysis<|message|>...` preambles
/// so the caller only sees the assistant's final message. Mirrors the
/// leader's `extract_response`+`strip_thinking` pair to keep the two
/// crates from drifting.
pub fn extract_final_text(raw: &str) -> String {
    let body = serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|v| v["response"].as_str().map(String::from))
        .unwrap_or_else(|| raw.to_string());
    strip_thinking(&body).to_string()
}

fn strip_thinking(text: &str) -> &str {
    let body: &str = if let Some(start) = text.rfind("<|channel|>final<|message|>") {
        &text[start + "<|channel|>final<|message|>".len()..]
    } else if let Some(start) = text.rfind("<|message|>") {
        &text[start + "<|message|>".len()..]
    } else if let Some(idx) = text.rfind("<|channel|>") {
        let rest = &text[idx..];
        match rest.find('\n') {
            Some(nl) => &rest[nl + 1..],
            None => rest,
        }
    } else if let Some(idx) = text.rfind("<|channel>") {
        let rest = &text[idx..];
        match rest.find('\n') {
            Some(nl) => &rest[nl + 1..],
            None => rest,
        }
    } else {
        text
    };

    let mut out = body.trim_start();
    for marker in ["<|end|>", "<|return|>", "<|endoftext|>"] {
        if let Some(i) = out.find(marker) {
            out = &out[..i];
        }
    }
    out.trim()
}

/// Mean-pool a flat `[N * D]` vector to `[D]` by averaging over N.
///
/// Gemma-4-E2B's image embedding is the pre-pooled vision tensor
/// (393216 = 256 patches × 1536 hidden dims). We pool across patches so
/// the shipped vector matches the text-embedding dim — essential for
/// hybrid retrieval (PRD §5.2).
pub fn mean_pool(flat: &[f32], hidden_dim: usize) -> Result<Vec<f32>> {
    if hidden_dim == 0 || flat.is_empty() {
        bail!("mean_pool: empty input or zero hidden_dim");
    }
    if flat.len() % hidden_dim != 0 {
        bail!(
            "mean_pool: {} not divisible by hidden_dim {}",
            flat.len(),
            hidden_dim
        );
    }
    let n_patches = flat.len() / hidden_dim;
    let mut pooled = vec![0f32; hidden_dim];
    for chunk in flat.chunks_exact(hidden_dim) {
        for (p, c) in pooled.iter_mut().zip(chunk.iter()) {
            *p += *c;
        }
    }
    let inv = 1.0 / n_patches as f32;
    for p in &mut pooled {
        *p *= inv;
    }
    Ok(pooled)
}

/// L2-normalize in place. Returns `true` when the vector was normalized,
/// `false` when it was left unchanged because the norm was zero or a
/// non-finite value appeared (NaN, ±inf). Silent-room audio frequently
/// yields an all-zero embedding; leaving that vector in-place (rather
/// than letting `0/0 = NaN` poison every downstream cosine) is the safe
/// thing to do.
pub fn l2_normalize(v: &mut [f32]) -> bool {
    // Reject anything non-finite up front — a single NaN from a buggy
    // upstream would otherwise propagate silently through cosine.
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

    #[test]
    fn l2_normalize_makes_unit_vector() {
        let mut v = vec![3.0, 4.0];
        assert!(l2_normalize(&mut v));
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_ignores_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        assert!(!l2_normalize(&mut v));
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_rejects_nan() {
        let mut v = vec![1.0, f32::NAN, 2.0];
        assert!(!l2_normalize(&mut v));
        // Input untouched on rejection.
        assert!(v[1].is_nan());
    }

    #[test]
    fn l2_normalize_rejects_infinity() {
        let mut v = vec![1.0, f32::INFINITY];
        assert!(!l2_normalize(&mut v));
    }

    #[test]
    fn mean_pool_averages_patches() {
        // two patches, hidden_dim=3
        let flat = vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0];
        let pooled = mean_pool(&flat, 3).unwrap();
        assert_eq!(pooled, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn mean_pool_rejects_mismatched_dim() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(mean_pool(&flat, 3).is_err());
    }

    #[test]
    fn extract_final_text_unwraps_harmony_preamble() {
        let raw = r#"{"response":"<|channel|>analysis<|message|>thinking...<|end|><|start|>assistant<|channel|>final<|message|>A cat on a sofa.<|return|>"}"#;
        assert_eq!(extract_final_text(raw), "A cat on a sofa.");
    }

    #[test]
    fn extract_final_text_falls_back_to_plain_response() {
        let raw = r#"{"response":"Just a plain caption."}"#;
        assert_eq!(extract_final_text(raw), "Just a plain caption.");
    }

    #[test]
    fn extract_final_text_handles_non_json_input() {
        assert_eq!(extract_final_text("raw string"), "raw string");
    }
}
