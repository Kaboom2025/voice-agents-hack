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
/// returns the full pre-pooled vision tensor (~393k f32s). We size to
/// 512k floats (2 MB) so a single image embed fits with headroom.
///
/// Cactus reports the required buffer in **bytes**; we pass
/// `EMBED_BUF_LEN * size_of::<f32>()` as the `buf_size` argument below.
const EMBED_BUF_LEN: usize = 512 * 1024;

const EMBED_BUF_BYTES: usize = EMBED_BUF_LEN * std::mem::size_of::<f32>();

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

/// L2-normalize in place. Zero-norm vectors are left alone.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_makes_unit_vector() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_ignores_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
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
}
