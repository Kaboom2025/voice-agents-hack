//! Thin safe wrapper around `cactus-sys` for Gemma 4 chat completion.
//!
//! The model is loaded once at startup and can be called from any async task
//! via the thread-safe [`CactusModel`] handle.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};

/// Opaque handle to a loaded Cactus model (Gemma 4).
///
/// Internally guards the raw `cactus_model_t` with a mutex so callers can
/// share the handle across tasks without data races.
pub struct CactusModel {
    inner: Mutex<cactus_sys::cactus_model_t>,
}

// Manual Debug because cactus_model_t is an opaque pointer.
impl std::fmt::Debug for CactusModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CactusModel").finish_non_exhaustive()
    }
}

// The C library is thread-safe when accessed one call at a time, which the
// mutex guarantees.
unsafe impl Send for CactusModel {}
unsafe impl Sync for CactusModel {}

impl CactusModel {
    /// Load a model from a weights directory (e.g. the Gemma 4 E2B folder).
    ///
    /// `corpus_dir` is an optional path for RAG corpus indexing; pass `None`
    /// to skip.
    pub fn load(model_path: &Path, corpus_dir: Option<&Path>) -> Result<Self> {
        let model_path_c = CString::new(
            model_path
                .to_str()
                .context("model_path is not valid UTF-8")?,
        )?;

        let corpus_ptr = match corpus_dir {
            Some(p) => {
                let s = p.to_str().context("corpus_dir is not valid UTF-8")?;
                CString::new(s)?.into_raw() as *const _
            }
            None => ptr::null(),
        };

        let handle =
            unsafe { cactus_sys::cactus_init(model_path_c.as_ptr(), corpus_ptr, false) };

        // Free the corpus CString if we allocated one.
        if !corpus_ptr.is_null() {
            unsafe {
                drop(CString::from_raw(corpus_ptr as *mut _));
            }
        }

        if handle.is_null() {
            bail!("cactus_init failed for {}", model_path.display());
        }

        Ok(Self {
            inner: Mutex::new(handle),
        })
    }

    /// Run a chat completion and return the raw JSON response string.
    ///
    /// `messages_json` must be a JSON array of `{"role": "…", "content": "…"}`
    /// objects.  `options_json` is optional (temperature, max_tokens, etc.).
    pub fn complete(
        &self,
        messages_json: &str,
        options_json: Option<&str>,
    ) -> Result<String> {
        let msgs = CString::new(messages_json)?;
        let opts_c = options_json.map(CString::new).transpose()?;
        let opts_ptr = opts_c.as_ref().map_or(ptr::null(), |c| c.as_ptr());

        // 32 KiB response buffer — plenty for caption/summary work.
        let mut buf = vec![0i8; 32 * 1024];

        let handle = self.inner.lock().unwrap();
        let rc = unsafe {
            cactus_sys::cactus_complete(
                *handle,
                msgs.as_ptr(),
                buf.as_mut_ptr(),
                buf.len(),
                opts_ptr,
                ptr::null(), // no tools
                None,        // no streaming callback
                ptr::null_mut(),
                ptr::null(), // no audio
                0,
            )
        };

        if rc <= 0 {
            bail!("cactus_complete returned {rc}");
        }

        // The buffer contains a NUL-terminated C string.
        let response = unsafe {
            CStr::from_ptr(buf.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        Ok(response)
    }

    /// Generate a text embedding vector for the given text.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let text_c = CString::new(text)?;

        // Allocate a buffer large enough for typical embedding dims (up to 4096).
        let buf_size = 4096;
        let mut buf = vec![0f32; buf_size];
        let mut dim: usize = 0;

        let handle = self.inner.lock().unwrap();
        let rc = unsafe {
            cactus_sys::cactus_embed(
                *handle,
                text_c.as_ptr(),
                buf.as_mut_ptr(),
                buf_size,
                &mut dim as *mut usize,
                true, // L2-normalize
            )
        };

        if rc <= 0 || dim == 0 {
            bail!("cactus_embed returned {rc} (dim={dim})");
        }

        buf.truncate(dim);
        Ok(buf)
    }
}

impl Drop for CactusModel {
    fn drop(&mut self) {
        let handle = self.inner.lock().unwrap();
        if !handle.is_null() {
            unsafe {
                cactus_sys::cactus_destroy(*handle);
            }
        }
    }
}
