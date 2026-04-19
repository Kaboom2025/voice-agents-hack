//! Safe wrapper over the Cactus C ABI for chat completion. Handle is
//! guarded by a `Mutex` so any async task can call through; the call is
//! CPU-heavy so callers should wrap in `spawn_blocking`.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};

pub mod ffi;

/// 64 KiB is plenty for a single chat response (Cactus caps at ~8k
/// tokens by default, ~32 KiB text worst case).
const RESPONSE_BUF_BYTES: usize = 64 * 1024;

pub struct CactusModel {
    inner: Mutex<ModelHandle>,
}

struct ModelHandle(ffi::CactusModel);

// SAFETY: the raw pointer is only ever touched through the Mutex.
unsafe impl Send for ModelHandle {}
unsafe impl Sync for ModelHandle {}

impl Drop for ModelHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: handle came from cactus_init; Mutex ensures we're
            // the only one touching it.
            unsafe { ffi::cactus_destroy(self.0) };
            self.0 = ptr::null_mut();
        }
    }
}

impl CactusModel {
    pub fn load(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            bail!("cactus model path does not exist: {}", model_path.display());
        }
        let model_cstr = CString::new(
            model_path
                .to_str()
                .context("model path is not valid utf-8")?,
        )?;

        // SAFETY: both pointers are valid for the duration of the call;
        // Cactus copies the strings internally.
        let handle = unsafe { ffi::cactus_init(model_cstr.as_ptr(), ptr::null(), false) };

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

    /// Run a chat completion. `messages_json` is a JSON array of
    /// `{"role": ..., "content": ...}`. Returns the raw JSON response
    /// string Cactus emits (containing `response`, timing stats, etc).
    pub fn complete(&self, messages_json: &str, options_json: Option<&str>) -> Result<String> {
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
                    ptr::null(),     // tools_json
                    None,            // no streaming callback
                    ptr::null_mut(), // user_data
                    ptr::null(),     // no audio
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

        // The buffer contains a NUL-terminated C string.
        let response = unsafe { CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned() };
        Ok(response)
    }
}

fn last_error() -> Option<String> {
    // SAFETY: Cactus returns either null or a static/owned C string
    // valid until the next FFI call. We copy it immediately.
    unsafe {
        let p = ffi::cactus_get_last_error();
        if p.is_null() {
            None
        } else {
            Some(CStr::from_ptr(p).to_string_lossy().into_owned())
        }
    }
}
