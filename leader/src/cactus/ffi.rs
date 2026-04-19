//! Hand-declared Cactus C ABI — only the symbols the leader calls.
//!
//! Kept in lockstep with `cactus/ffi/cactus_ffi.h`. If Cactus upgrades,
//! the signatures here must be updated too.

use std::os::raw::{c_char, c_float, c_int, c_void};

pub type CactusModel = *mut c_void;

#[allow(dead_code)]
pub type CactusTokenCallback =
    Option<unsafe extern "C" fn(token: *const c_char, token_id: u32, user_data: *mut c_void)>;

unsafe extern "C" {
    pub fn cactus_init(
        model_path: *const c_char,
        corpus_dir: *const c_char,
        cache_index: bool,
    ) -> CactusModel;

    pub fn cactus_destroy(model: CactusModel);

    /// Text embedding. `buf` must have room for at least `buf_size`
    /// bytes. On success, `*dim_out` is set to the actual dimension.
    /// Returns 0 on success, negative on failure.
    pub fn cactus_embed(
        model: CactusModel,
        text: *const c_char,
        buf: *mut c_float,
        buf_size: usize,
        dim_out: *mut usize,
        normalize: bool,
    ) -> c_int;

    /// Chat completion. Writes a NUL-terminated JSON response string
    /// into `response_buffer`. Returns 0+ on success, negative on error.
    pub fn cactus_complete(
        model: CactusModel,
        messages_json: *const c_char,
        response_buffer: *mut c_char,
        buffer_size: usize,
        options_json: *const c_char,
        tools_json: *const c_char,
        callback: CactusTokenCallback,
        user_data: *mut c_void,
        pcm_buffer: *const u8,
        pcm_buffer_size: usize,
    ) -> c_int;

    pub fn cactus_get_last_error() -> *const c_char;
}
