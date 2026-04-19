//! Link the leader against the installed Cactus dylib.
//!
//! Mirrors the follower's strategy (`follower/build.rs`) so the FFI
//! surface is declared by hand in `src/cactus/ffi.rs` instead of going
//! through `bindgen` / `cactus-sys`. Candidate library locations:
//!
//!   1. `$CACTUS_LIB_DIR`            — explicit override (dir containing libcactus.dylib)
//!   2. `$CACTUS_PREFIX/lib`         — homebrew-style layout
//!   3. `$HOME/cactus/cactus/build`  — the source-built layout on this laptop
//!   4. `/opt/homebrew/opt/cactus/lib` — homebrew default
//!
//! The `cactus` feature gates link + FFI; without it the leader falls
//! back to echoing the user's query in `/api/query`.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CACTUS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CACTUS_PREFIX");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CACTUS");

    if std::env::var_os("CARGO_FEATURE_CACTUS").is_none() {
        return;
    }

    let candidates: Vec<PathBuf> = {
        let mut v: Vec<PathBuf> = Vec::new();
        if let Some(d) = std::env::var_os("CACTUS_LIB_DIR") {
            v.push(PathBuf::from(d));
        }
        if let Some(p) = std::env::var_os("CACTUS_PREFIX") {
            v.push(PathBuf::from(p).join("lib"));
        }
        if let Some(home) = std::env::var_os("HOME") {
            v.push(PathBuf::from(home).join("cactus/cactus/build"));
        }
        v.push(PathBuf::from("/opt/homebrew/opt/cactus/lib"));
        v
    };

    let Some(lib_dir) = candidates
        .into_iter()
        .find(|d| d.join("libcactus.dylib").exists())
    else {
        println!(
            "cargo:warning=libcactus.dylib not found; set CACTUS_LIB_DIR or CACTUS_PREFIX"
        );
        return;
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cactus");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
