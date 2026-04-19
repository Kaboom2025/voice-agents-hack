//! Link against the installed Cactus dylib.
//!
//! Cactus is shipped as a Homebrew formula that places its dylib under
//! `/opt/homebrew/opt/cactus/lib/libcactus.dylib` on Apple Silicon. We
//! hand-declare the C ABI in `src/cactus/ffi.rs` so we don't need bindgen
//! or the cactus-sys crate's full C++ build tree. Linker configuration:
//!
//!   -L<prefix>/lib        — where to find the dylib at link time
//!   -lcactus              — link name (libcactus.dylib)
//!   -Wl,-rpath,<prefix>   — embed rpath so the binary finds the dylib at
//!                           runtime without DYLD_LIBRARY_PATH
//!
//! The `CACTUS_PREFIX` env var overrides the default for non-Homebrew
//! installs or CI environments. Only Apple Silicon macOS is wired up here
//! — Linux/x86 users would need to adjust the default prefix.

use std::path::PathBuf;

fn main() {
    // Only the FFI surface depends on the C lib; other files don't need
    // a rebuild when we re-run build.rs.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CACTUS_PREFIX");

    let prefix: PathBuf = std::env::var_os("CACTUS_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/homebrew/opt/cactus"));

    let lib_dir = prefix.join("lib");
    let dylib = lib_dir.join("libcactus.dylib");

    if !dylib.exists() {
        // Don't fail hard — users without Cactus installed can still build
        // with `--features synthetic-only` (see follower/src/cactus/mod.rs).
        // Instead emit a warning and skip linkage; the extern calls will
        // be compiled but not linkable at runtime. The follower checks
        // `CactusModel::available()` before dialing the FFI.
        println!(
            "cargo:warning=libcactus.dylib not found at {}; set CACTUS_PREFIX to override",
            dylib.display()
        );
        return;
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cactus");
    // Embed an rpath so the executable finds libcactus at runtime without
    // the user having to set DYLD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
