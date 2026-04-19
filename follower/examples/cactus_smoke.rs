//! Smoke test: load Gemma-4-E2B via Cactus, embed one image and one
//! text prompt, print the dimensions and first few values.
//!
//! Usage:
//!   cargo run -p follower --example cactus_smoke --release -- \
//!       --image path/to/test.jpg
//!
//! If `--image` is omitted, we generate a 64x64 gradient, write it to
//! `/tmp/cactus_smoke.jpg`, and embed that. This keeps the smoke test
//! self-contained.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use follower::cactus::CactusModel;
use image::{codecs::jpeg::JpegEncoder, ExtendedColorType};

#[derive(Parser, Debug)]
struct Args {
    /// Directory holding the model's `.weights` files.
    #[arg(
        long,
        default_value = "/opt/homebrew/opt/cactus/libexec/weights/gemma-4-e2b-it"
    )]
    model_path: PathBuf,

    /// Optional image path to embed. Falls back to a generated gradient.
    #[arg(long)]
    image: Option<PathBuf>,

    /// Optional text prompt to embed.
    #[arg(long, default_value = "a person walking past a doorway")]
    text: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .with_target(false)
        .init();

    let args = Args::parse();

    println!("→ loading model from {}", args.model_path.display());
    let t0 = Instant::now();
    let model = CactusModel::new(&args.model_path).context("cactus init")?;
    println!("  model loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let image_path = match args.image {
        Some(p) => p,
        None => {
            let p = std::env::temp_dir().join("cactus_smoke.jpg");
            write_gradient(&p)?;
            println!("→ generated test image: {}", p.display());
            p
        }
    };

    println!("→ embedding image {}", image_path.display());
    let t1 = Instant::now();
    let img_emb = model
        .embed_image(&image_path)
        .context("cactus image embed")?;
    println!(
        "  image embedding: dim={} first=[{:.4}, {:.4}, {:.4}, {:.4}] ({}ms)",
        img_emb.len(),
        img_emb.first().copied().unwrap_or(0.0),
        img_emb.get(1).copied().unwrap_or(0.0),
        img_emb.get(2).copied().unwrap_or(0.0),
        img_emb.get(3).copied().unwrap_or(0.0),
        t1.elapsed().as_millis()
    );

    println!("→ embedding text {:?}", args.text);
    let t2 = Instant::now();
    let txt_emb = model
        .embed_text(&args.text, true)
        .context("cactus text embed")?;
    println!(
        "  text embedding:  dim={} first=[{:.4}, {:.4}, {:.4}, {:.4}] ({}ms)",
        txt_emb.len(),
        txt_emb.first().copied().unwrap_or(0.0),
        txt_emb.get(1).copied().unwrap_or(0.0),
        txt_emb.get(2).copied().unwrap_or(0.0),
        txt_emb.get(3).copied().unwrap_or(0.0),
        t2.elapsed().as_millis()
    );

    println!("\n✓ cactus smoke ok");
    Ok(())
}

/// Write a 64x64 RGB gradient JPEG so the smoke test runs even without
/// a fixture image on disk.
fn write_gradient(path: &std::path::Path) -> Result<()> {
    let w: u32 = 64;
    let h: u32 = 64;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            rgb.push((x * 4) as u8);
            rgb.push((y * 4) as u8);
            rgb.push(((x ^ y) * 4) as u8);
        }
    }
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 85);
    enc.encode(&rgb, w, h, ExtendedColorType::Rgb8)?;
    Ok(())
}
