//! Follower CLI: webcam → Cactus (Gemma-4) embedding → iroh QUIC push.
//!
//! Synthetic fallback kicks in automatically when either the model or
//! the camera is unavailable (or when you pass `--synthetic`). That
//! keeps the transport testable in CI / headless / no-GPU environments
//! without changing the wire protocol.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Parser;
use common::{
    read_frame, write_frame, EmbeddingChunk, FollowerMsg, LeaderMsg, Ticket, INGEST_ALPN,
};
#[cfg(feature = "cactus")]
use follower::cactus::CactusModel;
use follower::camera::{self, CapturedFrame};
#[cfg(feature = "cactus")]
use follower::embedder::CactusEmbedder;
use follower::embedder::{Embedder, SyntheticEmbedder, GEMMA4_HIDDEN_DIM};
use iroh::Endpoint;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

#[derive(Parser, Debug)]
#[command(about = "iroh follower: capture webcam, embed via Gemma-4, push to leader")]
struct Args {
    /// Ticket string. If omitted, the follower reads it from `--ticket-file`.
    ticket: Option<String>,

    /// Path to a ticket file (the leader writes one on startup).
    #[arg(long, env = "LEADER_TICKET_FILE", default_value = ".leader.ticket")]
    ticket_file: PathBuf,

    /// Logical camera id (unique per follower).
    #[arg(long, default_value = "cam-0")]
    camera_id: String,

    /// Milliseconds between chunks. Keep ≥ Cactus latency (~5s/image on CPU).
    #[arg(long, default_value_t = 5000)]
    interval_ms: u64,

    /// Stop after this many chunks. 0 = run forever.
    #[arg(long, default_value_t = 0)]
    count: u64,

    /// OS camera index (0 = default webcam).
    #[arg(long, default_value_t = 0)]
    device_index: u32,

    /// Path to the Cactus-converted Gemma model directory.
    #[arg(
        long,
        env = "GEMMA_MODEL_PATH",
        default_value = "/opt/homebrew/opt/cactus/libexec/weights/gemma-4-e2b-it"
    )]
    model_path: PathBuf,

    /// Skip Cactus entirely and ship synthetic random vectors.
    #[arg(long, default_value_t = false)]
    synthetic: bool,

    /// Skip the webcam and use a solid-color placeholder frame. Useful
    /// when you want real embeddings but no camera hardware.
    #[arg(long, default_value_t = false)]
    no_camera: bool,

    /// Directory where captured JPEG frames are written (one file per
    /// chunk, named `<camera-id>-<seq>.jpg`). Created if missing.
    #[arg(long, env = "FOLLOWER_FRAME_DIR", default_value = "./frames")]
    frame_dir: PathBuf,

    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Pull values from `.env` if present; real env always wins.
    let _ = dotenvy::dotenv();

    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(&args.log)
        .with_target(false)
        .init();

    let ticket_str = match args.ticket.clone() {
        Some(t) => t,
        None => std::fs::read_to_string(&args.ticket_file)
            .with_context(|| {
                format!(
                    "no ticket given and ticket file {} not readable (is the leader running?)",
                    args.ticket_file.display()
                )
            })?
            .trim()
            .to_string(),
    };
    if ticket_str.is_empty() {
        bail!("ticket is empty");
    }

    let ticket: Ticket = ticket_str.parse().context("parse ticket")?;
    info!(leader = %ticket.leader.node_id, "dialing leader");

    let endpoint = Endpoint::builder().discovery_n0().bind().await?;
    let conn = endpoint.connect(ticket.leader, INGEST_ALPN).await?;
    info!("connected");
    let (mut send, mut recv) = conn.open_bi().await?;

    // All outbound frames flow through this channel so the chunk loop and
    // the on-demand frame-request handler don't race for the send half.
    let (writer_tx, mut writer_rx) = mpsc::channel::<FollowerMsg>(128);
    let writer_task = tokio::spawn(async move {
        while let Some(msg) = writer_rx.recv().await {
            if let Err(e) = write_frame(&mut send, &msg).await {
                warn!(%e, "writer: send failed");
                break;
            }
        }
        let _ = send.finish();
    });

    writer_tx
        .send(FollowerMsg::Hello {
            camera_id: args.camera_id.clone(),
        })
        .await
        .context("send hello")?;

    // Ensure the frame directory exists before the first write.
    std::fs::create_dir_all(&args.frame_dir)
        .with_context(|| format!("create frame dir {}", args.frame_dir.display()))?;
    info!(dir = %args.frame_dir.display(), "saving frames");

    // --- Build the embedder --------------------------------------
    let embedder: Arc<dyn Embedder> = build_embedder(&args).await?;

    // --- Build the frame source ----------------------------------
    let frames = if args.no_camera {
        info!("frame source: solid placeholder");
        FrameSource::Still(solid_placeholder())
    } else {
        match camera::spawn(args.device_index) {
            Ok(handle) => {
                info!(device = args.device_index, "webcam opened");
                FrameSource::Cam(handle.rx)
            }
            Err(e) => {
                warn!(error = %e, "camera open failed, using placeholder frame");
                FrameSource::Still(solid_placeholder())
            }
        }
    };

    // --- Reader task: drain LeaderMsg, serve FrameRequests --------
    let frames_for_reader = frames.clone();
    let writer_for_reader = writer_tx.clone();
    let reader_task = tokio::spawn(async move {
        loop {
            let msg: Option<LeaderMsg> = match read_frame(&mut recv).await {
                Ok(m) => m,
                Err(e) => {
                    warn!(%e, "reader: read failed");
                    break;
                }
            };
            let Some(msg) = msg else { break };
            match msg {
                LeaderMsg::Ack { chunk_id } => debug!(%chunk_id, "ack"),
                LeaderMsg::FrameRequest { req_id } => {
                    let frame = frames_for_reader.current();
                    let writer = writer_for_reader.clone();
                    tokio::spawn(async move {
                        let resp = match frame {
                            Some(f) => match tokio::task::spawn_blocking(move || {
                                encode_jpeg(&f, 85)
                            })
                            .await
                            {
                                Ok(Ok((jpeg, w, h))) => FollowerMsg::FrameResponse {
                                    req_id,
                                    ts_ms: now_ms(),
                                    width: w,
                                    height: h,
                                    jpeg,
                                },
                                Ok(Err(e)) => FollowerMsg::FrameError {
                                    req_id,
                                    message: format!("encode failed: {e}"),
                                },
                                Err(e) => FollowerMsg::FrameError {
                                    req_id,
                                    message: format!("encode task panicked: {e}"),
                                },
                            },
                            None => FollowerMsg::FrameError {
                                req_id,
                                message: "no frame available yet".into(),
                            },
                        };
                        let _ = writer.send(resp).await;
                    });
                }
            }
        }
    });

    // --- Push loop ----------------------------------------------
    let mut interval = tokio::time::interval(Duration::from_millis(args.interval_ms));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut sent: u64 = 0;
    let mut stop = std::pin::pin!(tokio::signal::ctrl_c());

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Grab a frame.
                let frame = match frames.current() {
                    Some(f) => f,
                    None => {
                        warn!("camera has no frame yet, waiting for next tick");
                        continue;
                    }
                };

                // Embed on a blocking thread — Cactus on CPU is slow.
                let seq = sent;
                let emb = embedder.clone();
                let out = match tokio::task::spawn_blocking(move || {
                    emb.embed_frame(&frame, seq)
                }).await {
                    Ok(Ok(o)) => o,
                    Ok(Err(e)) => { warn!(error = %e, "embed failed, skipping chunk"); continue; }
                    Err(e) => { warn!(error = %e, "embed task panicked"); continue; }
                };

                let ts = now_ms();
                let chunk = EmbeddingChunk {
                    chunk_id: format!("{}-{}", args.camera_id, sent),
                    camera_id: args.camera_id.clone(),
                    start_ts_ms: ts,
                    end_ts_ms: ts + args.interval_ms,
                    embedding: out.embedding,
                    caption: out.caption,
                };
                let dim = chunk.embedding.len();
                if writer_tx.send(FollowerMsg::Chunk(chunk)).await.is_err() {
                    warn!("writer channel closed; exiting");
                    break;
                }
                sent += 1;
                info!(sent, dim, "chunk sent");
                if args.count != 0 && sent >= args.count {
                    break;
                }
            }
            _ = &mut stop => {
                info!("ctrl-c, stopping");
                break;
            }
        }
    }

    let _ = writer_tx.send(FollowerMsg::Bye).await;
    drop(writer_tx);
    let _ = writer_task.await;
    reader_task.abort();
    endpoint.close().await;
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(feature = "cactus")]
async fn build_embedder(args: &Args) -> Result<Arc<dyn Embedder>> {
    if args.synthetic {
        info!("embedder: synthetic (flag)");
        return Ok(Arc::new(SyntheticEmbedder::new(GEMMA4_HIDDEN_DIM)));
    }
    let model_path = args.model_path.clone();
    let model = tokio::task::spawn_blocking(move || CactusModel::new(&model_path))
        .await
        .context("join cactus init")?;
    match model {
        Ok(m) => {
            info!(path = %args.model_path.display(), "cactus gemma-4 loaded");
            Ok(Arc::new(
                CactusEmbedder::new(Arc::new(m))
                    .with_tmp_dir(args.frame_dir.clone())
                    .with_file_prefix(args.camera_id.clone()),
            ))
        }
        Err(e) => {
            warn!(error = %e, "cactus init failed, falling back to synthetic");
            Ok(Arc::new(SyntheticEmbedder::new(GEMMA4_HIDDEN_DIM)))
        }
    }
}

#[cfg(not(feature = "cactus"))]
async fn build_embedder(_args: &Args) -> Result<Arc<dyn Embedder>> {
    info!("embedder: synthetic (cactus feature off)");
    Ok(Arc::new(SyntheticEmbedder::new(GEMMA4_HIDDEN_DIM)))
}

/// Encode an in-memory RGB frame to JPEG. Returns `(bytes, width, height)`.
fn encode_jpeg(frame: &CapturedFrame, quality: u8) -> Result<(Vec<u8>, u32, u32)> {
    use image::{codecs::jpeg::JpegEncoder, ExtendedColorType};
    let cap = (frame.width as usize) * (frame.height as usize) / 4;
    let mut buf = Vec::with_capacity(cap.max(64 * 1024));
    let mut encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .encode(
            frame.rgb.as_slice(),
            frame.width,
            frame.height,
            ExtendedColorType::Rgb8,
        )
        .context("jpeg encode")?;
    Ok((buf, frame.width, frame.height))
}

/// Source of frames for the embed loop and live snapshots. Cloning is cheap:
/// `watch::Receiver` clones share the underlying channel; `CapturedFrame` is
/// internally `Arc<Vec<u8>>` so its clone is a single refcount bump.
#[derive(Clone)]
enum FrameSource {
    Cam(tokio::sync::watch::Receiver<Option<CapturedFrame>>),
    Still(CapturedFrame),
}

impl FrameSource {
    fn current(&self) -> Option<CapturedFrame> {
        match self {
            FrameSource::Cam(rx) => rx.borrow().clone(),
            FrameSource::Still(f) => Some(f.clone()),
        }
    }
}

/// 64x64 mid-gray RGB frame used when no camera is available.
fn solid_placeholder() -> CapturedFrame {
    CapturedFrame {
        width: 64,
        height: 64,
        rgb: Arc::new(vec![128u8; 64 * 64 * 3]),
    }
}
