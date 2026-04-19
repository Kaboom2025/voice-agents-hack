//! Follower CLI: webcam → Gemini video embedding → iroh QUIC push.
//!
//! Embeds video windows (sliding frames) using GeminiVideoEmbedder, with synthetic
//! fallback when the API key is unavailable or `--synthetic` is passed.
//! That keeps the transport testable in CI / headless / no-GPU environments
//! without changing the wire protocol.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::Parser;
use common::{
    read_frame, write_frame, EmbeddingChunk, FollowerMsg, LeaderMsg, Ticket, INGEST_ALPN,
};
use follower::camera::{self, CapturedFrame};
use follower::embedder::GEMINI_EMBED_DIM;
use follower::frame_buffer::FrameBuffer;
use follower::gemini_embedder::{
    GeminiVideoEmbedder, SyntheticVideoEmbedder, VideoEmbedder, VideoEmbeddingOutput,
};
use iroh::Endpoint;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

#[derive(Parser, Debug)]
#[command(about = "iroh follower: capture webcam, embed via Gemini, push to leader")]
struct Args {
    /// Ticket string. If omitted, the follower reads it from `--ticket-file`.
    ticket: Option<String>,

    /// Path to a ticket file (the leader writes one on startup).
    #[arg(long, env = "LEADER_TICKET_FILE", default_value = ".leader.ticket")]
    ticket_file: PathBuf,

    /// Logical camera id (unique per follower).
    #[arg(long, default_value = "cam-0")]
    camera_id: String,

    /// Milliseconds between embedding steps.
    #[arg(long, default_value_t = 5000)]
    step_ms: u64,

    /// Sliding video window size in milliseconds.
    #[arg(long, default_value_t = 10_000)]
    window_ms: u64,

    /// Gemini API key. If set, uses GeminiVideoEmbedder. Falls back to synthetic.
    #[arg(long, env = "GEMINI_API_KEY")]
    gemini_api_key: Option<String>,

    /// Force synthetic random vectors regardless of API key availability.
    #[arg(long, default_value_t = false)]
    synthetic: bool,

    /// Stop after this many chunks. 0 = run forever.
    #[arg(long, default_value_t = 0)]
    count: u64,

    /// OS camera index (0 = default webcam).
    #[arg(long, default_value_t = 0)]
    device_index: u32,

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

    // --- Build the video embedder and frame buffer ------------------
    let video_embedder = build_video_embedder(&args);
    let frame_buffer = Arc::new(FrameBuffer::new(args.window_ms));

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
                            Some(f) => {
                                match tokio::task::spawn_blocking(move || encode_jpeg(&f, 85)).await
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
                                }
                            }
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

    // --- Feeder task: push frames into the FrameBuffer ---------------
    {
        let buf = frame_buffer.clone();
        let frames_clone = frames.clone();
        tokio::spawn(async move {
            // For Still frames, just push once and be done.
            // For Cam, continuously watch for updates.
            match frames_clone {
                FrameSource::Still(f) => {
                    buf.push(now_ms(), f);
                }
                FrameSource::Cam(mut rx) => loop {
                    if rx.changed().await.is_err() {
                        break;
                    }
                    if let Some(frame) = rx.borrow().clone() {
                        buf.push(now_ms(), frame);
                    }
                },
            }
        });
    }

    // --- Push loop: emit embeddings on a sliding window schedule -----
    let mut interval = tokio::time::interval(Duration::from_millis(args.step_ms));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut sent: u64 = 0;
    let mut stop = std::pin::pin!(tokio::signal::ctrl_c());

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let buf = frame_buffer.clone();
                let emb = video_embedder.clone();
                let out: VideoEmbeddingOutput = match emb.embed_window(&buf).await {
                    Ok(o) => o,
                    Err(e) => { warn!(error = %e, "embed_window failed, skipping chunk"); continue; }
                };

                let chunk = EmbeddingChunk {
                    chunk_id: format!("{}-{}", args.camera_id, sent),
                    camera_id: args.camera_id.clone(),
                    start_ts_ms: out.start_ts_ms,
                    end_ts_ms: out.end_ts_ms,
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

fn build_video_embedder(args: &Args) -> Arc<dyn VideoEmbedder> {
    if args.synthetic {
        info!("video embedder: synthetic (flag)");
        return Arc::new(SyntheticVideoEmbedder::new(GEMINI_EMBED_DIM));
    }
    if let Some(ref key) = args.gemini_api_key {
        info!("video embedder: GeminiVideoEmbedder (gemini-embedding-2-preview)");
        return Arc::new(GeminiVideoEmbedder::new(key.clone()));
    }
    info!("video embedder: synthetic (no GEMINI_API_KEY)");
    Arc::new(SyntheticVideoEmbedder::new(GEMINI_EMBED_DIM))
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
