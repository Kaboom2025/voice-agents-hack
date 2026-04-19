//! Follower CLI: webcam + mic → Gemini/Cactus embedding → iroh QUIC push.
//!
//! PRD §5.1: each 5 s chunk samples K=4 evenly-spaced frames from the
//! camera plus the audio segment from the microphone. The embedder
//! produces a `[video_emb || audio_emb]` vector per chunk.
//!
//! Embeds video windows (sliding frames) using GeminiVideoEmbedder.
//! Requires either a Cactus/Gemma model or a Gemini API key — will
//! not start without a real embedding backend.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::Parser;
use common::{
    read_frame, write_frame, EmbeddingChunk, FollowerMsg, LeaderMsg, Ticket, INGEST_ALPN,
};
use follower::audio;
#[cfg(feature = "cactus")]
use follower::cactus::CactusModel;
use follower::camera::{self, CapturedFrame};
#[cfg(feature = "cactus")]
use follower::embedder::CactusEmbedder;
use follower::embedder::{ChunkInput, Embedder};
use follower::gemini_embedder::{GeminiEmbedder, GeminiVideoEmbedder};
use iroh::Endpoint;
use tokio::sync::{mpsc, watch};
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

    /// Gemini API key. If set, uses GeminiVideoEmbedder. Required when
    /// Cactus model is not available.
    #[arg(long, env = "GEMINI_API_KEY")]
    gemini_api_key: Option<String>,

    /// Path to the Cactus-converted Gemma weights directory. Used when
    /// built with `--features cactus` for on-device embedding.
    #[arg(
        long,
        env = "GEMMA_MODEL_PATH",
        default_value = "weights/gemma-4-e2b-it"
    )]
    model_path: PathBuf,

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

    /// Skip the microphone. Useful in headless / no-audio environments.
    #[arg(long, default_value_t = false)]
    no_audio: bool,

    /// Number of evenly-spaced frames to sample per chunk (PRD §5.1 K).
    #[arg(long, default_value_t = 4)]
    frames_per_chunk: usize,

    /// Directory where captured JPEG frames are written (one file per
    /// chunk, named `<camera-id>-<seq>.jpg`). Created if missing.
    #[arg(long, env = "FOLLOWER_FRAME_DIR", default_value = "./frames")]
    frame_dir: PathBuf,

    /// Maximum number of reconnection attempts before giving up. 0 = retry
    /// forever.
    #[arg(long, default_value_t = 0)]
    max_retries: u64,

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

    // Ensure the frame directory exists before the first write.
    std::fs::create_dir_all(&args.frame_dir)
        .with_context(|| format!("create frame dir {}", args.frame_dir.display()))?;
    info!(dir = %args.frame_dir.display(), "saving frames");

    // --- Build the embedder (once, reused across reconnects) -----
    let embedder: Arc<dyn Embedder> = build_embedder(&args).await?;

    // --- Build the frame source (once, reused across reconnects) -
    let frames = if args.no_camera {
        info!("frame source: disabled (--no-camera)");
        FrameSource::None
    } else {
        match camera::spawn(args.device_index) {
            Ok(handle) => {
                info!(device = args.device_index, "webcam opened");
                FrameSource::Cam(handle.rx)
            }
            Err(e) => {
                bail!("camera open failed: {e}. Use --no-camera to run without a webcam.");
            }
        }
    };

    // --- Build the audio source (once, reused across reconnects) -
    let audio_buf = if args.no_audio {
        info!("audio: disabled (--no-audio)");
        None
    } else {
        match audio::start_capture() {
            Ok(handle) => {
                info!("audio capture started");
                Some(handle)
            }
            Err(e) => {
                warn!(error = %e, "mic open failed, continuing without audio");
                None
            }
        }
    };

    // --- iroh endpoint (once, reused across reconnects) ----------
    let endpoint = Endpoint::builder().discovery_n0().bind().await?;

    // --- Graceful shutdown on ctrl-c -----------------------------
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        info!("ctrl-c received, shutting down");
        let _ = shutdown_tx.send(true);
    });

    // --- Reconnect loop ------------------------------------------
    let mut attempt: u64 = 0;
    let mut total_sent: u64 = 0;
    loop {
        if *shutdown_rx.borrow() {
            info!("shutdown requested, stopping");
            break;
        }
        attempt += 1;
        if args.max_retries > 0 && attempt > args.max_retries {
            warn!(attempts = attempt - 1, "max retries exceeded, giving up");
            break;
        }
        if attempt > 1 {
            let backoff = Duration::from_secs((attempt - 1).min(30));
            info!(attempt, backoff_secs = backoff.as_secs(), "reconnecting");
            let mut rx = shutdown_rx.clone();
            tokio::select! {
                _ = tokio::time::sleep(backoff) => {}
                _ = wait_for_shutdown(&mut rx) => {
                    info!("ctrl-c during backoff, stopping");
                    break;
                }
            }
        }

        info!(leader = %ticket.leader.node_id, attempt, "dialing leader");
        let mut rx = shutdown_rx.clone();
        let conn = tokio::select! {
            result = endpoint.connect(ticket.leader.clone(), INGEST_ALPN) => {
                match result {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, "connect failed");
                        continue;
                    }
                }
            }
            _ = wait_for_shutdown(&mut rx) => {
                info!("ctrl-c during connect, stopping");
                break;
            }
        };
        info!("connected");

        match run_session(
            &conn,
            &args,
            &embedder,
            &frames,
            audio_buf.as_ref(),
            &mut total_sent,
            shutdown_rx.clone(),
        )
        .await
        {
            Ok(SessionEnd::Done) => break,
            Ok(SessionEnd::CtrlC) => {
                info!("ctrl-c, stopping");
                break;
            }
            Ok(SessionEnd::Disconnected(reason)) => {
                warn!(%reason, "session ended, will reconnect");
            }
            Err(e) => {
                warn!(error = %e, "session error, will reconnect");
            }
        }
    }

    endpoint.close().await;
    Ok(())
}

enum SessionEnd {
    /// --count reached or clean Bye.
    Done,
    /// User pressed ctrl-c.
    CtrlC,
    /// Transport error; reconnect.
    Disconnected(String),
}

/// One connection session. Returns when the session should end or when
/// the transport breaks (caller decides whether to reconnect).
async fn run_session(
    conn: &iroh::endpoint::Connection,
    args: &Args,
    embedder: &Arc<dyn Embedder>,
    frames: &FrameSource,
    audio_buf: Option<&audio::AudioHandle>,
    total_sent: &mut u64,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<SessionEnd> {
    let (mut send, mut recv) = conn.open_bi().await.context("open bidi stream")?;

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

    // --- Push loop: collect K frames over the chunk window, then embed --
    let chunk_duration = Duration::from_millis(args.step_ms);
    let k = args.frames_per_chunk.max(1);
    let frame_interval = chunk_duration / k as u32;
    let mut sent_this_session: u64 = 0;

    let result = loop {
        // Collect K evenly-spaced frames across the chunk window.
        let chunk_start_ms = now_ms();
        let mut sampled_frames: Vec<CapturedFrame> = Vec::with_capacity(k);
        let mut frame_ticker = tokio::time::interval(frame_interval);
        frame_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        let mut aborted = false;
        for _i in 0..k {
            tokio::select! {
                _ = frame_ticker.tick() => {
                    if let Some(f) = frames.current() {
                        sampled_frames.push(f);
                    }
                }
                _ = wait_for_shutdown(&mut shutdown_rx) => {
                    aborted = true;
                    break;
                }
            }
        }
        if aborted {
            break SessionEnd::CtrlC;
        }

        if sampled_frames.is_empty() {
            warn!("no frames captured this chunk, skipping");
            continue;
        }

        // Extract middle frame JPEG before moving sampled_frames into input.
        let representative_jpeg = {
            sampled_frames
                .get(sampled_frames.len() / 2)
                .and_then(
                    |mid_frame| match GeminiVideoEmbedder::encode_jpeg_bytes(mid_frame, 60) {
                        Ok(jpeg) => Some(jpeg),
                        Err(e) => {
                            warn!(error = %e, "failed to encode representative JPEG");
                            None
                        }
                    },
                )
        };

        // Drain the audio accumulated during this window.
        let audio_samples = audio_buf.map(|h| h.buffer.drain()).unwrap_or_default();

        // Capture end timestamp NOW, before the (potentially slow) embed
        // call, so the chunk's time range reflects the actual capture
        // window rather than being inflated by embedding latency.
        let chunk_end_ms = now_ms();

        let input = ChunkInput {
            frames: sampled_frames.clone(),
            audio_samples,
        };

        // Async embed — Gemini awaits HTTP directly; Cactus offloads to
        // spawn_blocking internally; synthetic returns instantly.
        let seq = *total_sent;
        let embed_fut = embedder.embed_chunk(&input, seq);
        tokio::pin!(embed_fut);

        let out = tokio::select! {
            res = &mut embed_fut => match res {
                Ok(o) => o,
                Err(e) => {
                    warn!(error = %e, "embed failed, skipping chunk");
                    continue;
                }
            },
            _ = wait_for_shutdown(&mut shutdown_rx) => {
                break SessionEnd::CtrlC;
            }
        };

        let ts = chunk_start_ms;
        let chunk = EmbeddingChunk {
            chunk_id: format!("{}-{}", args.camera_id, *total_sent),
            camera_id: args.camera_id.clone(),
            start_ts_ms: ts,
            end_ts_ms: chunk_end_ms,
            embedding: out.embedding,
            video_dim: out.video_dim,
            audio_dim: out.audio_dim,
            caption: out.caption,
            representative_jpeg,
        };
        let dim = chunk.embedding.len();
        let vd = chunk.video_dim;
        let ad = chunk.audio_dim;
        if writer_tx.send(FollowerMsg::Chunk(chunk)).await.is_err() {
            break SessionEnd::Disconnected("writer channel closed".into());
        }
        *total_sent += 1;
        sent_this_session += 1;
        info!(
            total = *total_sent,
            session = sent_this_session,
            dim,
            video_dim = vd,
            audio_dim = ad,
            "chunk sent"
        );
        if args.count != 0 && *total_sent >= args.count {
            break SessionEnd::Done;
        }
    };

    let _ = writer_tx.send(FollowerMsg::Bye).await;
    drop(writer_tx);
    let _ = writer_task.await;
    reader_task.abort();
    Ok(result)
}

/// Resolves when the shutdown watch channel becomes `true`.
async fn wait_for_shutdown(rx: &mut watch::Receiver<bool>) {
    while !*rx.borrow() {
        if rx.changed().await.is_err() {
            return; // sender dropped — treat as shutdown
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

async fn build_embedder(args: &Args) -> Result<Arc<dyn Embedder>> {
    // 1) Try local Cactus/Gemma (on-device, no API key needed).
    #[cfg(feature = "cactus")]
    {
        if args.model_path.exists() {
            info!(path = %args.model_path.display(), "loading gemma-4 via cactus for embedding");
            match CactusModel::new(&args.model_path) {
                Ok(model) => {
                    info!("embedder: CactusEmbedder (local gemma-4)");
                    return Ok(Arc::new(CactusEmbedder::new(Arc::new(model))));
                }
                Err(e) => {
                    warn!(error = %e, "cactus model load failed, trying gemini");
                }
            }
        } else {
            info!(path = %args.model_path.display(), "gemma weights not found, trying gemini");
        }
    }

    // 2) Gemini API.
    if let Some(ref key) = args.gemini_api_key {
        info!("embedder: GeminiEmbedder (gemini-embedding-2-preview)");
        return Ok(Arc::new(GeminiEmbedder::new(key.clone())));
    }

    // No embedding backend available — refuse to start.
    bail!(
        "no embedding backend available. Either:\n  \
         1) Build with --features cactus and provide model weights at --model-path, or\n  \
         2) Set GEMINI_API_KEY (or --gemini-api-key) for the Gemini API."
    )
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
    None,
}

impl FrameSource {
    fn current(&self) -> Option<CapturedFrame> {
        match self {
            FrameSource::Cam(rx) => rx.borrow().clone(),
            FrameSource::None => None,
        }
    }
}
