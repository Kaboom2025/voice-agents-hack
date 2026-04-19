//! Leader: accepts ingest connections from any number of followers, prints
//! a ticket on startup, and logs every chunk it receives.
//!
//! When `--model-path` is set (or `CACTUS_MODEL_PATH` env var), the leader
//! loads Gemma 4 via cactus-sys and can caption / summarise incoming chunks.

mod cactus_llm;

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context, Result};
use clap::Parser;
use common::{FollowerMsg, INGEST_ALPN, LeaderMsg, Ticket, read_frame, write_frame};
use iroh::{
    Endpoint, SecretKey, Watcher,
    endpoint::Connection,
    protocol::{AcceptError, ProtocolHandler, Router},
};
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(about = "iroh leader: accepts data from followers")]
struct Args {
    /// Path to a file holding the leader's 32-byte secret key as hex.
    /// Created with mode 0600 on first run if missing. Pin this file to keep
    /// your node id stable across restarts.
    #[arg(long, env = "LEADER_KEY_FILE", default_value = ".leader.key")]
    key_file: PathBuf,

    /// Path where the dialable ticket is written on startup. Followers in the
    /// same directory will pick it up automatically.
    #[arg(long, env = "LEADER_TICKET_FILE", default_value = ".leader.ticket")]
    ticket_file: PathBuf,

    /// Filter logs (default `info`).
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log: String,

    /// Path to the Cactus model weights directory (e.g. gemma-4-e2b-it).
    /// When set, the leader loads Gemma 4 on startup and can caption chunks.
    #[arg(long, env = "CACTUS_MODEL_PATH")]
    model_path: Option<PathBuf>,
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

    let secret_key = load_or_create_key(&args.key_file)?;

    let endpoint = Endpoint::builder()
        .secret_key(secret_key)
        .discovery_n0()
        .bind()
        .await?;

    let id = endpoint.node_id();
    info!(node_id = %id, key_file = %args.key_file.display(), "leader endpoint bound");

    // Wait for a relay URL to be established so the ticket works across networks.
    let relay = endpoint.home_relay().initialized().await;
    info!(%relay, "relay established");

    // Now grab the full NodeAddr (includes the relay URL + direct addrs).
    let addr = endpoint.node_addr().initialized().await;

    let ticket = Ticket::new(addr);
    let ticket_str = ticket.to_string();
    std::fs::write(&args.ticket_file, &ticket_str)
        .with_context(|| format!("write ticket file {}", args.ticket_file.display()))?;

    println!("\n  leader ready");
    println!("  endpoint id: {id}");
    println!("  ticket file: {}", args.ticket_file.display());
    println!("  ticket (share with remote followers):\n\n{ticket_str}\n");

    // Optionally load Gemma 4 via cactus-sys.
    let model = match &args.model_path {
        Some(path) => {
            info!(path = %path.display(), "loading Gemma 4 model via cactus …");
            let m = cactus_llm::CactusModel::load(path, None)
                .with_context(|| format!("failed to load model from {}", path.display()))?;
            info!("Gemma 4 model loaded");
            Some(Arc::new(m))
        }
        None => {
            info!("no --model-path set, running without LLM");
            None
        }
    };

    let handler = IngestHandler {
        chunks_total: Arc::new(AtomicU64::new(0)),
        model: model.clone(),
    };
    let router = Router::builder(endpoint)
        .accept(INGEST_ALPN, handler)
        .spawn();

    wait_for_shutdown().await?;
    info!("shutting down");
    let _ = std::fs::remove_file(&args.ticket_file);
    router.shutdown().await?;
    Ok(())
}

/// Resolves on Ctrl-C (SIGINT) or SIGTERM so the ticket file gets cleaned up
/// regardless of how the process is asked to exit.
async fn wait_for_shutdown() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate())?;
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await?;
        Ok(())
    }
}

/// Load a hex-encoded 32-byte secret key from `path`, or generate a fresh one
/// and persist it there (mode 0600 on unix).
fn load_or_create_key(path: &Path) -> Result<SecretKey> {
    if path.exists() {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read key file {}", path.display()))?;
        let bytes = data_encoding::HEXLOWER_PERMISSIVE
            .decode(text.trim().as_bytes())
            .with_context(|| format!("key file {} is not valid hex", path.display()))?;
        let arr: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .with_context(|| format!("key file {} must decode to 32 bytes", path.display()))?;
        info!(path = %path.display(), "loaded secret key");
        Ok(SecretKey::from_bytes(&arr))
    } else {
        let sk = SecretKey::generate(&mut rand_core_06::OsRng);
        let encoded = data_encoding::HEXLOWER.encode(&sk.to_bytes());
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        std::fs::write(path, encoded)
            .with_context(|| format!("write key file {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .with_context(|| format!("chmod 600 {}", path.display()))?;
        }
        info!(path = %path.display(), "generated new secret key");
        Ok(sk)
    }
}

#[derive(Debug, Clone)]
struct IngestHandler {
    chunks_total: Arc<AtomicU64>,
    model: Option<Arc<cactus_llm::CactusModel>>,
}

impl ProtocolHandler for IngestHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let remote = conn
            .remote_node_id()
            .map(|id| id.to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        info!(%remote, "follower connected");

        if let Err(err) = self.serve(conn).await {
            warn!(%remote, %err, "follower session ended with error");
        } else {
            info!(%remote, "follower disconnected");
        }
        Ok(())
    }
}

impl IngestHandler {
    async fn serve(&self, conn: Connection) -> Result<()> {
        // We expect a single bidirectional stream per follower, opened by the
        // follower. Multiple streams per connection are fine — just loop.
        loop {
            let (mut send, mut recv) = match conn.accept_bi().await {
                Ok(s) => s,
                Err(_) => return Ok(()), // connection closed
            };

            let counter = self.chunks_total.clone();
            let model = self.model.clone();
            tokio::spawn(async move {
                let mut camera = String::from("<unknown>");
                loop {
                    let msg: Option<FollowerMsg> = match read_frame(&mut recv).await {
                        Ok(m) => m,
                        Err(e) => {
                            warn!(%e, "read failed");
                            return;
                        }
                    };
                    let Some(msg) = msg else { return };
                    match msg {
                        FollowerMsg::Hello { camera_id } => {
                            info!(%camera_id, "hello");
                            camera = camera_id;
                        }
                        FollowerMsg::Chunk(chunk) => {
                            let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                            info!(
                                total = n,
                                camera = %chunk.camera_id,
                                chunk = %chunk.chunk_id,
                                dim = chunk.embedding.len(),
                                caption = chunk.caption.as_deref().unwrap_or(""),
                                "recv chunk",
                            );

                            // If we have a loaded Gemma 4 model, generate a
                            // summary / caption for the incoming chunk.
                            if let Some(ref m) = model {
                                let prompt = format!(
                                    "Camera {} sent a video chunk ({}–{} ms). \
                                     The follower-side caption is: \"{}\". \
                                     Provide a one-sentence summary.",
                                    chunk.camera_id,
                                    chunk.start_ts_ms,
                                    chunk.end_ts_ms,
                                    chunk.caption.as_deref().unwrap_or("(none)"),
                                );
                                let messages = serde_json::json!([
                                    {"role": "user", "content": prompt}
                                ]);
                                let options = serde_json::json!({
                                    "max_tokens": 128
                                });
                                let chunk_id = chunk.chunk_id.clone();
                                let m = m.clone();
                                let msgs_str = messages.to_string();
                                let opts_str = options.to_string();
                                // Run inference on a blocking thread so we
                                // don't stall the tokio runtime.
                                tokio::task::spawn_blocking(move || {
                                    match m.complete(&msgs_str, Some(&opts_str)) {
                                        Ok(resp) => {
                                            info!(
                                                chunk = %chunk_id,
                                                gemma4_response = %resp,
                                                "LLM caption",
                                            );
                                        }
                                        Err(e) => {
                                            warn!(
                                                chunk = %chunk_id,
                                                %e,
                                                "Gemma 4 completion failed",
                                            );
                                        }
                                    }
                                });
                            }

                            // TODO: persist to ChromaDB / SQLite.
                            let ack = LeaderMsg::Ack {
                                chunk_id: chunk.chunk_id,
                            };
                            if let Err(e) = write_frame(&mut send, &ack).await {
                                error!(%e, "ack write failed");
                                return;
                            }
                        }
                        FollowerMsg::Bye => {
                            info!(%camera, "bye");
                            return;
                        }
                    }
                }
            });
        }
    }
}
