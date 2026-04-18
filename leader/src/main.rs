//! Leader: accepts ingest connections from any number of followers, prints
//! a ticket on startup, and logs every chunk it receives.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::Result;
use clap::Parser;
use common::{FollowerMsg, INGEST_ALPN, LeaderMsg, Ticket, read_frame, write_frame};
use iroh::{
    Endpoint, Watcher,
    endpoint::Connection,
    protocol::{AcceptError, ProtocolHandler, Router},
};
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(about = "iroh leader: accepts data from followers")]
struct Args {
    /// Filter logs (default `info`).
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(args.log)
        .with_target(false)
        .init();

    let endpoint = Endpoint::builder()
        .discovery_n0()
        .bind()
        .await?;

    let id = endpoint.node_id();
    info!(node_id = %id, "leader endpoint bound");

    // Wait until our address (relay/direct) is known so the ticket is dialable.
    let addr = endpoint.node_addr().initialized().await;

    let ticket = Ticket::new(addr);
    println!("\n  leader ready");
    println!("  endpoint id: {id}");
    println!("  ticket (share with followers):\n\n{ticket}\n");

    let handler = IngestHandler::default();
    let router = Router::builder(endpoint)
        .accept(INGEST_ALPN, handler)
        .spawn();

    tokio::signal::ctrl_c().await?;
    info!("shutting down");
    router.shutdown().await?;
    Ok(())
}

#[derive(Debug, Default, Clone)]
struct IngestHandler {
    chunks_total: Arc<AtomicU64>,
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
