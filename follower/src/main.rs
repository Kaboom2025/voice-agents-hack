//! Follower: dials the leader using a ticket and streams synthetic
//! `EmbeddingChunk`s on a tick. Replace `make_chunk` with the real Cactus +
//! Gemma embedding output once the capture pipeline is wired up.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use common::{EmbeddingChunk, FollowerMsg, INGEST_ALPN, LeaderMsg, Ticket, read_frame, write_frame};
use iroh::Endpoint;
use rand::Rng;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(about = "iroh follower: streams embedding chunks to the leader")]
struct Args {
    /// Ticket printed by the leader.
    ticket: String,

    /// Logical camera id (any string unique per follower).
    #[arg(long, default_value = "cam-0")]
    camera_id: String,

    /// Milliseconds between chunks.
    #[arg(long, default_value_t = 1000)]
    interval_ms: u64,

    /// Embedding dimensionality (placeholder for the real model output).
    #[arg(long, default_value_t = 768)]
    dim: usize,

    /// Stop after this many chunks. 0 = run forever.
    #[arg(long, default_value_t = 0)]
    count: u64,

    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(&args.log)
        .with_target(false)
        .init();

    let ticket: Ticket = args.ticket.parse().context("parse ticket")?;
    info!(leader = %ticket.leader.node_id, "dialing leader");

    let endpoint = Endpoint::builder()
        .discovery_n0()
        .bind()
        .await?;

    let conn = endpoint.connect(ticket.leader, INGEST_ALPN).await?;
    info!("connected");

    let (mut send, mut recv) = conn.open_bi().await?;

    write_frame(
        &mut send,
        &FollowerMsg::Hello {
            camera_id: args.camera_id.clone(),
        },
    )
    .await?;

    // Spawn a task that just drains acks so the stream's flow control stays open.
    let ack_task = tokio::spawn(async move {
        while let Ok(Some(LeaderMsg::Ack { chunk_id })) = read_frame(&mut recv).await {
            tracing::debug!(%chunk_id, "ack");
        }
    });

    let mut interval = tokio::time::interval(Duration::from_millis(args.interval_ms));
    let mut sent: u64 = 0;
    let mut stop = std::pin::pin!(tokio::signal::ctrl_c());

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let chunk = make_chunk(&args.camera_id, args.dim, sent);
                if let Err(e) = write_frame(&mut send, &FollowerMsg::Chunk(chunk)).await {
                    warn!(%e, "send failed; exiting");
                    break;
                }
                sent += 1;
                info!(sent, "chunk sent");
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

    let _ = write_frame(&mut send, &FollowerMsg::Bye).await;
    let _ = send.finish();
    ack_task.abort();
    endpoint.close().await;
    Ok(())
}

fn make_chunk(camera_id: &str, dim: usize, seq: u64) -> EmbeddingChunk {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut rng = rand::rng();
    let embedding: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
    EmbeddingChunk {
        chunk_id: format!("{camera_id}-{seq}"),
        camera_id: camera_id.to_string(),
        start_ts_ms: now_ms,
        end_ts_ms: now_ms + 5_000,
        embedding,
        caption: Some(format!("synthetic chunk #{seq}")),
    }
}
