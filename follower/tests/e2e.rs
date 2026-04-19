//! Hermetic end-to-end over real iroh QUIC — verifies the transport
//! layer without requiring Cactus or a webcam.

use std::time::Duration;

use common::{read_frame, write_frame, FollowerMsg, LeaderMsg, INGEST_ALPN};
use iroh::{
    endpoint::Connection,
    protocol::{AcceptError, ProtocolHandler, Router},
    Endpoint, Watcher,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
struct TapHandler {
    count: Arc<AtomicU64>,
    tx: mpsc::UnboundedSender<common::EmbeddingChunk>,
}

impl ProtocolHandler for TapHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let me = self.clone();
        tokio::spawn(async move {
            while let Ok((mut send, mut recv)) = conn.accept_bi().await {
                let me2 = me.clone();
                tokio::spawn(async move {
                    loop {
                        let Ok(Some(msg)) = read_frame::<_, FollowerMsg>(&mut recv).await else {
                            return;
                        };
                        match msg {
                            FollowerMsg::Hello { .. } => {}
                            FollowerMsg::Chunk(chunk) => {
                                me2.count.fetch_add(1, Ordering::Relaxed);
                                let id = chunk.chunk_id.clone();
                                let _ = me2.tx.send(chunk);
                                let _ =
                                    write_frame(&mut send, &LeaderMsg::Ack { chunk_id: id }).await;
                            }
                            FollowerMsg::Bye => return,
                            FollowerMsg::FrameResponse { .. } | FollowerMsg::FrameError { .. } => {}
                        }
                    }
                });
            }
        });
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_delivers_chunks_to_ingest_router() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

    // Leader
    let leader_ep = Endpoint::builder().discovery_n0().bind().await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let handler = TapHandler {
        count: Arc::new(AtomicU64::new(0)),
        tx,
    };
    let router = Router::builder(leader_ep.clone())
        .accept(INGEST_ALPN, handler.clone())
        .spawn();
    let leader_addr = leader_ep.node_addr().initialized().await;

    // Follower — connect + ship N chunks with simple test embeddings
    let follower_ep = Endpoint::builder().discovery_n0().bind().await.unwrap();
    let conn = follower_ep.connect(leader_addr, INGEST_ALPN).await.unwrap();
    let (mut send, _recv) = conn.open_bi().await.unwrap();
    write_frame(
        &mut send,
        &FollowerMsg::Hello {
            camera_id: "cam-test".into(),
        },
    )
    .await
    .unwrap();

    const WANT: u64 = 3;
    const DIM: usize = 1536;
    for seq in 0..WANT {
        // Create a simple deterministic embedding for testing transport.
        let mut embedding = vec![0f32; DIM];
        embedding[seq as usize % DIM] = 1.0; // one-hot for uniqueness
        let chunk = common::EmbeddingChunk {
            chunk_id: format!("cam-test-{seq}"),
            camera_id: "cam-test".into(),
            start_ts_ms: 0,
            end_ts_ms: 0,
            embedding,
            video_dim: DIM,
            audio_dim: 0,
            caption: Some(format!("test chunk #{seq}")),
            representative_jpeg: None,
        };
        write_frame(&mut send, &FollowerMsg::Chunk(chunk))
            .await
            .unwrap();
    }

    // Collect on the leader side
    let mut got = Vec::new();
    while got.len() < WANT as usize {
        let chunk = tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("chunk timed out")
            .expect("handler dropped tx");
        got.push(chunk);
    }
    assert_eq!(handler.count.load(Ordering::Relaxed), WANT);
    for (i, c) in got.iter().enumerate() {
        assert_eq!(c.chunk_id, format!("cam-test-{i}"));
        assert_eq!(c.embedding.len(), DIM);
    }

    follower_ep.close().await;
    router.shutdown().await.unwrap();
}
