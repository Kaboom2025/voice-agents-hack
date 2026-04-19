//! HLS recorder: tees decoded RGB frames from the camera into an ffmpeg
//! subprocess that writes a live HLS playlist to disk.
//!
//! Output layout (co-located demo setup — leader reads the same dir):
//!
//!   <recordings_dir>/<camera_id>/stream.m3u8
//!   <recordings_dir>/<camera_id>/seg_000001.ts ...
//!
//! ffmpeg is invoked with `program_date_time` so each segment carries an
//! `#EXT-X-PROGRAM-DATE-TIME` tag, which lets the UI seek to a wall-clock
//! timestamp (the `start_ts_ms` returned by retrieval).
//!
//! The recorder is best-effort: if ffmpeg is missing or writes to stdin
//! fail, we log and disable recording for the rest of the session without
//! tearing down the embedding pipeline.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::watch;
use tracing::{info, warn};

use crate::camera::CapturedFrame;

/// Configuration for the HLS recorder. Matches the CLI flags exposed in
/// `main.rs` but is kept separate so the module stays testable.
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub camera_id: String,
    pub recordings_dir: PathBuf,
    /// Target encoding framerate. Lower than the camera fps saves disk.
    pub fps: u32,
    /// HLS segment length (seconds). 4 s is a good balance of seek
    /// granularity vs. segment count on disk.
    pub segment_secs: u32,
}

impl RecorderConfig {
    fn output_dir(&self) -> PathBuf {
        self.recordings_dir.join(&self.camera_id)
    }
}

/// Spawn the recorder. Returns immediately; the actual ffmpeg subprocess
/// is started on the first real frame (when we finally know the frame
/// dimensions). The returned future completes when the camera channel
/// closes or shutdown is requested.
pub fn spawn(
    cfg: RecorderConfig,
    mut frames: watch::Receiver<Option<CapturedFrame>>,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut state: Option<RecorderState> = None;
        let frame_interval = Duration::from_millis(1000 / cfg.fps.max(1) as u64);
        let mut last_emit = Instant::now() - frame_interval;

        loop {
            // Wait for either a new frame or a shutdown signal.
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        info!("recorder: shutdown signal, stopping");
                        break;
                    }
                }
                changed = frames.changed() => {
                    if changed.is_err() {
                        info!("recorder: camera channel closed, stopping");
                        break;
                    }
                }
            }

            let frame = match frames.borrow_and_update().clone() {
                Some(f) => f,
                None => continue,
            };

            // Throttle to target fps.
            let now = Instant::now();
            if now.duration_since(last_emit) < frame_interval {
                continue;
            }
            last_emit = now;

            // Lazy-start ffmpeg once we know the frame dimensions.
            if state.is_none() {
                match RecorderState::start(&cfg, frame.width, frame.height).await {
                    Ok(s) => {
                        info!(
                            camera = %cfg.camera_id,
                            width = frame.width,
                            height = frame.height,
                            fps = cfg.fps,
                            dir = %cfg.output_dir().display(),
                            "recorder: ffmpeg started"
                        );
                        state = Some(s);
                    }
                    Err(e) => {
                        warn!(error = %e, "recorder: failed to start ffmpeg, disabling recording");
                        return;
                    }
                }
            }

            let s = state.as_mut().expect("state initialised above");
            if let Err(e) = s.write_frame(&frame).await {
                warn!(error = %e, "recorder: stdin write failed, stopping recorder");
                break;
            }
        }

        if let Some(mut s) = state {
            s.shutdown().await;
        }
    })
}

struct RecorderState {
    child: Child,
    stdin: ChildStdin,
    width: u32,
    height: u32,
}

impl RecorderState {
    async fn start(cfg: &RecorderConfig, width: u32, height: u32) -> Result<Self> {
        let out_dir = cfg.output_dir();
        tokio::fs::create_dir_all(&out_dir)
            .await
            .with_context(|| format!("create recordings dir {}", out_dir.display()))?;

        let playlist = out_dir.join("stream.m3u8");
        let segment_tpl = out_dir.join("seg_%06d.ts");
        let gop = (cfg.fps.max(1) * cfg.segment_secs.max(1)) as u32;

        let mut cmd = Command::new("ffmpeg");
        cmd.kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .arg("-hide_banner")
            .arg("-loglevel").arg("warning")
            .arg("-y")
            // rawvideo input from our camera thread
            .arg("-f").arg("rawvideo")
            .arg("-pixel_format").arg("rgb24")
            .arg("-video_size").arg(format!("{}x{}", width, height))
            .arg("-framerate").arg(cfg.fps.to_string())
            .arg("-i").arg("-")
            // encoder
            .arg("-c:v").arg("libx264")
            .arg("-preset").arg("veryfast")
            .arg("-tune").arg("zerolatency")
            .arg("-pix_fmt").arg("yuv420p")
            .arg("-g").arg(gop.to_string())
            // HLS muxer
            .arg("-hls_time").arg(cfg.segment_secs.to_string())
            .arg("-hls_list_size").arg("0")
            .arg("-hls_flags").arg("program_date_time+append_list+independent_segments")
            .arg("-hls_segment_filename").arg(path_str(&segment_tpl))
            .arg("-f").arg("hls")
            .arg(path_str(&playlist));

        let mut child = cmd
            .spawn()
            .context("spawn ffmpeg (is it on PATH?)")?;
        let stdin = child
            .stdin
            .take()
            .context("ffmpeg stdin unavailable")?;

        Ok(Self { child, stdin, width, height })
    }

    async fn write_frame(&mut self, frame: &CapturedFrame) -> Result<()> {
        // Resolution renegotiation isn't supported by our ffmpeg invocation;
        // drop any frame that doesn't match the dims we started with.
        if frame.width != self.width || frame.height != self.height {
            return Ok(());
        }
        self.stdin
            .write_all(frame.rgb.as_slice())
            .await
            .context("write frame to ffmpeg stdin")?;
        Ok(())
    }

    async fn shutdown(&mut self) {
        // Closing stdin flushes ffmpeg's muxer and writes the trailing tags.
        // We replace stdin with a throwaway pipe so the drop closes it.
        let _ = self.stdin.shutdown().await;
        match tokio::time::timeout(Duration::from_secs(5), self.child.wait()).await {
            Ok(Ok(status)) => info!(?status, "recorder: ffmpeg exited"),
            Ok(Err(e)) => warn!(error = %e, "recorder: ffmpeg wait failed"),
            Err(_) => {
                warn!("recorder: ffmpeg did not exit in 5s, killing");
                let _ = self.child.kill().await;
            }
        }
    }
}

fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}
