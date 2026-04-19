//! Webcam capture via `nokhwa`. A dedicated OS thread owns the camera
//! (AVFoundation on macOS is happiest not being bounced across async
//! executors) and publishes the latest decoded RGB frame through a
//! `tokio::sync::watch` channel. Consumers read the most recent frame
//! without blocking — old frames are simply overwritten, matching the
//! PRD §5.1 rule: when the embedder falls behind, drop samples first.
//!
//! **macOS first run:** the kernel prompts for camera permission. Grant
//! it to your terminal (System Settings > Privacy & Security > Camera).

use std::sync::Arc;

use anyhow::{Context, Result};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution},
    Camera,
};
use tokio::sync::watch;
use tracing::{info, warn};

/// One decoded camera frame. `rgb` is tightly packed RGB8, row-major,
/// `width * height * 3` bytes long.
#[derive(Clone)]
pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    pub rgb: Arc<Vec<u8>>,
}

impl CapturedFrame {
    /// Hash the raw pixels. Two frames with identical content hash equal;
    /// any visible scene change flips the hash.
    pub fn blake3(&self) -> blake3::Hash {
        blake3::hash(self.rgb.as_slice())
    }
}

/// Handle you hold on the async side. `watch_rx.borrow_and_update()` gives
/// you the latest frame (or `None` until the first frame lands).
pub struct CameraHandle {
    pub rx: watch::Receiver<Option<CapturedFrame>>,
    _thread: std::thread::JoinHandle<()>,
}

/// Spawn the capture thread. Returns once the camera has been opened and
/// the first frame is in flight; errors from the thread after that point
/// are logged but don't tear down the follower (we prefer to keep the
/// iroh session up and surface the problem in metrics).
pub fn spawn(device_index: u32) -> Result<CameraHandle> {
    let (tx, rx) = watch::channel(None);
    // `ready` is a one-shot we use to surface open errors synchronously —
    // otherwise the caller gets back a handle that never produces frames.
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<()>>(1);

    let thread = std::thread::Builder::new()
        .name(format!("webcam-{device_index}"))
        .spawn(move || {
            let mut camera = match open_camera(device_index) {
                Ok(c) => {
                    let _ = ready_tx.send(Ok(()));
                    c
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };

            loop {
                match capture_once(&mut camera) {
                    Ok(frame) => {
                        // `send` only fails if every receiver has been
                        // dropped, which means the follower is shutting
                        // down — at which point we should exit cleanly.
                        if tx.send(Some(frame)).is_err() {
                            info!("all frame receivers dropped, stopping capture");
                            return;
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "frame capture failed");
                        // Brief backoff so we don't hot-loop on a
                        // permanently broken camera.
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }
        })
        .context("spawn camera thread")?;

    // Block until we know whether the open succeeded.
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(CameraHandle {
            rx,
            _thread: thread,
        }),
        Ok(Err(e)) => Err(e),
        Err(_) => anyhow::bail!("camera thread panicked during init"),
    }
}

fn open_camera(device_index: u32) -> Result<Camera> {
    let index = CameraIndex::Index(device_index);

    // Prefer 1280x720 MJPEG @ 30fps — cuts Gemini upload bandwidth ~2x vs 1080p
    // while staying a near-universal webcam format. Fall back to the camera's
    // highest-framerate format if the target isn't available.
    let target = CameraFormat::new(Resolution::new(1280, 720), FrameFormat::MJPEG, 30);
    let mut camera = match Camera::new(
        index.clone(),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(target)),
    ) {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "720p MJPEG unavailable, falling back to highest-framerate format");
            Camera::new(
                index,
                RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
            )
            .with_context(|| format!("open camera index {device_index}"))?
        }
    };
    camera.open_stream().context("start camera stream")?;
    let res = camera.resolution();
    info!(
        device = device_index,
        width = res.width_x,
        height = res.height_y,
        "camera open"
    );
    Ok(camera)
}

fn capture_once(camera: &mut Camera) -> Result<CapturedFrame> {
    let buf = camera.frame().context("grab frame")?;
    let res = buf.resolution();
    let decoded = buf.decode_image::<RgbFormat>().context("decode RGB")?;
    Ok(CapturedFrame {
        width: res.width_x,
        height: res.height_y,
        rgb: Arc::new(decoded.into_raw()),
    })
}
