use std::collections::VecDeque;
use std::sync::Mutex;

use crate::camera::CapturedFrame;

pub struct FrameBuffer {
    inner: Mutex<VecDeque<(u64, CapturedFrame)>>,
    window_ms: u64,
}

impl FrameBuffer {
    pub fn new(window_ms: u64) -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
            window_ms,
        }
    }

    pub fn push(&self, ts_ms: u64, frame: CapturedFrame) {
        let mut q = self.inner.lock().unwrap();
        q.push_back((ts_ms, frame));
        if let Some(&(newest_ts, _)) = q.back() {
            let cutoff = newest_ts.saturating_sub(self.window_ms);
            while q.front().is_some_and(|&(ts, _)| ts < cutoff) {
                q.pop_front();
            }
        }
    }

    pub fn window(&self) -> Vec<(u64, CapturedFrame)> {
        self.inner.lock().unwrap().iter().cloned().collect()
    }

    pub fn sample(&self, target_fps: f32) -> Vec<CapturedFrame> {
        let frames = self.window();
        if frames.is_empty() {
            return vec![];
        }
        let start_ts = frames.first().unwrap().0;
        let end_ts = frames.last().unwrap().0;
        let duration_secs = (end_ts - start_ts) as f32 / 1000.0;
        let n = ((target_fps * duration_secs).ceil() as usize)
            .max(1)
            .min(frames.len());
        (0..n)
            .map(|i| {
                let t = start_ts + ((i as f32 / n as f32) * (end_ts - start_ts) as f32) as u64;
                frames
                    .iter()
                    .min_by_key(|(ts, _)| (*ts as i64 - t as i64).unsigned_abs())
                    .map(|(_, f)| f.clone())
                    .unwrap()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn frame(pixel: u8) -> CapturedFrame {
        CapturedFrame {
            width: 2,
            height: 2,
            rgb: Arc::new(vec![pixel; 2 * 2 * 3]),
        }
    }

    #[test]
    fn empty_buffer_returns_empty_window() {
        let buf = FrameBuffer::new(10_000);
        assert!(buf.window().is_empty());
    }

    #[test]
    fn push_and_retrieve_single_frame() {
        let buf = FrameBuffer::new(10_000);
        buf.push(1000, frame(10));
        let w = buf.window();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].0, 1000);
    }

    #[test]
    fn frames_older_than_window_are_pruned() {
        let buf = FrameBuffer::new(5_000);
        buf.push(0, frame(1));
        buf.push(3_000, frame(2));
        buf.push(6_000, frame(3));
        let w = buf.window();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].0, 3_000);
        assert_eq!(w[1].0, 6_000);
    }

    #[test]
    fn sample_frames_returns_target_count() {
        let buf = FrameBuffer::new(10_000);
        for i in 0..20u64 {
            buf.push(i * 500, frame(i as u8));
        }
        let sampled = buf.sample(2.0);
        assert!(!sampled.is_empty());
        assert!(sampled.len() <= 20);
    }

    #[test]
    fn sample_frames_never_exceeds_available_frames() {
        let buf = FrameBuffer::new(10_000);
        buf.push(0, frame(1));
        buf.push(1_000, frame(2));
        let sampled = buf.sample(10.0);
        assert!(sampled.len() <= 2);
    }
}
