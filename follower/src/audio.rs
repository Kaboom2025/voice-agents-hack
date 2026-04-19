//! Microphone capture via `cpal`. A dedicated audio thread receives
//! callbacks from the OS audio subsystem and accumulates 16 kHz mono
//! f32 samples into a thread-safe ring buffer. The main chunk loop
//! calls [`AudioBuffer::drain`] at chunk boundaries to grab the audio
//! segment for that window.
//!
//! **macOS first run:** grant microphone permission to your terminal app
//! (System Settings > Privacy & Security > Microphone).

use std::io::Cursor;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tracing::{info, warn};

/// Target sample rate for embeddings (PRD §5.1: 16 kHz mono).
pub const SAMPLE_RATE: u32 = 16_000;

/// Thread-safe buffer that accumulates audio samples between chunk
/// boundaries. [`drain`] returns everything captured since the last
/// drain, resetting the buffer.
#[derive(Clone)]
pub struct AudioBuffer {
    inner: Arc<Mutex<Vec<f32>>>,
}

impl AudioBuffer {
    fn new() -> Self {
        // Pre-allocate for ~10 s at 16 kHz.
        Self {
            inner: Arc::new(Mutex::new(Vec::with_capacity(SAMPLE_RATE as usize * 10))),
        }
    }

    /// Take all accumulated samples since the last drain.
    pub fn drain(&self) -> Vec<f32> {
        let mut buf = self.inner.lock().unwrap();
        std::mem::take(&mut *buf)
    }

    /// Push new samples from the audio callback. Caps at ~30 s to
    /// prevent unbounded growth if the embed loop stalls.
    fn push(&self, samples: &[f32]) {
        let mut buf = self.inner.lock().unwrap();
        const MAX_SAMPLES: usize = SAMPLE_RATE as usize * 30;
        if buf.len() + samples.len() > MAX_SAMPLES {
            let excess = (buf.len() + samples.len()) - MAX_SAMPLES;
            let len = buf.len();
            buf.drain(..excess.min(len));
        }
        buf.extend_from_slice(samples);
    }
}

/// Handle returned by [`start_capture`]. Dropping it stops the audio
/// stream. Clone the [`buffer`] field to share it with the embed loop.
pub struct AudioHandle {
    pub buffer: AudioBuffer,
    // cpal::Stream stops on drop; we keep it alive here.
    _stream: cpal::Stream,
}

/// Open the default input device and start capturing audio. Returns an
/// [`AudioHandle`] whose buffer accumulates mono f32 samples at
/// [`SAMPLE_RATE`] Hz. If the hardware runs at a different rate we do
/// naive linear resampling — good enough for embedding, not for
/// playback.
pub fn start_capture() -> Result<AudioHandle> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default audio input device")?;

    let name = device.name().unwrap_or_else(|_| "<unknown>".into());
    info!(device = %name, "opening audio input");

    let config = device
        .default_input_config()
        .context("no default audio input config")?;

    let source_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    info!(
        sample_rate = source_rate,
        channels,
        format = ?config.sample_format(),
        "audio input config",
    );

    let buffer = AudioBuffer::new();
    let buf_clone = buffer.clone();

    let err_fn = |err: cpal::StreamError| warn!(%err, "audio stream error");

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mono = downmix_to_mono(data, channels);
                let resampled = resample_if_needed(&mono, source_rate);
                buf_clone.push(&resampled);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => {
            let buf_clone2 = buffer.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let floats: Vec<f32> =
                        data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    let mono = downmix_to_mono(&floats, channels);
                    let resampled = resample_if_needed(&mono, source_rate);
                    buf_clone2.push(&resampled);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let buf_clone2 = buffer.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let floats: Vec<f32> = data
                        .iter()
                        .map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0)
                        .collect();
                    let mono = downmix_to_mono(&floats, channels);
                    let resampled = resample_if_needed(&mono, source_rate);
                    buf_clone2.push(&resampled);
                },
                err_fn,
                None,
            )?
        }
        other => anyhow::bail!("unsupported audio sample format: {other:?}"),
    };

    stream.play().context("start audio stream")?;
    info!("audio capture started");

    Ok(AudioHandle {
        buffer,
        _stream: stream,
    })
}

// ──────────────────────── helpers ────────────────────────

fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    data.chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Naive linear-interpolation resampling to [`SAMPLE_RATE`].
fn resample_if_needed(data: &[f32], source_rate: u32) -> Vec<f32> {
    if source_rate == SAMPLE_RATE || data.is_empty() {
        return data.to_vec();
    }
    let ratio = source_rate as f64 / SAMPLE_RATE as f64;
    let out_len = (data.len() as f64 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let idx = src.floor() as usize;
        let frac = src - idx as f64;
        let sample = if idx + 1 < data.len() {
            data[idx] as f64 * (1.0 - frac) + data[idx + 1] as f64 * frac
        } else if idx < data.len() {
            data[idx] as f64
        } else {
            0.0
        };
        out.push(sample as f32);
    }
    out
}

/// Encode mono f32 audio samples to a 16-bit PCM WAV buffer in memory.
/// Returns the complete WAV file as bytes (with RIFF header, fmt chunk, data chunk).
pub fn encode_wav_bytes(samples: &[f32]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let capacity = 44 + samples.len() * 2;
    let buf = Vec::with_capacity(capacity);
    let mut cursor = Cursor::new(buf);
    let mut writer = hound::WavWriter::new(&mut cursor, spec)
        .context("failed to create in-memory WAV writer")?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(sample)?;
    }
    writer
        .finalize()
        .context("failed to finalize in-memory WAV writer")?;
    Ok(cursor.into_inner())
}

/// Write a mono f32 audio buffer to a 16-bit PCM WAV file at
/// [`SAMPLE_RATE`]. Used to produce a file path for `cactus_audio_embed`.
pub fn write_wav(path: &std::path::Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for &s in samples {
        // Clamp to [-1, 1] then scale to i16 range.
        let clamped = s.clamp(-1.0, 1.0);
        let sample = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(sample)?;
    }
    writer
        .finalize()
        .with_context(|| format!("finalize {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_wav_bytes_has_riff_header() {
        let samples = vec![0.0; 100];
        let buf = encode_wav_bytes(&samples).unwrap();
        assert!(buf.starts_with(b"RIFF"));
        assert!(buf.len() >= 44);
    }

    #[test]
    fn encode_wav_bytes_has_wave_marker() {
        let samples = vec![0.0; 100];
        let buf = encode_wav_bytes(&samples).unwrap();
        assert_eq!(&buf[8..12], b"WAVE");
    }

    #[test]
    fn encode_wav_bytes_length_matches_samples() {
        let samples = vec![0.5; 1000];
        let buf = encode_wav_bytes(&samples).unwrap();
        // RIFF header (44 bytes) + 2 bytes per i16 sample
        assert_eq!(buf.len(), 44 + samples.len() * 2);
    }
}
