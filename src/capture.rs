//! Live input: what the computer is playing (a sink's monitor) or a
//! microphone, through PulseAudio (PipeWire answers as PulseAudio too).
//!
//! The capture callback writes into a `Ring`: the last ~44 s of stereo
//! frames, readable by the visuals without locks, plus a running timeline
//! overview (peak, rms per `WAVE_FRAMES`).

use crate::audio::Samples;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, HostId, StreamConfig};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::{Acquire, Relaxed, Release}};
use std::sync::Arc;

/// Frames kept (a power of two; ~44 s at 48 kHz).
const CAP: usize = 1 << 21;
/// Frames per overview bucket (as for songs).
pub const WAVE_FRAMES: usize = 1024;
/// Overview buckets kept.
const WAVE_CAP: usize = 4096;

/// A source that can be captured.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceInfo {
    /// PulseAudio's name for it (stable between runs).
    pub id: String,
    /// What to call it: the output's or input's own description.
    pub name: String,
    /// A monitor: whatever is played on that output.
    pub monitor: bool,
}

impl SourceInfo {
    /// Second line in the list: what kind of source this is.
    pub fn kind(&self) -> &'static str {
        if self.monitor { "what’s playing" } else { "input" }
    }
}

fn host() -> Result<cpal::Host, String> {
    cpal::host_from_id(HostId::PulseAudio).map_err(|e| format!("PulseAudio is not available: {e}"))
}

/// Every source, monitors (what's playing) first, the default output's
/// monitor at the top. Slow-ish (talks to the sound server): call it off the
/// UI thread.
pub fn list() -> Vec<SourceInfo> {
    let Ok(host) = host() else { return Vec::new() };
    let default_sink = host.default_output_device().and_then(|d| d.id().ok()).map(|id| id.to_string());
    let Ok(devices) = host.input_devices() else { return Vec::new() };
    let mut out: Vec<SourceInfo> = devices
        .filter_map(|d| {
            let id = pulse_name(&d.id().ok()?.to_string());
            let desc = d.description().ok().map(|x| x.name().to_string()).unwrap_or_else(|| id.clone());
            let monitor = id.ends_with(".monitor");
            let name = if monitor { desc.strip_prefix("Monitor of ").unwrap_or(&desc).to_string() } else { desc };
            Some(SourceInfo { id, name, monitor })
        })
        .collect();
    let default_monitor = default_sink.map(|s| format!("{}.monitor", pulse_name(&s)));
    out.sort_by_key(|s| (!s.monitor, Some(&s.id) != default_monitor.as_ref()));
    out
}

/// A device id's PulseAudio name (ids print as "host:name").
fn pulse_name(id: &str) -> String {
    id.split_once(':').map_or(id, |(_, n)| n).to_string()
}

/// The last `CAP` frames of a live input.
pub struct Ring {
    pub rate: u32,
    data: Box<[AtomicU32]>,
    /// Frames written so far (the newest is `written - 1`).
    written: AtomicU64,
    /// (peak, rms) per `WAVE_FRAMES`, as f32 bit pairs.
    wave: Box<[AtomicU64]>,
    waves: AtomicU64,
}

impl Ring {
    fn new(rate: u32) -> Self {
        Self {
            rate,
            data: (0..CAP * 2).map(|_| AtomicU32::new(0)).collect(),
            written: AtomicU64::new(0),
            wave: (0..WAVE_CAP).map(|_| AtomicU64::new(0)).collect(),
            waves: AtomicU64::new(0),
        }
    }

    pub fn written(&self) -> u64 {
        self.written.load(Acquire)
    }

    /// The newest frame: blocks arrive every ~10 ms, more often than frames
    /// are drawn, so this moves smoothly enough as it is.
    pub fn now(&self) -> f64 {
        self.written() as f64
    }

    /// The overview of the last `n` buckets (oldest first), scaled to the
    /// loudest peak among them; fewer at the start.
    pub fn waveform(&self, n: usize) -> Vec<(f32, f32)> {
        let total = self.waves.load(Acquire) as usize;
        let n = n.min(total).min(WAVE_CAP - 1);
        let v: Vec<(f32, f32)> = (total - n..total)
            .map(|i| {
                let b = self.wave[i % WAVE_CAP].load(Relaxed);
                (f32::from_bits(b as u32), f32::from_bits((b >> 32) as u32))
            })
            .collect();
        let top = v.iter().fold(0.05f32, |m, w| m.max(w.0));
        v.into_iter().map(|(p, r)| (p / top, r / top)).collect()
    }
}

impl Ring {
    /// Unscaled peaks of the last `n` buckets (oldest first): input level.
    pub fn peaks(&self, n: usize) -> Vec<f32> {
        let total = self.waves.load(Acquire) as usize;
        let n = n.min(total).min(WAVE_CAP - 1);
        (total - n..total).map(|i| f32::from_bits(self.wave[i % WAVE_CAP].load(Relaxed) as u32)).collect()
    }
}

impl Samples for Ring {
    #[inline]
    fn sample(&self, i: i64, ch: usize) -> f32 {
        let w = self.written.load(Relaxed);
        if i < 0 || i as u64 >= w || w - i as u64 >= CAP as u64 {
            return 0.0;
        }
        f32::from_bits(self.data[(i as usize % CAP) * 2 + ch].load(Relaxed))
    }
}

/// The capture side of a `Ring` (lives in the audio callback).
struct Writer {
    ring: Arc<Ring>,
    acc: (f32, f32, usize),
}

impl Writer {
    fn push(&mut self, input: &[f32], channels: usize) {
        let r = &self.ring;
        let mut w = r.written.load(Relaxed);
        for f in input.chunks_exact(channels) {
            let (l, rr) = if channels == 1 { (f[0], f[0]) } else { (f[0], f[1]) };
            let i = (w as usize % CAP) * 2;
            r.data[i].store(l.to_bits(), Relaxed);
            r.data[i + 1].store(rr.to_bits(), Relaxed);
            w += 1;
            let m = 0.5 * (l + rr);
            self.acc.0 = self.acc.0.max(m.abs());
            self.acc.1 += m * m;
            self.acc.2 += 1;
            if self.acc.2 == WAVE_FRAMES {
                let (peak, sum, n) = std::mem::take(&mut self.acc);
                let rms = (sum / n as f32).sqrt();
                let k = r.waves.load(Relaxed);
                r.wave[k as usize % WAVE_CAP].store(peak.to_bits() as u64 | (rms.to_bits() as u64) << 32, Relaxed);
                r.waves.store(k + 1, Release);
            }
        }
        r.written.store(w, Release);
    }
}

/// A running capture.
pub struct Capture {
    _stream: cpal::Stream,
    pub ring: Arc<Ring>,
    pub source: SourceInfo,
    /// While paused the visuals hold this frame (capture continues).
    frozen: Option<f64>,
}

impl Capture {
    pub fn start(source: &SourceInfo) -> Result<Self, String> {
        let host = host()?;
        let device = host
            .input_devices()
            .map_err(|e| e.to_string())?
            .find(|d| d.id().is_ok_and(|id| pulse_name(&id.to_string()) == source.id))
            .ok_or_else(|| format!("{} is gone", source.name))?;
        let default = device.default_input_config().map_err(|e| format!("cannot capture {}: {e}", source.name))?;
        let channels = default.channels().clamp(1, 2);
        let rate = default.sample_rate();
        let ring = Arc::new(Ring::new(rate));
        let build = |buffer_size| {
            let config = StreamConfig { channels, sample_rate: rate, buffer_size };
            let mut writer = Writer { ring: ring.clone(), acc: (0.0, 0.0, 0) };
            let ch = channels as usize;
            device.build_input_stream::<f32, _, _>(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| writer.push(data, ch),
                |e| eprintln!("mviz: capture error: {e}"),
                None,
            )
        };
        // Small blocks (~10 ms) keep the picture close to the sound.
        let stream = build(BufferSize::Fixed(rate / 100))
            .or_else(|_| build(BufferSize::Default))
            .map_err(|e| format!("cannot capture {}: {e}", source.name))?;
        stream.play().map_err(|e| format!("cannot capture {}: {e}", source.name))?;
        Ok(Self { _stream: stream, ring, source: source.clone(), frozen: None })
    }

    /// The frame the visuals show.
    pub fn position(&self) -> f64 {
        self.frozen.unwrap_or_else(|| self.ring.now())
    }

    pub fn paused(&self) -> bool {
        self.frozen.is_some()
    }

    pub fn toggle_pause(&mut self) {
        self.frozen = if self.frozen.is_some() { None } else { Some(self.ring.now()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_reads_back_recent_frames_only() {
        let ring = Arc::new(Ring::new(48000));
        let mut w = Writer { ring: ring.clone(), acc: (0.0, 0.0, 0) };
        let input: Vec<f32> = (0..4000).flat_map(|i| [i as f32, -(i as f32)]).collect();
        w.push(&input, 2);
        assert_eq!(ring.written(), 4000);
        assert_eq!(ring.sample(10, 0), 10.0);
        assert_eq!(ring.sample(10, 1), -10.0);
        assert_eq!(ring.sample(4000, 0), 0.0, "not written yet");
        assert_eq!(ring.sample(-1, 0), 0.0);
        assert_eq!(ring.waveform(100).len(), 3, "three full buckets");
    }

    #[test]
    fn mono_is_duplicated() {
        let ring = Arc::new(Ring::new(48000));
        let mut w = Writer { ring: ring.clone(), acc: (0.0, 0.0, 0) };
        w.push(&[0.5, 0.25], 1);
        assert_eq!((ring.sample(1, 0), ring.sample(1, 1)), (0.25, 0.25));
    }
}
