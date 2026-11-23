//! Decoding, output-device setup and gapless looping playback.
//!
//! The song is decoded in the background into a `Track` while it already
//! plays: decoding is hundreds of times faster than real time, so playback
//! starts after the first fraction of a second and never catches up. The
//! audio callback walks through the track (wrapping at the end once its
//! length is known) and publishes where it is, so the visuals can compute
//! exactly which sample is audible right now.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, FromSample, SampleFormat, SizedSample, StreamConfig, SupportedBufferSize};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::{Acquire, Relaxed, Release}};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, StandardTag, StandardVisualKey};

/// Frames per chunk (about 0.7 s). A chunk becomes readable once complete.
const CHUNK: usize = 32768;
/// Room for about three hours at 48 kHz.
const MAX_CHUNKS: usize = 1 << 14;
/// Frames per waveform bucket (the timeline overview).
const WAVE_FRAMES: usize = 1024;

/// Stereo audio the visuals can read at any (signed) frame index: a song,
/// or the live capture buffer. Frames that don't exist read as silence.
pub trait Samples {
    fn sample(&self, i: i64, ch: usize) -> f32;
}

impl<T: Samples + ?Sized> Samples for Arc<T> {
    #[inline]
    fn sample(&self, i: i64, ch: usize) -> f32 {
        (**self).sample(i, ch)
    }
}

impl Samples for Track {
    #[inline]
    fn sample(&self, i: i64, ch: usize) -> f32 {
        Track::sample(self, i, ch)
    }
}

/// The output device. Pinned to ALSA (which reaches PipeWire / PulseAudio
/// through their ALSA plugins): the PulseAudio host is compiled in for
/// capture only, and playback timing is tuned for ALSA.
fn output_device() -> Option<cpal::Device> {
    cpal::host_from_id(cpal::HostId::Alsa)
        .unwrap_or_else(|_| cpal::default_host())
        .default_output_device()
}

/// A song's interleaved stereo samples, filled while it is already in use.
/// One writer (the decoder) publishes complete chunks; any number of readers
/// (audio callback, visuals, analysis) see only published data.
pub struct Track {
    pub rate: u32,
    chunks: Box<[OnceLock<Box<[f32]>>]>,
    /// Frames readable so far.
    available: AtomicUsize,
    /// Set once decoding has finished: the final length in frames.
    total: AtomicUsize,
    done: AtomicBool,
    /// Expected length in frames (from the file header; 0 = unknown).
    estimate: usize,
    /// (peak, rms) per `WAVE_FRAMES` frames, as decoded so far.
    wave: Mutex<Vec<(f32, f32)>>,
}

impl Track {
    fn new(rate: u32, estimate: usize) -> Self {
        Self {
            rate,
            chunks: (0..MAX_CHUNKS).map(|_| OnceLock::new()).collect(),
            available: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            done: AtomicBool::new(false),
            estimate,
            wave: Mutex::new(Vec::new()),
        }
    }

    /// A complete track from samples already in memory (tests).
    #[cfg(test)]
    pub fn from_interleaved(samples: &[f32], rate: u32) -> Arc<Track> {
        let track = Arc::new(Track::new(rate, samples.len() / 2));
        let mut w = TrackWriter::new(track.clone());
        for f in samples.chunks_exact(2) {
            w.push([f[0], f[1]]);
        }
        w.finish();
        track
    }

    /// Frames readable now.
    pub fn available(&self) -> usize {
        self.available.load(Acquire)
    }

    /// The final length, once decoding has finished.
    pub fn total(&self) -> Option<usize> {
        self.done.load(Acquire).then(|| self.total.load(Relaxed))
    }

    /// Best known length in frames: final, or the header's estimate.
    pub fn length(&self) -> usize {
        self.total().unwrap_or_else(|| self.estimate.max(self.available()))
    }

    /// One stereo frame, if decoded.
    #[inline]
    pub fn frame(&self, i: usize) -> Option<[f32; 2]> {
        let chunk = self.chunks.get(i / CHUNK)?.get()?;
        let j = (i % CHUNK) * 2;
        Some([*chunk.get(j)?, *chunk.get(j + 1)?])
    }

    /// One channel of a frame at a signed index: wraps around the loop once
    /// the length is known, and is silent where nothing is decoded (yet).
    #[inline]
    pub fn sample(&self, i: i64, ch: usize) -> f32 {
        let i = match self.total() {
            Some(n) if n > 0 => i.rem_euclid(n as i64) as usize,
            _ if i < 0 => return 0.0,
            _ => i as usize,
        };
        self.frame(i).map_or(0.0, |f| f[ch])
    }

    /// Every frame, in order (only once decoding has finished).
    pub fn frames(&self) -> impl Iterator<Item = [f32; 2]> + '_ {
        let n = self.total().unwrap_or(0);
        (0..n.div_ceil(CHUNK)).flat_map(move |c| {
            let chunk = self.chunks[c].get().map(|b| &b[..]).unwrap_or(&[]);
            chunk.chunks_exact(2).map(|f| [f[0], f[1]])
        })
    }

    /// The timeline overview so far, scaled to the loudest peak, and the
    /// number of buckets the whole song is expected to have.
    pub fn waveform(&self) -> (Vec<(f32, f32)>, usize) {
        let wave = self.wave.lock().unwrap().clone();
        let top = wave.iter().fold(1e-4f32, |m, w| m.max(w.0));
        let expected = self.length().div_ceil(WAVE_FRAMES).max(wave.len());
        (wave.into_iter().map(|(p, r)| (p / top, r / top)).collect(), expected)
    }
}

/// The decoder's side of a `Track`.
struct TrackWriter {
    track: Arc<Track>,
    buf: Vec<f32>,
    chunk: usize,
    frames: usize,
    wave_acc: (f32, f32, usize),
}

impl TrackWriter {
    fn new(track: Arc<Track>) -> Self {
        Self { track, buf: Vec::with_capacity(CHUNK * 2), chunk: 0, frames: 0, wave_acc: (0.0, 0.0, 0) }
    }

    fn push(&mut self, f: [f32; 2]) {
        self.buf.extend_from_slice(&f);
        self.frames += 1;
        let m = 0.5 * (f[0] + f[1]);
        let acc = &mut self.wave_acc;
        acc.0 = acc.0.max(m.abs());
        acc.1 += m * m;
        acc.2 += 1;
        if acc.2 == WAVE_FRAMES {
            self.flush_wave();
        }
        if self.buf.len() == CHUNK * 2 {
            self.publish();
        }
    }

    fn flush_wave(&mut self) {
        let (peak, sum, n) = std::mem::take(&mut self.wave_acc);
        if n > 0 {
            self.track.wave.lock().unwrap().push((peak, (sum / n as f32).sqrt()));
        }
    }

    fn publish(&mut self) {
        if self.buf.is_empty() || self.chunk >= MAX_CHUNKS {
            return;
        }
        let data = std::mem::replace(&mut self.buf, Vec::with_capacity(CHUNK * 2)).into_boxed_slice();
        let frames = data.len() / 2;
        let _ = self.track.chunks[self.chunk].set(data);
        self.chunk += 1;
        self.track.available.fetch_add(frames, Release);
    }

    fn finish(mut self) {
        self.flush_wave();
        self.publish();
        self.track.total.store(self.track.available(), Relaxed);
        self.track.done.store(true, Release);
    }
}

/// Streaming cubic (Catmull-Rom) resampler for stereo frames.
struct Resampler {
    /// Input frames per output frame.
    step: f64,
    /// Position of the next output frame within `buf`.
    t: f64,
    buf: Vec<[f32; 2]>,
}

impl Resampler {
    fn new(from: u32, to: u32) -> Self {
        Self { step: from as f64 / to as f64, t: 0.0, buf: Vec::new() }
    }

    fn interpolate(&self, x: f64) -> [f32; 2] {
        let n = self.buf.len() as isize;
        let i = x.floor() as isize;
        let t = (x - i as f64) as f32;
        let get = |k: isize| self.buf[k.clamp(0, n - 1) as usize];
        let (p0, p1, p2, p3) = (get(i - 1), get(i), get(i + 1), get(i + 2));
        let mut out = [0.0; 2];
        for c in 0..2 {
            let (a, b, cc, d) = (p0[c], p1[c], p2[c], p3[c]);
            out[c] = b + 0.5 * t * (cc - a + t * (2.0 * a - 5.0 * b + 4.0 * cc - d + t * (3.0 * (b - cc) + d - a)));
        }
        out
    }

    fn push(&mut self, input: &[[f32; 2]], out: &mut TrackWriter) {
        self.buf.extend_from_slice(input);
        // Each output needs two input frames after it.
        while self.t + 2.0 < self.buf.len() as f64 {
            out.push(self.interpolate(self.t));
            self.t += self.step;
        }
        let keep_from = (self.t.floor() as usize).saturating_sub(1);
        self.buf.drain(..keep_from);
        self.t -= keep_from as f64;
    }

    fn finish(&mut self, out: &mut TrackWriter) {
        while self.t < self.buf.len() as f64 {
            out.push(self.interpolate(self.t));
            self.t += self.step;
        }
    }
}

/// What the file says about itself (for the system's media controls).
#[derive(Clone, Debug, Default)]
pub struct TrackInfo {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    /// Embedded cover art: (MIME type, image bytes).
    pub cover: Option<(String, Vec<u8>)>,
}

/// An opened audio file, ready to decode.
pub struct Source {
    pub info: TrackInfo,
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    pub rate: u32,
    /// Length in frames, if the file says.
    pub frames: Option<u64>,
}

impl Source {
    /// Opens a file and reads its header (fast; nothing is decoded yet).
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }
        let mut format = symphonia::default::get_probe()
            .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
            .map_err(|e| format!("unsupported format: {e}"))?;
        let track = format.default_track(TrackType::Audio).ok_or("no audio track found")?;
        let track_id = track.id;
        let frames = track.num_frames;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or("track has no audio codec parameters")?;
        let rate = params.sample_rate.ok_or("unknown sample rate")?;
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .map_err(|e| format!("unsupported codec: {e}"))?;
        let info = read_info(format.as_mut());
        Ok(Self { info, format, decoder, track_id, rate, frames })
    }

    /// An empty track at `rate` (the output rate), sized from the header.
    pub fn track(&self, rate: u32) -> Arc<Track> {
        let estimate = self.frames.map_or(0, |n| (n as f64 * rate as f64 / self.rate as f64) as usize);
        Arc::new(Track::new(rate, estimate))
    }

    /// Decodes everything into `track`, resampling to its rate.
    pub fn decode_into(mut self, track: Arc<Track>) -> Result<(), String> {
        let mut writer = TrackWriter::new(track.clone());
        let mut resampler = (self.rate != track.rate).then(|| Resampler::new(self.rate, track.rate));
        let mut buf: Vec<f32> = Vec::new();
        let mut frames: Vec<[f32; 2]> = Vec::new();
        let mut any = false;
        loop {
            let packet = match self.format.next_packet() {
                Ok(Some(p)) => p,
                Ok(None) => break,
                Err(e) if !any => return Err(format!("read error: {e}")),
                Err(_) => break,
            };
            if packet.track_id != self.track_id {
                continue;
            }
            match self.decoder.decode(&packet) {
                Ok(audio) => {
                    let ch = audio.spec().channels().count().max(1);
                    buf.resize(audio.samples_interleaved(), 0.0);
                    audio.copy_to_slice_interleaved(&mut buf);
                    frames.clear();
                    frames.extend(buf.chunks_exact(ch).map(|f| if ch == 1 { [f[0], f[0]] } else { [f[0], f[1]] }));
                    match resampler.as_mut() {
                        Some(r) => r.push(&frames, &mut writer),
                        None => frames.iter().for_each(|&f| writer.push(f)),
                    }
                    any = true;
                }
                Err(SymError::DecodeError(_)) => continue,
                Err(e) if !any => return Err(format!("decode error: {e}")),
                Err(_) => break,
            }
        }
        if let Some(r) = resampler.as_mut() {
            r.finish(&mut writer);
        }
        writer.finish();
        if track.available() == 0 {
            return Err("file contains no audio".into());
        }
        Ok(())
    }
}

/// Title, artist, album and front cover from the file's tags.
fn read_info(format: &mut dyn FormatReader) -> TrackInfo {
    let mut info = TrackInfo::default();
    let mut meta = format.metadata();
    let Some(rev) = meta.skip_to_latest() else { return info };
    let tags = rev.media.tags.iter().chain(rev.per_track.iter().flat_map(|t| t.metadata.tags.iter()));
    for tag in tags {
        match &tag.std {
            Some(StandardTag::TrackTitle(v)) if info.title.is_none() => info.title = Some(v.to_string()),
            Some(StandardTag::Artist(v)) if info.artist.is_none() => info.artist = Some(v.to_string()),
            Some(StandardTag::Album(v)) if info.album.is_none() => info.album = Some(v.to_string()),
            _ => {}
        }
    }
    let visuals: Vec<_> = rev.media.visuals.iter().chain(rev.per_track.iter().flat_map(|t| t.metadata.visuals.iter())).collect();
    let front = visuals.iter().find(|v| v.usage == Some(StandardVisualKey::FrontCover)).or(visuals.first());
    if let Some(v) = front {
        let mime = v.media_type.clone().unwrap_or_else(|| "image/jpeg".into());
        info.cover = Some((mime, v.data.to_vec()));
    }
    info
}

/// Decodes a whole file at its own rate (tests and offline tools).
#[cfg(test)]
pub fn load(path: &Path) -> Result<Arc<Track>, String> {
    let source = Source::open(path)?;
    let track = source.track(source.rate);
    source.decode_into(track.clone())?;
    Ok(track)
}

/// Picks an output config, preferring the song's own sample rate (no resampling).
pub fn pick_config(file_rate: u32) -> Result<(StreamConfig, SampleFormat), String> {
    let device = output_device().ok_or("no audio output device")?;
    let default = device.default_output_config().map_err(|e| format!("no output config: {e}"))?;
    let mut best: Option<(i32, cpal::SupportedStreamConfig)> = None;
    if let Ok(configs) = device.supported_output_configs() {
        for c in configs {
            if c.min_sample_rate() > file_rate || c.max_sample_rate() < file_rate || !is_supported(c.sample_format()) {
                continue;
            }
            let score = 4 * (c.channels() == 2) as i32
                + 2 * (c.channels() >= 2) as i32
                + (c.sample_format() == SampleFormat::F32) as i32;
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, c.with_sample_rate(file_rate)));
            }
        }
    }
    let chosen = best.map(|b| b.1).unwrap_or(default);
    let mut config = chosen.config();
    config.buffer_size = match chosen.buffer_size() {
        SupportedBufferSize::Range { min, max } => BufferSize::Fixed(1024.clamp(*min, *max)),
        SupportedBufferSize::Unknown => BufferSize::Default,
    };
    Ok((config, chosen.sample_format()))
}

fn is_supported(f: SampleFormat) -> bool {
    matches!(f, SampleFormat::F32 | SampleFormat::I16 | SampleFormat::I32 | SampleFormat::U16)
}

const NO_SEEK: u64 = u64::MAX;

/// State shared between the audio callback and the UI thread.
struct Shared {
    /// Next frame the callback will write.
    pos: AtomicU64,
    /// Frame at the start of the most recent callback buffer.
    buf_start: AtomicU64,
    /// When that callback ran (ns since `epoch`) and how long until its first frame is heard.
    cb_nanos: AtomicU64,
    latency_nanos: AtomicU64,
    paused: AtomicBool,
    /// Timeline is being dragged: stay silent until released.
    scrub: AtomicBool,
    volume: AtomicU32,
    seek: AtomicU64,
    /// The song to continue with at the end of this one (gapless), instead
    /// of looping. Taken by the callback (with `try_lock`: it never waits).
    next: Mutex<Option<Arc<Track>>>,
    /// Counts the callback's switches to a queued song.
    switches: AtomicU64,
}

pub struct Player {
    _stream: cpal::Stream,
    shared: Arc<Shared>,
    epoch: Instant,
    pub rate: u32,
    pub track: Arc<Track>,
    /// While paused the visuals hold this exact frame.
    frozen: Option<f64>,
    /// While the timeline is dragged the visuals show this frame.
    preview: Option<f64>,
    /// The queued next song (also in `shared.next` until the callback takes it).
    queued: Option<Arc<Track>>,
    switches_seen: u64,
}

impl Player {
    /// Starts playback of `track` (which may still be decoding) at `volume` (0..=1).
    pub fn start(track: Arc<Track>, config: StreamConfig, format: SampleFormat, volume: f32) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            pos: AtomicU64::new(0),
            buf_start: AtomicU64::new(0),
            cb_nanos: AtomicU64::new(0),
            latency_nanos: AtomicU64::new(0),
            paused: AtomicBool::new(false),
            scrub: AtomicBool::new(false),
            volume: AtomicU32::new(volume.clamp(0.0, 1.0).to_bits()),
            seek: AtomicU64::new(NO_SEEK),
            next: Mutex::new(None),
            switches: AtomicU64::new(0),
        });
        let epoch = Instant::now();
        let device = output_device().ok_or("no audio output device")?;
        let build = |config: StreamConfig| -> Result<cpal::Stream, cpal::Error> {
            let (t, sh) = (track.clone(), shared.clone());
            match format {
                SampleFormat::F32 => build::<f32>(&device, config, sh, t, epoch),
                SampleFormat::I16 => build::<i16>(&device, config, sh, t, epoch),
                SampleFormat::I32 => build::<i32>(&device, config, sh, t, epoch),
                _ => build::<u16>(&device, config, sh, t, epoch),
            }
        };
        let stream = match build(config.clone()) {
            Ok(s) => s,
            // Some backends reject a fixed buffer size; fall back to their default.
            Err(_) => build(StreamConfig { buffer_size: BufferSize::Default, ..config.clone() })
                .map_err(|e| format!("cannot open audio stream: {e}"))?,
        };
        stream.play().map_err(|e| format!("cannot start audio: {e}"))?;
        Ok(Self {
            _stream: stream,
            shared,
            epoch,
            rate: config.sample_rate,
            track,
            frozen: None,
            preview: None,
            queued: None,
            switches_seen: 0,
        })
    }

    /// Plays `track` (decoded at this player's rate) after the current song
    /// ends, without a gap, instead of looping.
    pub fn queue(&mut self, track: Arc<Track>) {
        *self.shared.next.lock().unwrap() = Some(track.clone());
        self.queued = Some(track);
    }

    /// Takes back the queued song (the order changed). Returns false if the
    /// callback has already moved on to it: `poll_switch` will report that.
    pub fn unqueue(&mut self) -> bool {
        let taken = self.shared.next.lock().unwrap().take().is_some();
        if taken {
            self.queued = None;
        }
        self.queued.is_none()
    }

    /// Checks whether playback has moved on to the queued song; if so it is
    /// now `self.track`, and this returns true (once).
    pub fn poll_switch(&mut self) -> bool {
        let n = self.shared.switches.load(Acquire);
        if n == self.switches_seen {
            return false;
        }
        self.switches_seen = n;
        if let Some(t) = self.queued.take() {
            self.track = t;
        }
        self.preview = None;
        true
    }

    /// The frame currently coming out of the speakers (fractional, within the track).
    pub fn position(&self) -> f64 {
        if let Some(f) = self.preview.or(self.frozen) {
            return f;
        }
        let cb = self.shared.cb_nanos.load(Relaxed);
        if cb == 0 {
            return 0.0;
        }
        let start = self.shared.buf_start.load(Relaxed) as f64;
        let lat = self.shared.latency_nanos.load(Relaxed) as f64;
        let now = self.epoch.elapsed().as_nanos() as f64;
        // Clamp so a stalled callback cannot make the visuals run away.
        let since_cb = (now - cb as f64).clamp(0.0, 1e8);
        let f = start + (since_cb - lat) / 1e9 * self.rate as f64;
        match self.track.total() {
            Some(n) if n > 0 => f.rem_euclid(n as f64),
            _ => f.max(0.0),
        }
    }

    /// Song length in seconds (the header's estimate until decoding has finished).
    pub fn duration(&self) -> f64 {
        self.track.length().max(1) as f64 / self.rate as f64
    }

    pub fn paused(&self) -> bool {
        self.frozen.is_some()
    }

    pub fn toggle_pause(&mut self) {
        if self.frozen.is_some() {
            self.frozen = None;
            self.shared.paused.store(false, Relaxed);
        } else {
            self.frozen = Some(self.position());
            self.shared.paused.store(true, Relaxed);
        }
    }

    /// A frame for `secs`, kept within what has been decoded so far.
    fn secs_to_frame(&self, secs: f64) -> u64 {
        ((secs.max(0.0) * self.rate as f64) as u64).min(self.track.available().max(1) as u64 - 1)
    }

    /// Jumps to `secs` (with the usual click-free fade).
    pub fn seek(&mut self, secs: f64) {
        let frame = self.secs_to_frame(secs);
        self.shared.seek.store(frame, Relaxed);
        if self.frozen.is_some() {
            self.frozen = Some(frame as f64);
        }
    }

    /// Timeline drag in progress: audio fades out and the visuals follow the cursor.
    pub fn scrub(&mut self, secs: f64) {
        self.preview = Some(self.secs_to_frame(secs) as f64);
        self.shared.scrub.store(true, Relaxed);
    }

    /// Timeline released: jump there and fade back in.
    pub fn end_scrub(&mut self) {
        if let Some(frame) = self.preview.take() {
            self.shared.seek.store(frame as u64, Relaxed);
            if self.frozen.is_some() {
                self.frozen = Some(frame);
            }
        }
        // Release: the callback sees the seek no later than it sees scrubbing end.
        self.shared.scrub.store(false, Release);
    }

    /// Volume knob 0..=1 (applied as gain = knob², which feels linear to the ear).
    pub fn volume(&self) -> f32 {
        f32::from_bits(self.shared.volume.load(Relaxed))
    }

    pub fn set_volume(&self, v: f32) {
        self.shared.volume.store(v.clamp(0.0, 1.0).to_bits(), Relaxed);
    }
}

fn build<T: SizedSample + FromSample<f32>>(
    device: &cpal::Device,
    config: StreamConfig,
    shared: Arc<Shared>,
    mut track: Arc<Track>,
    epoch: Instant,
) -> Result<cpal::Stream, cpal::Error> {
    let ch = config.channels as usize;
    let mut gain = 0.0f32;
    let mut pending_seek: Option<u64> = None;
    device.build_output_stream(
        config,
        move |out: &mut [T], info: &cpal::OutputCallbackInfo| {
            let ts = info.timestamp();
            let silent = shared.paused.load(Relaxed) || shared.scrub.load(Acquire);
            let seek = shared.seek.swap(NO_SEEK, Relaxed);
            if seek != NO_SEEK {
                pending_seek = Some(seek);
            }
            let mut total = track.total();
            let mut pos = shared.pos.load(Relaxed) as usize;
            if let Some(n) = total.filter(|&n| n > 0) {
                pos %= n;
            }
            shared.buf_start.store(pos as u64, Relaxed);
            shared.latency_nanos.store(ts.playback.duration_since(ts.callback).as_nanos() as u64, Relaxed);
            shared.cb_nanos.store(epoch.elapsed().as_nanos().max(1) as u64, Relaxed);

            let vol = f32::from_bits(shared.volume.load(Relaxed));
            for frame in out.chunks_mut(ch) {
                // Seeks fade out (~5 ms), jump while silent, then fade back in,
                // so jumping never cuts the waveform. Other changes glide over ~10 ms.
                let target = if silent || pending_seek.is_some() { 0.0 } else { vol * vol };
                let speed = if pending_seek.is_some() { 0.01 } else { 0.002 };
                gain += (target - gain) * speed;
                if let Some(to) = pending_seek {
                    if gain < 1e-3 {
                        pos = to as usize;
                        pending_seek = None;
                        gain = 0.0;
                    }
                }
                // Not decoded yet (only in the first instant, or after a very
                // slow disk): silence, without advancing.
                let sample = if silent && gain < 1e-4 { None } else { track.frame(pos) };
                let Some([l, r]) = sample else {
                    if silent {
                        gain = 0.0;
                    }
                    frame.fill(T::EQUILIBRIUM);
                    continue;
                };
                let (l, r) = (l * gain, r * gain);
                pos += 1;
                if total.is_some_and(|n| pos >= n) {
                    pos = 0; // gapless loop, or on to the queued song
                    if let Ok(mut next) = shared.next.try_lock() {
                        if let Some(t) = next.take() {
                            track = t;
                            total = track.total();
                            shared.switches.fetch_add(1, Release);
                        }
                    }
                }
                if ch == 1 {
                    frame[0] = T::from_sample(0.5 * (l + r));
                } else {
                    frame[0] = T::from_sample(l);
                    frame[1] = T::from_sample(r);
                    frame[2..].fill(T::EQUILIBRIUM);
                }
            }
            shared.pos.store(pos as u64, Relaxed);
        },
        |e| eprintln!("audio error: {e}"),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_publishes_whole_chunks_and_wraps_when_done() {
        let samples: Vec<f32> = (0..(CHUNK as u32 * 2 + 100) * 2).map(|i| i as f32).collect();
        let track = Track::from_interleaved(&samples, 48000);
        let n = CHUNK * 2 + 100;
        assert_eq!(track.total(), Some(n));
        assert_eq!(track.frame(CHUNK + 5), Some([((CHUNK + 5) * 2) as f32, ((CHUNK + 5) * 2 + 1) as f32]));
        // Loops: one before the start is the last frame.
        assert_eq!(track.sample(-1, 0), ((n - 1) * 2) as f32);
        assert_eq!(track.frames().count(), n);
    }

    #[test]
    fn streaming_resampler_matches_length_and_shape() {
        let track = Arc::new(Track::new(48000, 0));
        let mut w = TrackWriter::new(track.clone());
        let mut r = Resampler::new(44100, 48000);
        // 1 s of a 440 Hz sine at 44.1 kHz, fed in uneven pieces.
        let input: Vec<[f32; 2]> = (0..44100).map(|i| {
            let v = (i as f32 / 44100.0 * 440.0 * std::f32::consts::TAU).sin();
            [v, -v]
        }).collect();
        for piece in input.chunks(1000 + 37) {
            r.push(piece, &mut w);
        }
        r.finish(&mut w);
        w.finish();
        let n = track.total().unwrap();
        assert!((n as i64 - 48000).abs() <= 2, "resampled to {n} frames");
        // Still a 440 Hz sine at 48 kHz.
        for i in [100, 12345, 40000] {
            let want = (i as f32 / 48000.0 * 440.0 * std::f32::consts::TAU).sin();
            let got = track.frame(i).unwrap();
            assert!((got[0] - want).abs() < 0.01 && (got[1] + want).abs() < 0.01, "frame {i}: {got:?} vs {want}");
        }
    }
}
