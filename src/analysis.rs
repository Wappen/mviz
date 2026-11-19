//! Offline analysis of the whole song, run once at load.
//!
//! Runs in the background once the song is decoded, while it already plays.
//! It finds bass onsets (whose note sets the background hue), the
//! low-passed bass envelope that lights the background, and the reference
//! level for the field's brightness scale.

use crate::audio::Track;
use crate::dsp::{hz_to_midi, parabolic, percentile, Fft, ABS_FLOOR};
use crate::modes::stereo;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

pub const HOP: usize = 512;
const N: usize = 4096;
/// Short window for onset detection: a long one "sees" a hit too early.
const ON_N: usize = 1024;
const BASS_PITCH_N: usize = 16384;

/// A bass onset; it only sets the background hue (the bass note's pitch class).
pub struct BassHit {
    pub t: f32,
    /// Pitch class 0..12 of the bass note under the hit.
    pub pc: f32,
}

pub struct Timeline {
    pub hop: f32,
    pub bass: Vec<BassHit>,
    /// Low-passed (80 Hz, Q 0.5) amplitude per hop for (left, right), ~1.0 = loud.
    pub bass_env: Vec<(f32, f32)>,
    /// Loud reference level for the stereo-field brightness scale.
    pub stereo_ref: f32,
}

/// Analyses a fully decoded track.
pub fn analyze(track: &Track, progress: &AtomicU32) -> Timeline {
    let rate = track.rate;
    let rate_f = rate as f32;
    let hop_s = HOP as f32 / rate_f;
    let mono: Vec<f32> = track.frames().map(|f| 0.5 * (f[0] + f[1])).collect();
    let len = mono.len();
    let n_hops = len / HOP + 1;
    let bin_hz = rate_f / N as f32;
    let bin = |f: f32| ((f / bin_hz).round() as usize).min(N / 2);
    let mono_at = |i: isize| if i < 0 || i as usize >= len { 0.0 } else { mono[i as usize] };

    let stereo_bins: Vec<(usize, usize, f32)> = stereo::band_edges()
        .into_iter()
        .map(|(lo, hi)| (bin(lo), bin(hi), (lo * hi).sqrt() / 1000.0))
        .filter(|(k0, k1, _)| k1 > k0)
        .collect();

    let mut fft = Fft::new(N);
    let mut mags = vec![0f32; N / 2 + 1];
    let mut on_fft = Fft::new(ON_N);
    let on_bin = |f: f32| ((f * ON_N as f32 / rate_f).round() as usize).min(ON_N / 2);
    let (ob0, ob1) = (on_bin(30.0).max(1), on_bin(150.0));
    let mut prev_log = vec![0f32; ON_N / 2 + 1];
    let mut bass_odf = Vec::with_capacity(n_hops);
    let mut stereo_levels = Vec::new();

    for i in 0..n_hops {
        let start = (i * HOP) as isize - (N / 2) as isize;
        let spec = fft.run(|j| mono_at(start + j as isize));
        for (m, c) in mags.iter_mut().zip(spec) {
            *m = c.norm();
        }

        // Onset detection function: rectified rise of log magnitude in the bass.
        let on_start = (i * HOP) as isize - (ON_N / 2) as isize;
        let on_spec = on_fft.run(|j| mono_at(on_start + j as isize));
        let mut bass_flux = 0.0;
        for k in ob0..=ob1 {
            let l = (1.0 + 1000.0 * on_spec[k].norm()).ln();
            bass_flux += (l - prev_log[k]).max(0.0);
            prev_log[k] = l;
        }
        bass_odf.push(bass_flux);

        if i % 4 == 0 {
            for &(k0, k1, tilt) in &stereo_bins {
                let e: f32 = mags[k0..k1].iter().map(|m| m * m).sum();
                if e > 1e-10 {
                    stereo_levels.push(e * tilt);
                }
            }
        }
        if i % 256 == 0 {
            progress.store((i * 900 / n_hops) as u32, Relaxed);
        }
    }

    // --- Bass hits -------------------------------------------------------
    let bass_on = pick_onsets(&bass_odf, 3, 10, 0.25, (0.09 / hop_s) as usize);
    let mut pitch_fft = Fft::new(BASS_PITCH_N);
    let pbin = rate_f / BASS_PITCH_N as f32;
    let bass: Vec<BassHit> = bass_on
        .iter()
        .map(|&(n, _)| {
            // Long window starting at the hit: fine enough to resolve bass notes.
            let s0 = (n * HOP) as isize;
            let spec = pitch_fft.run(|j| mono_at(s0 + j as isize));
            let (k0, k1) = ((30.0 / pbin) as usize, (160.0 / pbin) as usize);
            let mut best = k0.max(1);
            for k in k0.max(1)..k1 {
                if spec[k].norm() > spec[best].norm() {
                    best = k;
                }
            }
            let l = |k: usize| spec[k].norm().max(1e-12).ln();
            let kf = best as f32 + parabolic(l(best - 1), l(best), l(best + 1));
            BassHit { t: n as f32 * hop_s, pc: hz_to_midi(kf * pbin).rem_euclid(12.0) }
        })
        .collect();
    let bass_env = bass_envelope(track, rate, n_hops);

    let stereo_ref = percentile(&mut stereo_levels, 0.99).max(1e-9) * 2.0;
    progress.store(1000, Relaxed);
    Timeline { hop: hop_s, bass, bass_env, stereo_ref }
}

/// A 2nd-order low-pass (RBJ biquad) at `fc` for sample rate `rate`.
struct LowPass {
    b: [f64; 3],
    a: [f64; 2],
    x: [f64; 2],
    y: [f64; 2],
}

impl LowPass {
    fn new(fc: f64, q: f64, rate: f64) -> Self {
        let w0 = std::f64::consts::TAU * fc / rate;
        let (cw, alpha) = (w0.cos(), w0.sin() / (2.0 * q));
        let a0 = 1.0 + alpha;
        Self {
            b: [(1.0 - cw) / 2.0 / a0, (1.0 - cw) / a0, (1.0 - cw) / 2.0 / a0],
            a: [-2.0 * cw / a0, (1.0 - alpha) / a0],
            x: [0.0; 2],
            y: [0.0; 2],
        }
    }

    fn run(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.b[1] * self.x[0] + self.b[2] * self.x[1] - self.a[0] * self.y[0] - self.a[1] * self.y[1];
        self.x = [x, self.x[0]];
        self.y = [y, self.y[0]];
        y
    }
}

/// Samples per block when the bass power is reduced in rate (before the
/// envelope filter, which only keeps changes below `ENVELOPE_HZ`).
const ENV_BLOCK: usize = 32;
/// How fast the background may follow the bass. Must stay well below twice
/// the lowest bass frequency (≈ 60 Hz), so the glow follows the bass's
/// loudness, not its individual waves.
const ENVELOPE_HZ: f64 = 10.0;

/// Loudness of each channel's bass, for the background glow.
/// The bass is a 2nd-order low-pass (80 Hz, Q 0.5): flat below 80 Hz,
/// −12 dB/octave above, so sub-bass drives it and anything above ~150 Hz
/// barely does. Its power is then smoothed by a 10 Hz low-pass run forwards
/// and backwards (no delay): that removes the ripple of each bass wave
/// (twice its frequency: 60 Hz and up), which a short measuring window
/// would turn into flicker, while keeping hits and swells. Normalised so the
/// loud parts of the song sit near 1.0; one value per hop.
fn bass_envelope(track: &Track, rate: u32, n_hops: usize) -> Vec<(f32, f32)> {
    let n = track.total().unwrap_or(0);
    let blocks = n.div_ceil(ENV_BLOCK).max(1);
    let mut power = vec![[0f64; 2]; blocks];
    for ch in 0..2 {
        let mut bass = LowPass::new(80.0, 0.5, rate as f64);
        for (i, frame) in track.frames().enumerate() {
            let y = bass.run(frame[ch] as f64);
            power[i / ENV_BLOCK][ch] += y * y / ENV_BLOCK as f64;
        }
        // Zero-phase smoothing: forwards, then backwards.
        let block_rate = rate as f64 / ENV_BLOCK as f64;
        let mut smooth = LowPass::new(ENVELOPE_HZ, std::f64::consts::FRAC_1_SQRT_2, block_rate);
        for p in power.iter_mut() {
            p[ch] = smooth.run(p[ch]);
        }
        let mut smooth = LowPass::new(ENVELOPE_HZ, std::f64::consts::FRAC_1_SQRT_2, block_rate);
        for p in power.iter_mut().rev() {
            p[ch] = smooth.run(p[ch]);
        }
    }
    let rms: Vec<(f32, f32)> = (0..n_hops)
        .map(|i| {
            let p = power[((i * HOP) / ENV_BLOCK).min(blocks - 1)];
            (p[0].max(0.0).sqrt() as f32, p[1].max(0.0).sqrt() as f32)
        })
        .collect();
    let mut mono: Vec<f32> = rms.iter().map(|&(l, r)| ((l * l + r * r) / 2.0).sqrt()).collect();
    let reference = percentile(&mut mono, 0.99).max(ABS_FLOOR);
    rms.into_iter().map(|(l, r)| (l / reference, r / reference)).collect()
}

/// Peak picking with a centred (look-ahead) adaptive threshold.
/// Returns (frame, strength 0..1).
fn pick_onsets(odf: &[f32], max_w: usize, mean_w: usize, delta: f32, min_gap: usize) -> Vec<(usize, f32)> {
    let mut sorted = odf.to_vec();
    let norm = percentile(&mut sorted, 0.98).max(1e-9);
    let o: Vec<f32> = odf.iter().map(|v| v / norm).collect();
    let mut out: Vec<(usize, f32)> = Vec::new();
    for n in 0..o.len() {
        let win = |w: usize| n.saturating_sub(w)..(n + w + 1).min(o.len());
        if o[n] < 0.05 || o[win(max_w)].iter().any(|&v| v > o[n]) {
            continue;
        }
        let r = win(mean_w);
        let mean = o[r.clone()].iter().sum::<f32>() / r.len() as f32;
        if o[n] < mean + delta {
            continue;
        }
        if out.last().is_some_and(|&(last, _)| n - last < min_gap.max(1)) {
            continue;
        }
        out.push((n, o[n]));
    }
    // Rank each hit against the song's strongest hits, so a kick outshines a passing bass note.
    let mut peaks: Vec<f32> = out.iter().map(|h| h.1).collect();
    let top = percentile(&mut peaks, 0.9).max(1e-9);
    for h in &mut out {
        h.1 = (h.1 / top).min(1.0).powf(2.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// 4 s at 48 kHz: kicks on every beat (120 bpm) with an A bass (55 Hz),
    /// a held C5 (523 Hz) panned left, and closed hi-hats on the off-beats
    /// (which must not count as bass hits).
    fn synth() -> Vec<f32> {
        let rate = 48000.0;
        let n = 4 * 48000;
        let mut seed = 1u32;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / rate;
            let beat = t % 0.5;
            let kick = (TAU * 55.0 * t).sin() * (-beat * 12.0).exp() * 0.8;
            let note = if (1.0..3.0).contains(&t) { (TAU * 523.25 * t).sin() * 0.25 } else { 0.0 };
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 * 2.0 - 1.0;
            let off = (t + 0.25) % 0.5;
            let hat = noise * (-off * 60.0).exp() * 0.2 * (TAU * 9000.0 * t).sin();
            out.push(kick + note + hat);
            out.push(kick + note * 0.3 + hat);
        }
        out
    }

    #[test]
    fn finds_kicks_and_their_pitch() {
        let tl = analyze(&crate::audio::Track::from_interleaved(&synth(), 48000), &AtomicU32::new(0));
        let kicks: Vec<f32> = tl.bass.iter().map(|b| b.t).collect();
        println!("kicks {kicks:?}");
        assert!((7..=9).contains(&tl.bass.len()), "kick count {}", tl.bass.len());
        for b in &tl.bass {
            let off = (b.t + 0.25) % 0.5 - 0.25;
            assert!(off.abs() < 0.03, "kick off-beat at {}", b.t);
            assert!((b.pc - 9.0).abs() < 0.6, "bass pitch class {} (want A=9)", b.pc);
        }
    }
}

#[cfg(test)]
mod bass_glow {
    use super::*;
    use std::f32::consts::TAU;

    /// Peak-to-peak ripple of the background level (as a share of its mean)
    /// for a steady bass note: the background should not pulse with the wave.
    fn ripple(freq: f32) -> f32 {
        let rate = 48000;
        let samples: Vec<f32> = (0..rate * 3)
            .flat_map(|i| {
                let v = (TAU * freq * i as f32 / rate as f32).sin() * 0.5;
                [v, v]
            })
            .collect();
        let track = crate::audio::Track::from_interleaved(&samples, rate as u32);
        let env = bass_envelope(&track, rate as u32, samples.len() / 2 / HOP + 1);
        // Steady middle second only (no start/end transients).
        let mid: Vec<f32> = env[env.len() / 3..2 * env.len() / 3].iter().map(|e| e.0).collect();
        let (lo, hi) = mid.iter().fold((f32::MAX, 0.0f32), |a, &v| (a.0.min(v), a.1.max(v)));
        let mean = mid.iter().sum::<f32>() / mid.len() as f32;
        (hi - lo) / mean
    }

    #[test]
    fn steady_deep_bass_gives_a_steady_glow() {
        for f in [30.0, 41.0, 55.0, 70.0] {
            let r = ripple(f);
            println!("{f} Hz: ripple {:.1} %", r * 100.0);
            assert!(r < 0.03, "{f} Hz: background ripples by {:.1} %", r * 100.0);
        }
    }
}

#[cfg(test)]
mod real_file {
    /// `MVIZ_FILE=song.mp3 cargo test --release real_file -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn analyze_real_file() {
        let path = std::env::var("MVIZ_FILE").expect("set MVIZ_FILE");
        let t0 = std::time::Instant::now();
        let track = crate::audio::load(std::path::Path::new(&path)).unwrap();
        let rate = track.rate;
        let t1 = t0.elapsed();
        let tl = super::analyze(&track, &Default::default());
        let secs = track.length() as f32 / rate as f32;
        println!(
            "{secs:.0}s @ {rate} Hz: decode {t1:?}, analysis {:?}; {} bass hits ({:.1}/s)",
            t0.elapsed() - t1,
            tl.bass.len(),
            tl.bass.len() as f32 / secs,
        );
        let mut env: Vec<f32> = tl.bass_env.iter().map(|e| e.0.max(e.1)).collect();
        env.sort_by(f32::total_cmp);
        let q = |p: f32| env[((env.len() - 1) as f32 * p) as usize];
        println!("  bass envelope p10 {:.2} p50 {:.2} p90 {:.2} max {:.2}", q(0.1), q(0.5), q(0.9), q(1.0));
        // Flicker: frame-to-frame jitter of the glow at 60 fps (second difference,
        // relative to the mean level).
        let at = |t: f64| {
            let x = (t / tl.hop as f64).max(0.0);
            let i = (x as usize).min(tl.bass_env.len() - 2);
            let (a, b) = (tl.bass_env[i].0, tl.bass_env[i + 1].0);
            a + (b - a) * (x - i as f64) as f32
        };
        let n = (secs as f64 * 60.0) as usize - 2;
        let vals: Vec<f32> = (0..n).map(|k| at(k as f64 / 60.0)).collect();
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let jitter = vals.windows(3).map(|w| (w[0] - 2.0 * w[1] + w[2]).abs()).sum::<f32>() / (vals.len() - 2) as f32;
        println!("  glow jitter {:.2} % of mean level per frame", 100.0 * jitter / mean);
    }
}
