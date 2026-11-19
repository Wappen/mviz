//! Live stereo analysis for the sound field (drawn by `field.rs`).
//!
//! Seen from the listener's seat: x = where a sound sits between the left
//! and right speakers, y = pitch (log scale, every octave the same height).
//! Lower sounds are larger and softer; deep bass spreads into wide, flat
//! pools (a carpet along the bottom), fading out gradually with pitch; see
//! `carpet`. The background glow follows the bass, each side its own channel.
//!
//! Every frame the audio around the playback position is analysed into
//! pitch bands (their brightness) and FFT bins (each with its smoothed
//! left/right statistics, which place it in the field).
//!
//! The glows are the live light and appear instantly; all fading happens in
//! a phosphor texture on the GPU (see `gfx::FADE`).

use super::draw_background;
use crate::analysis::Timeline;
use crate::audio::Track;
use crate::dsp::Fft;
use crate::gfx::{Batch, PhosphorFrame};
use crate::params;
use rustfft::num_complex::Complex32;

pub(super) const F_LO: f32 = 30.0;
/// Highest analysed frequency: the axis continues above `f_hi` into the top
/// margin, so the field thins out there naturally instead of ending in a
/// hard edge.
const BAND_HI: f32 = 20000.0;
const PER_OCT: f32 = 30.0;

/// How much a pitch is "carpet": 1 at the lowest pitch, shrinking gradually
/// over the whole range and reaching 0 only at the top (with power 3:
/// 60 Hz 0.70, 250 Hz 0.29, 1 kHz 0.09, 4 kHz 0.01).
fn carpet(q: f32, power: f32) -> f32 {
    (1.0 - q).max(0.0).powf(power)
}

/// FFT sizes, longest first, with the frequency where the next (shorter) one
/// takes over: long windows resolve close bass notes, short ones keep treble
/// sharp in time. Around each hand-over the two are crossfaded over
/// `RES_FADE_OCT` octaves either side, so time resolution changes smoothly.
pub(super) const RES: [(usize, f32); 3] = [(8192, 200.0), (4096, 2000.0), (1024, f32::INFINITY)];
const RES_FADE_OCT: f32 = 1.0 / 3.0;


pub fn band_edges() -> Vec<(f32, f32)> {
    let n = ((BAND_HI / F_LO).log2() * PER_OCT).ceil() as usize;
    let edge = |i: usize| F_LO * 2f32.powf(i as f32 / PER_OCT);
    (0..n).map(|i| (edge(i), edge(i + 1))).collect()
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much of the timing (phase) difference to trust at `f`: 1 below
/// `delay_fade_lo`, 0 above `delay_fade_hi`, where the phase difference no
/// longer maps to a unique delay.
pub(super) fn timing_weight(f: f32) -> f32 {
    let p = params::get();
    1.0 - smoothstep(p.delay_fade_lo, p.delay_fade_hi.max(p.delay_fade_lo + 1.0), f)
}

/// Which FFT sizes cover frequency `f`, and with what weight (summing to 1).
fn res_weights(f: f32) -> Vec<(usize, f32)> {
    for i in 0..RES.len() - 1 {
        let edge = RES[i].1;
        let x = (f / edge).log2() / RES_FADE_OCT; // -1 .. 1 inside the crossfade
        if x < 1.0 {
            let next = smoothstep(-1.0, 1.0, x);
            return [(i, 1.0 - next), (i + 1, next)].into_iter().filter(|p| p.1 > 0.0).collect();
        }
    }
    vec![(RES.len() - 1, 1.0)]
}

/// Screen y (share of height) of a frequency: log scale, lowest at the bottom.
/// Above `f_hi` the axis simply continues into the top margin (instead of
/// piling everything higher onto one row), up to the window edge.
pub(super) fn y_of(p: &params::Params, f: f32) -> f32 {
    let q = ((f / F_LO).log2() / (p.f_hi / F_LO).log2()).max(0.0);
    (1.0 - p.bottom_margin - q * (1.0 - p.top_margin - p.bottom_margin)).max(0.0)
}

/// Spot radii for a frequency: deeper = bigger and softer, and more carpet
/// (wider and flatter).
pub(super) fn spot_size(p: &params::Params, f: f32, w: f32, h: f32) -> (f32, f32) {
    let q = ((f / F_LO).log2() / (p.f_hi / F_LO).log2()).clamp(0.0, 1.0);
    let c = carpet(q, p.carpet_power);
    let r = (h * p.size_base * (f.max(F_LO) / 100.0).powf(-p.size_exp)).max(h * p.size_floor);
    (r + (w * p.carpet_width - r).max(0.0) * c, r * (1.0 - (1.0 - p.carpet_flat) * c))
}

/// Part of a band measured with one FFT size.
pub(super) struct Part {
    pub res: usize,
    pub k0: usize,
    pub k1: usize,
    pub weight: f32,
}

pub(super) struct Band {
    pub fc: f32,
    pub parts: Vec<Part>,
}

/// A band's state.
#[derive(Clone, Default)]
pub(super) struct Live {
    /// Level right now (0..1): instant rise, `RELEASE` fall.
    pub bright: f32,
    /// Weighted energy right now (L² + R²), to share the light among the bins.
    pub energy: f32,
}

/// Smoothed L², R² and L·conj(R) of a bin: the 2×2 covariance of the two
/// channels.
#[derive(Clone, Copy, Default)]
pub(super) struct Stats {
    pub l2: f32,
    pub r2: f32,
    pub cross: Complex32,
    /// The same over `SLOW` (for timing, whose low-frequency phase needs a
    /// longer average to tell a real delay from noise).
    pub slow_l2: f32,
    pub slow_r2: f32,
    pub slow_cross: Complex32,
}

impl Stats {
    fn add(&mut self, keep: f32, keep_slow: f32, l: Complex32, r: Complex32) {
        let (l2, r2, x) = (l.norm_sqr(), r.norm_sqr(), l * r.conj());
        self.l2 = self.l2 * keep + l2;
        self.r2 = self.r2 * keep + r2;
        self.cross = self.cross * keep + x;
        self.slow_l2 = self.slow_l2 * keep_slow + l2;
        self.slow_r2 = self.slow_r2 * keep_slow + r2;
        self.slow_cross = self.slow_cross * keep_slow + x;
    }

    /// A covariance that has been the same for a long time (tests).
    #[cfg(test)]
    pub fn steady(l2: f32, r2: f32, cross: Complex32) -> Self {
        Self { l2, r2, cross, slow_l2: l2, slow_r2: r2, slow_cross: cross }
    }
}

pub struct Stereo {
    ffts: Vec<Fft>,
    pub(super) bands: Vec<Band>,
    pub(super) spec: Vec<[Vec<Complex32>; 2]>, // per resolution: (L, R)
    pub(super) live: Vec<Live>,
    /// Per resolution, per FFT bin: smoothed stats (placement).
    pub(super) bins: Vec<Vec<Stats>>,
    /// Hz per bin, per resolution.
    pub(super) bin_hz: Vec<f32>,
    last_frame: f64,
    /// Brightness reference in use (0 = not yet set): provisional from what
    /// has played so far until the full analysis arrives, then blended to its
    /// value.
    reference: f32,
    provisional: f32,
    /// How much of the bass background is shown (fades in once analysed).
    background: f32,
}

impl Stereo {
    pub fn new(rate: u32) -> Self {
        let ffts: Vec<Fft> = RES.iter().map(|&(n, _)| Fft::new(n)).collect();
        let bin_hz: Vec<f32> = RES.iter().map(|&(n, _)| rate as f32 / n as f32).collect();
        let mut bands: Vec<Band> = Vec::new();
        for (lo, hi) in band_edges() {
            let fc = (lo * hi).sqrt();
            let mut parts: Vec<Part> = res_weights(fc)
                .into_iter()
                .filter_map(|(res, weight)| {
                    let hz = bin_hz[res];
                    let (k0, k1) = ((lo / hz).round() as usize, (hi / hz).round() as usize);
                    // Narrower than one FFT bin: left to a neighbouring band.
                    (k1 > k0).then_some(Part { res, k0, k1, weight })
                })
                .collect();
            if parts.is_empty() {
                continue;
            }
            let total: f32 = parts.iter().map(|p| p.weight).sum();
            parts.iter_mut().for_each(|p| p.weight /= total);
            // Centre of the frequencies actually covered.
            let p = &parts[0];
            let hz = bin_hz[p.res];
            let fc = (((p.k0 as f32 - 0.5) * hz).max(1.0) * (p.k1 as f32 - 0.5) * hz).sqrt();
            bands.push(Band { fc, parts });
        }
        let nb = bands.len();
        Self {
            spec: RES.iter().map(|&(n, _)| [vec![Complex32::default(); n / 2 + 1], vec![Complex32::default(); n / 2 + 1]]).collect(),
            ffts,
            live: vec![Live::default(); nb],
            bins: RES.iter().map(|&(n, _)| vec![Stats::default(); n / 2 + 1]).collect(),
            bin_hz,
            last_frame: f64::NAN,
            reference: 0.0,
            provisional: 0.0,
            background: 0.0,
            bands,
        }
    }

    /// Analyses the audio centred on `frame` and updates every band and bin.
    fn update(&mut self, track: &Track, frame: f64, dt: f32, analysed_reference: Option<f32>) {
        let centre = frame as i64;
        for (fft, spec) in self.ffts.iter_mut().zip(&mut self.spec) {
            let start = centre - fft.n as i64 / 2;
            for ch in 0..2 {
                // The song loops, so (once decoded) the audio before 0:00 is its end.
                let out = fft.run(|j| track.sample(start + j as i64, ch));
                spec[ch].copy_from_slice(out);
            }
        }
        let p = params::get();
        let keep = (-dt / p.smooth.max(1e-3)).exp();
        let keep_slow = (-dt / p.slow.max(1e-3)).exp();
        let fall = 1.0 - (-dt / p.release.max(1e-4)).exp();
        for (g, band) in self.live.iter_mut().zip(&self.bands) {
            let mut energy = 0.0;
            for part in &band.parts {
                let [sl, sr] = &self.spec[part.res];
                let bins = &mut self.bins[part.res];
                for k in part.k0..part.k1.min(sl.len()) {
                    energy += part.weight * (sl[k].norm_sqr() + sr[k].norm_sqr());
                    bins[k].add(keep, keep_slow, sl[k], sr[k]);
                }
            }
            g.energy = energy;
        }

        // Brightness reference. The full analysis measures it over the whole
        // song (the loud end of all bands' levels); until that has finished,
        // estimate it from what has played: the loud end of this frame's
        // levels, held at its peak and slowly released.
        let mut levels: Vec<f32> = self
            .live
            .iter()
            .zip(&self.bands)
            .filter(|(g, _)| g.energy > 1e-10)
            .map(|(g, b)| g.energy * b.fc / 1000.0)
            .collect();
        if !levels.is_empty() {
            let loud = crate::dsp::percentile(&mut levels, 0.95);
            self.provisional = (self.provisional * (-dt / 20.0).exp()).max(loud).max(1e-5);
        }
        let goal = analysed_reference.unwrap_or(self.provisional);
        if self.reference <= 0.0 {
            self.reference = goal;
        } else if goal > 0.0 {
            // Glide (in dB) so switching to the analysed value is not a jump.
            let a = 1.0 - (-dt / 0.8).exp();
            self.reference = (self.reference.ln() + (goal.ln() - self.reference.ln()) * a).exp();
        }
        let reference = self.reference.max(1e-9);
        for (g, band) in self.live.iter_mut().zip(&self.bands) {
            let now = brightness(&p, g.energy, band.fc, reference);
            let before = g.bright;
            g.bright = if now > before { now } else { before + (now - before) * fall };
        }
    }

    /// Background into `background`; the live light and how far to advance
    /// the phosphor (none while paused) into the returned frame.
    #[allow(clippy::too_many_arguments)]
    /// `tl` is the full analysis once it has finished (`None` before).
    pub fn draw(&mut self, background: &mut Batch, track: &Track, rate: u32, frame: f64, dt: f32, tl: Option<&Timeline>, w: f32, h: f32) -> PhosphorFrame {
        // Paused (same frame): nothing changes, and the phosphor holds still.
        let advance = (frame != self.last_frame).then(|| {
            let dt = dt.min(0.1);
            self.update(track, frame, dt, tl.map(|t| t.stereo_ref));
            self.last_frame = frame;
            dt
        });
        // The bass background needs the analysis; fade it in once it is there.
        if tl.is_some() {
            self.background = (self.background + dt.min(0.1) / 0.6).min(1.0);
        }
        let mut live = Batch::default();
        let tonality = if params::get().bg_noise_white > 0.0 { self.bass_tonality() } else { 1.0 };
        draw_background(background, &mut live, tl, (frame / rate as f64) as f32, tonality, self.background, w, h);
        super::field::draw(self, &mut live, w, h);
        PhosphorFrame { live, advance }
    }
}

impl Stereo {
    /// How pitched the deep bass (the background's range, up to 100 Hz) is
    /// right now: the bands' tonality weighted by their energy.
    fn bass_tonality(&self) -> f32 {
        let (mut sum, mut weight) = (0.0, 0.0);
        for (g, band) in self.live.iter().zip(&self.bands).filter(|(_, b)| b.fc <= 100.0) {
            sum += super::field::tonality(self, band) * g.energy;
            weight += g.energy;
        }
        if weight > 1e-12 { sum / weight } else { 1.0 }
    }
}

/// 0..1 brightness of a band's energy, with a treble tilt (`tilt_db` per
/// octave) so treble is as visible as the (naturally louder) low end.
fn brightness(p: &params::Params, e: f32, fc: f32, reference: f32) -> f32 {
    if e < 1e-12 {
        return 0.0;
    }
    let tilt = p.tilt_db * (fc / 1000.0).log2();
    let weight = if p.loudness > 0.0 { tilt + (crate::dsp::perceived_gain_db(fc, p.phon, p.felt_bass) - tilt) * p.loudness } else { tilt };
    let db = 10.0 * (e / reference).log10() + weight;
    ((db + p.db_range) / p.db_range.max(1.0)).clamp(0.0, 1.0).powf(p.bright_curve)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolutions_crossfade_smoothly() {
        let check = |f: f32| {
            let w = res_weights(f);
            let total: f32 = w.iter().map(|p| p.1).sum();
            assert!((total - 1.0).abs() < 1e-5, "weights at {f} sum to {total}");
            w
        };
        assert_eq!(check(100.0), vec![(0, 1.0)]);
        assert_eq!(check(500.0), vec![(1, 1.0)]);
        assert_eq!(check(8000.0), vec![(2, 1.0)]);
        let mid = check(200.0);
        assert!(mid.len() == 2 && (mid[0].1 - 0.5).abs() < 1e-5, "{mid:?}");
        // Monotonic hand-over through the 2 kHz crossfade.
        let mut prev = 0.0;
        for i in 0..=20 {
            let f = 2000.0 * 2f32.powf((i as f32 / 10.0 - 1.0) * RES_FADE_OCT);
            let share: f32 = check(f).iter().filter(|p| p.0 == 2).map(|p| p.1).sum();
            assert!(share >= prev - 1e-6, "short-FFT share fell at {f}");
            prev = share;
        }
    }

    /// `MVIZ_FILE=song.mp3 cargo test --release frame_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn frame_cost() {
        let path = std::env::var("MVIZ_FILE").expect("set MVIZ_FILE");
        let track = crate::audio::load(std::path::Path::new(&path)).unwrap();
        let rate = track.rate;
        let tl = crate::analysis::analyze(&track, &Default::default());
        let mut st = Stereo::new(rate);
        let (w, h) = (3840.0, 2160.0);
        let (mut analysis, mut grid) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        let mut glows = 0;
        let frames = 600;
        for i in 0..frames {
            let frame = (30.0 + i as f64 / 60.0) * rate as f64;
            let t0 = std::time::Instant::now();
            st.update(&track, frame, 1.0 / 60.0, Some(tl.stereo_ref));
            let t1 = std::time::Instant::now();
            let mut out = Batch::default();
            super::super::field::draw(&st, &mut out, w, h);
            glows += out.glows().len();
            grid += t1.elapsed();
            analysis += t1 - t0;
        }
        println!("per frame: analysis {:?}, glows {:?} ({} glows)", analysis / frames, grid / frames, glows / frames as usize);
    }

    /// `MVIZ_FILE=song.mp3 cargo test --release position_jumps -- --ignored --nocapture`
    /// Share of audible bins whose position jumps far from one frame to the next.
    #[test]
    #[ignore]
    fn position_jumps() {
        let path = std::env::var("MVIZ_FILE").expect("set MVIZ_FILE");
        let track = crate::audio::load(std::path::Path::new(&path)).unwrap();
        let rate = track.rate;
        let tl = crate::analysis::analyze(&track, &Default::default());
        let mut st = Stereo::new(rate);
        let mut prev: Vec<Vec<f32>> = RES.iter().map(|&(n, _)| vec![f32::NAN; n / 2 + 1]).collect();
        let (mut lit, mut far, mut beyond) = (0usize, 0usize, 0usize);
        let (mut lit_lo, mut far_lo) = (0usize, 0usize);
        for i in 0..1200 {
            st.update(&track, (30.0 + i as f64 / 60.0) * rate as f64, 1.0 / 60.0, Some(tl.stereo_ref));
            for (g, band) in st.live.iter().zip(&st.bands) {
                if g.bright < 0.05 {
                    continue;
                }
                for part in &band.parts {
                    for k in part.k0.max(1)..part.k1 {
                        let f = k as f32 * st.bin_hz[part.res];
                        let p = super::super::field::position(&params::get(), &st.bins[part.res][k], f);
                        let q = prev[part.res][k];
                        if i > 30 && !q.is_nan() {
                            lit += 1;
                            // Near the far edges both sides are the same place (light is shared).
                            let mut d = (p - q).abs();
                            if p.abs() > 1.7 && q.abs() > 1.7 {
                                d = d.min((p + q).abs());
                            }
                            far += (d > 0.5) as usize;
                            if f < 700.0 {
                                lit_lo += 1;
                                far_lo += (d > 0.5) as usize;
                            }
                            beyond += (p.abs() > 1.05) as usize;
                        }
                        prev[part.res][k] = p;
                    }
                }
            }
        }
        println!("below 700 Hz: {lit_lo} bin-frames, jumps {:.3}%; above: jumps {:.3}%", 100.0 * far_lo as f64 / lit_lo as f64, 100.0 * (far - far_lo) as f64 / (lit - lit_lo) as f64);
        println!("bin-frames {lit}: jumps > 0.5 {:.3}%, beyond the speakers {:.2}%", 100.0 * far as f64 / lit as f64, 100.0 * beyond as f64 / lit as f64);
    }

    /// Renders the field for a song into PPM images, with the phosphor
    /// simulated on the CPU (quarter resolution).
    /// `MVIZ_FILE=song.mp3 MVIZ_OUT=dir cargo test --release dump_field -- --ignored`
    #[test]
    #[ignore]
    fn dump_field() {
        let path = std::env::var("MVIZ_FILE").expect("set MVIZ_FILE");
        let out = std::env::var("MVIZ_OUT").expect("set MVIZ_OUT");
        let track = crate::audio::load(std::path::Path::new(&path)).unwrap();
        let rate = track.rate;
        let tl = crate::analysis::analyze(&track, &Default::default());
        let mut st = Stereo::new(rate);
        let (w, h) = (2472.0, 779.0);
        let (cw, ch) = (618usize, 195usize); // canvas: a quarter of the window
        let start: f64 = std::env::var("MVIZ_START").ok().and_then(|v| v.parse().ok()).unwrap_or(40.0);
        let every: usize = std::env::var("MVIZ_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
        let frames: usize = std::env::var("MVIZ_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
        // CPU copy of the phosphor: glow = glow·keep + max(previous − now, 0); shown = now + glow.
        let keep = (-1.0f32 / 60.0 / params::get().fade).exp();
        let mut prev = vec![0.0f32; cw * ch * 3];
        let mut glow = vec![0.0f32; cw * ch * 3];
        for i in 0..frames {
            st.update(&track, (start + i as f64 / 60.0) * rate as f64, 1.0 / 60.0, Some(tl.stereo_ref));
            let mut batch = Batch::default();
            super::super::field::draw(&st, &mut batch, w, h);
            let mut now = vec![0.0f32; cw * ch * 3];
            let s = cw as f32 / w;
            for (cx, cy, rx, ry, c) in batch.glows() {
                let (cx, cy, rx, ry) = (cx * s, cy * s, (rx * s).max(0.5), (ry * s).max(0.5));
                let (x0, x1) = (((cx - rx).floor().max(0.0)) as usize, ((cx + rx).ceil() as usize).min(cw));
                let (y0, y1) = (((cy - ry).floor().max(0.0)) as usize, ((cy + ry).ceil() as usize).min(ch));
                for y in y0..y1 {
                    for x in x0..x1 {
                        let (dx, dy) = ((x as f32 + 0.5 - cx) / rx, (y as f32 + 0.5 - cy) / ry);
                        let r2 = dx * dx + dy * dy;
                        if r2 < 1.0 {
                            let a = c[3] * (-r2 * 4.5).exp() * (1.0 - r2);
                            for k in 0..3 {
                                now[(y * cw + x) * 3 + k] += c[k] * a;
                            }
                        }
                    }
                }
            }
            for j in 0..now.len() {
                glow[j] = glow[j] * keep + (prev[j] - now[j]).max(0.0);
            }
            prev = now;
            if i % every == every - 1 {
                let mut ppm = format!("P6 {cw} {ch} 255\n").into_bytes();
                for j in 0..prev.len() {
                    ppm.push(((prev[j] + glow[j]).clamp(0.0, 1.0) * 255.0) as u8);
                }
                std::fs::write(format!("{out}/field_{i}.ppm"), ppm).unwrap();
            }
        }
    }

    #[test]
    fn timing_fades_out_between_500_and_700_hz() {
        assert_eq!(timing_weight(300.0), 1.0);
        assert_eq!(timing_weight(800.0), 0.0);
        let w = timing_weight(600.0);
        assert!(w > 0.3 && w < 0.7);
    }
}
