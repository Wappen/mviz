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

use super::live_bass::LiveBass;
use super::tracks::Tracks;
use super::{draw_background, Bass};
use crate::analysis::Timeline;
use crate::audio::Samples;
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
/// Bass amplitude (80 Hz low-pass, see `analysis::bass_envelope`) that
/// lights the background fully, per square root of the field's brightness
/// reference: where a typical song's loud bass sits (measured over a music
/// library: half the songs between 2.1 and 3.3). A bass-heavy mix lights
/// the background more, a thin one less.
const BASS_PER_FIELD: f32 = 2.8;

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
/// Longest time between two analyses (seconds). A frame covers its stretch of
/// audio with as many evenly spaced analyses as that needs, each handed to
/// the phosphor in turn (`gfx::PhosphorFrame::steps`), like a phosphor lit
/// the whole time the frame lasts: a sound moving fast leaves a smooth trail
/// instead of a row of spots or ridges (one per frame), whatever the frame
/// rate.
const SUB_STEP: f64 = 0.004;

/// Whether the glows of FFT size `res` are drawn at every analysis step
/// (`SUB_STEP`), or once per frame: only short windows (21 ms at 48 kHz)
/// change within a frame. The long ones are the large, overlapping low
/// glows, which would multiply the drawing for no visible difference (their
/// trails showed under 3 % ripple per frame).
fn stepped(res: usize) -> bool {
    RES[res].0 <= 2048
}
/// The most analyses for one frame (a stall or a seek is not drawn as motion).
const MAX_SUB_STEPS: usize = 25;


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

/// Averaged L², R² and L·conj(R) of a bin (per analysis): the 2×2
/// covariance of the two channels.
///
/// A loud hit (a snare) takes over every bin it covers, and while its
/// direction lingers in the average, the steady sounds sharing those bins
/// are drawn there too: pulled into the hit. So when the band's sound has
/// fallen well below its average (the hit is dying away), the level
/// balance forgets quickly (`forget`): the hit's direction goes with it.
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
    /// One analysis of the bin. `fading`: how far its band's sound has
    /// fallen below its average (`Stats::fading`), 0 = not at all.
    fn add(&mut self, p: &params::Params, dt: f32, fading: f32, l: Complex32, r: Complex32) {
        let fast = (1.0 - fading) / p.smooth.max(1e-3) + fading / p.forget.max(1e-3).min(p.smooth.max(1e-3));
        let f = 1.0 - (-dt * fast).exp();
        // Only the level balance forgets: polarity and timing need their
        // long average to tell a real phase from noise, whatever is playing
        // (forgetting them too lets bins flicker out of phase).
        let s = 1.0 - (-dt / p.slow.max(1e-3)).exp();
        self.learn(l, r, f, s);
    }

    /// Moves the averages towards one analysis `l`, `r` by `fast` (level
    /// balance) and `slow` (polarity and timing), each 0 .. 1.
    pub fn learn(&mut self, l: Complex32, r: Complex32, fast: f32, slow: f32) {
        let (l2, r2, x) = (l.norm_sqr(), r.norm_sqr(), l * r.conj());
        self.l2 += (l2 - self.l2) * fast;
        self.r2 += (r2 - self.r2) * fast;
        self.cross += (x - self.cross) * fast;
        self.slow_l2 += (l2 - self.slow_l2) * slow;
        self.slow_r2 += (r2 - self.slow_r2) * slow;
        self.slow_cross += (x - self.slow_cross) * slow;
    }

    /// How far a band's sound has fallen below its average, from its power
    /// now and its averaged power: 0 = not (much) below, 1 = `fade_hi` times
    /// below or more. Judged for the whole band: one bin's power dips by
    /// itself all the time (noise, beating), a band's much less.
    fn fading(p: &params::Params, now: f32, avg: f32) -> f32 {
        if avg <= 0.0 {
            0.0
        } else if now <= 0.0 {
            1.0
        } else {
            smoothstep(p.fade_lo, p.fade_hi.max(p.fade_lo + 1e-3), avg / now)
        }
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
    /// Per resolution: the partial tracks (placing the bins around them),
    /// and the range of bins they are looked for in.
    pub(super) tracks: Vec<Tracks>,
    track_range: Vec<(usize, usize)>,
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
    /// Live input: there is no audio after `frame` yet, so every window
    /// ends there instead of being centred on it, and the background's bass
    /// is measured as it arrives (`live_bass`) instead of analysed.
    pub causal: bool,
    live_bass: Option<LiveBass>,
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
            tracks: RES.iter().map(|_| Tracks::default()).collect(),
            track_range: (0..RES.len())
                .map(|res| {
                    let ks = bands.iter().flat_map(|b| &b.parts).filter(|p| p.res == res);
                    let k0 = ks.clone().map(|p| p.k0).min().unwrap_or(0);
                    (k0, ks.map(|p| p.k1).max().unwrap_or(0))
                })
                .collect(),
            bin_hz,
            last_frame: f64::NAN,
            reference: 0.0,
            provisional: 0.0,
            background: 0.0,
            causal: false,
            live_bass: None,
            bands,
        }
    }

    /// Analyses the audio centred on `frame` and updates every band and bin.
    fn update(&mut self, track: &dyn Samples, frame: f64, dt: f32, analysed_reference: Option<f32>) {
        let centre = frame as i64;
        for (fft, spec) in self.ffts.iter_mut().zip(&mut self.spec) {
            let start = if self.causal { centre - fft.n as i64 } else { centre - fft.n as i64 / 2 };
            for ch in 0..2 {
                // The song loops, so (once decoded) the audio before 0:00 is its end.
                let out = fft.run(|j| track.sample(start + j as i64, ch));
                spec[ch].copy_from_slice(out);
            }
        }
        let p = params::get();
        let fall = 1.0 - (-dt / p.release.max(1e-4)).exp();
        for (g, band) in self.live.iter_mut().zip(&self.bands) {
            let (mut energy, mut old) = (0.0, 0.0);
            for part in &band.parts {
                let [sl, sr] = &self.spec[part.res];
                let bins = &self.bins[part.res];
                for k in part.k0..part.k1.min(sl.len()) {
                    energy += part.weight * (sl[k].norm_sqr() + sr[k].norm_sqr());
                    old += part.weight * (bins[k].l2 + bins[k].r2);
                }
            }
            let fading = Stats::fading(&p, energy, old);
            for part in &band.parts {
                let [sl, sr] = &self.spec[part.res];
                let bins = &mut self.bins[part.res];
                for k in part.k0..part.k1.min(sl.len()) {
                    bins[k].add(&p, dt, fading, sl[k], sr[k]);
                }
            }
            g.energy = energy;
        }
        if p.tracks {
            for (res, tracks) in self.tracks.iter_mut().enumerate() {
                let [sl, sr] = &self.spec[res];
                let (k0, k1) = self.track_range[res];
                tracks.update(&p, dt, sl, sr, &self.bins[res], k0, k1);
            }
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
    pub fn draw(&mut self, background: &mut Batch, track: &dyn Samples, rate: u32, frame: f64, dt: f32, tl: Option<&Timeline>, w: f32, h: f32) -> PhosphorFrame {
        let (advance, bass, steps) = self.step(track, rate, frame, dt, tl, w, h);
        let mut live = Batch::default();
        let tonality = if params::get().bg_noise_white > 0.0 { self.bass_tonality() } else { 1.0 };
        draw_background(background, &mut live, bass.as_ref(), tonality, self.background, w, h);
        if advance.is_some() {
            super::field::draw(self, &mut live, w, h, |r| !stepped(r));
        }
        PhosphorFrame { live, steps, advance }
    }
}

impl Stereo {
    /// Analyses the audio up to `frame` (unless paused there: then it
    /// returns no advance and no steps), drawing the glows of the `stepped`
    /// FFT sizes at each analysis, and measures the background's bass.
    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, track: &dyn Samples, rate: u32, frame: f64, dt: f32, tl: Option<&Timeline>, w: f32, h: f32) -> (Option<f32>, Option<Bass>, Vec<Batch>) {
        let dt = dt.min(0.1);
        // Paused (same frame): nothing changes, and the phosphor holds still.
        let mut steps = Vec::new();
        let advance = (frame != self.last_frame).then(|| {
            // The audio since the last analysis, in evenly spaced steps; after
            // a seek or a stall only `frame` itself.
            let gap = frame - self.last_frame;
            let n = (gap / (SUB_STEP * rate as f64)).ceil();
            let n = if n >= 1.0 && n <= MAX_SUB_STEPS as f64 { n as usize } else { 1 };
            for i in 1..=n {
                let at = if n == 1 { frame } else { self.last_frame + gap * i as f64 / n as f64 };
                self.update(track, at, dt / n as f32, tl.map(|t| t.stereo_ref));
                let mut field = Batch::default();
                super::field::draw(self, &mut field, w, h, stepped);
                steps.push(field);
            }
            self.last_frame = frame;
            dt
        });
        // The bass background: from the analysis (once it is there), or for
        // live input measured as it arrives. Both share the field's loudness
        // reference, so the bass lights the background as brightly as it
        // weighs in the whole mix. Fade it in once it is there.
        let loud = BASS_PER_FIELD * self.reference.max(1e-9).sqrt();
        let bass = match tl {
            Some(tl) => Some(Bass::at(tl, (frame / rate as f64) as f32)),
            None if self.causal => {
                let b = self.live_bass.get_or_insert_with(|| LiveBass::new(rate)).update(track, frame as i64, dt, loud);
                Some(Bass { pc: b.pc, level: b.level })
            }
            None => None,
        }
        .map(|b| Bass { pc: b.pc, level: (b.level.0 / loud, b.level.1 / loud) });
        if bass.is_some() {
            self.background = (self.background + dt / 0.6).min(1.0);
        }
        (advance, bass, steps)
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
            super::super::field::draw(&st, &mut out, w, h, |_| true);
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

    /// The phosphor as on the GPU (`gfx`), on the CPU at half the window
    /// size (RGB), for each analysis step:
    /// glow' = blur(glow)·keep + max(previous live − live, 0).
    struct Sim {
        w: f32,
        cw: usize,
        ch: usize,
        glow: Vec<[f32; 3]>,
        prev: Vec<[f32; 3]>,
        /// Live light shown for the last frame (the average over its steps).
        live: Vec<[f32; 3]>,
    }

    impl Sim {
        fn new(w: f32, h: f32) -> Self {
            let (cw, ch) = ((w * 0.5) as usize, (h * 0.5) as usize);
            let z = vec![[0.0; 3]; cw * ch];
            Self { w, cw, ch, glow: z.clone(), prev: z.clone(), live: z }
        }

        /// Runs `f(x, y, pixel)` over `out`, in bands of rows on all cores.
        fn each(cw: usize, out: &mut [[f32; 3]], f: impl Fn(usize, usize, &mut [f32; 3]) + Sync) {
            let rows = (out.len() / cw).div_ceil(16);
            std::thread::scope(|sc| {
                for (j, chunk) in out.chunks_mut(rows * cw).enumerate() {
                    let f = &f;
                    sc.spawn(move || {
                        for (i, v) in chunk.iter_mut().enumerate() {
                            f(i % cw, j * rows + i / cw, v);
                        }
                    });
                }
            });
        }

        fn raster(&self, batch: &Batch) -> Vec<[f32; 3]> {
            let s = self.cw as f32 / self.w;
            let glows: Vec<_> = batch.glows().into_iter().map(|(cx, cy, rx, ry, c)| (cx * s, cy * s, (rx * s).max(0.5), (ry * s).max(0.5), c)).collect();
            let mut out = vec![[0.0f32; 3]; self.cw * self.ch];
            Self::each(self.cw, &mut out, |x, y, v| {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                for &(cx, cy, rx, ry, c) in &glows {
                    let (dx, dy) = ((px - cx) / rx, (py - cy) / ry);
                    let r2 = dx * dx + dy * dy;
                    if r2 < 1.0 {
                        let a = c[3] * (-r2 * 4.5).exp() * (1.0 - r2);
                        for k in 0..3 {
                            v[k] += c[k] * a;
                        }
                    }
                }
            });
            out
        }

        /// The glow's spread: a cross of bilinear reads `o` pixels out.
        fn blur(&mut self, o: f32) {
            let (cw, ch) = (self.cw, self.ch);
            let g = &self.glow;
            let at = |x: f32, y: f32| -> [f32; 3] {
                let (x, y) = ((x - 0.5).clamp(0.0, cw as f32 - 1.0), (y - 0.5).clamp(0.0, ch as f32 - 1.0));
                let (x0, y0) = (x.floor() as usize, y.floor() as usize);
                let (x1, y1) = ((x0 + 1).min(cw - 1), (y0 + 1).min(ch - 1));
                let (fx, fy) = (x - x0 as f32, y - y0 as f32);
                std::array::from_fn(|k| {
                    let a = g[y0 * cw + x0][k] * (1.0 - fx) + g[y0 * cw + x1][k] * fx;
                    let b = g[y1 * cw + x0][k] * (1.0 - fx) + g[y1 * cw + x1][k] * fx;
                    a * (1.0 - fy) + b * fy
                })
            };
            let mut next = vec![[0.0f32; 3]; cw * ch];
            Self::each(cw, &mut next, |x, y, v| {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let (c, l, r, u, d) = (at(px, py), at(px - o, py), at(px + o, py), at(px, py - o), at(px, py + o));
                *v = std::array::from_fn(|k| c[k] * 0.5 + (l[k] + r[k] + u[k] + d[k]) * 0.125);
            });
            self.glow = next;
        }

        /// One 60 fps frame of `st` at `frame`.
        fn frame(&mut self, st: &mut Stereo, track: &dyn Samples, rate: u32, frame: f64, h: f32) {
            let dt = 1.0f32 / 60.0;
            let pf = st.draw(&mut Batch::default(), track, rate, frame, dt, None, self.w, h);
            if pf.advance.is_none() {
                return;
            }
            // Lit throughout the frame, plus each step's dots.
            let slow = self.raster(&pf.live);
            let steps = if pf.steps.is_empty() { vec![Batch::default()] } else { pf.steps };
            let n = steps.len();
            let p = params::get();
            let keep = (-dt / n as f32 / p.fade).exp();
            let o = crate::gfx::blur_offset(p.fade_blur, dt / n as f32);
            let mut live = vec![[0.0f32; 3]; self.cw * self.ch];
            for step in &steps {
                let mut now = self.raster(step);
                for (v, s) in now.iter_mut().zip(&slow) {
                    for k in 0..3 {
                        v[k] += s[k];
                    }
                }
                self.blur(o);
                for (((g, p), v), l) in self.glow.iter_mut().zip(&self.prev).zip(&now).zip(&mut live) {
                    for k in 0..3 {
                        g[k] = g[k] * keep + (p[k] - v[k]).max(0.0);
                        l[k] += v[k] / n as f32;
                    }
                }
                self.prev = now;
            }
            self.live = live;
        }

        fn shown(&self) -> Vec<[f32; 3]> {
            self.live.iter().zip(&self.glow).map(|(l, g)| std::array::from_fn(|k| l[k] + g[k])).collect()
        }

        fn save(&self, path: &str) {
            let mut ppm = format!("P6 {} {} 255\n", self.cw, self.ch).into_bytes();
            for c in self.shown() {
                ppm.extend(c.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8));
            }
            std::fs::write(path, ppm).unwrap();
        }
    }

    /// Renders the field for a song into PPM images (phosphor simulated on
    /// the CPU), every `MVIZ_EVERY` of `MVIZ_FRAMES` frames from `MVIZ_START`
    /// s, and prints the mean light (sum of channels) of the live light and
    /// the glow after the first second.
    /// `MVIZ_FILE=song.mp3 MVIZ_OUT=dir cargo test --release dump_field -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_field() {
        let path = std::env::var("MVIZ_FILE").expect("set MVIZ_FILE");
        let out = std::env::var("MVIZ_OUT").expect("set MVIZ_OUT");
        let track = crate::audio::load(std::path::Path::new(&path)).unwrap();
        let rate = track.rate;
        let var = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (start, every, frames) = (var("MVIZ_START", 40.0), var("MVIZ_EVERY", 60.0) as usize, var("MVIZ_FRAMES", 300.0) as usize);
        let (w, h) = (1600.0, 1000.0);
        let mut st = Stereo::new(rate);
        let mut sim = Sim::new(w, h);
        let (mut live, mut glow) = (0.0f64, 0.0f64);
        let mean = |v: &[[f32; 3]]| v.iter().map(|c| (c[0] + c[1] + c[2]) as f64).sum::<f64>() / v.len() as f64;
        for i in 0..frames {
            sim.frame(&mut st, &track, rate, (start + i as f64 / 60.0) * rate as f64, h);
            if i >= 60 {
                live += mean(&sim.live);
                glow += mean(&sim.glow);
            }
            if i % every == every - 1 {
                sim.save(&format!("{out}/field_{i}.ppm"));
            }
        }
        let k = frames.saturating_sub(60).max(1) as f64;
        println!("from {start} s: mean live {:.5}, glow {:.5}", live / k, glow / k);
    }

    /// A chord at 0.7 (right) with a loud centred snare every half second:
    /// where the light in the chord's rows is drawn: at the chord, at the
    /// snare, and dragged in between, before the first hit, during
    /// each hit, and in its tail (when the chord should be back in place).
    /// `cargo test --release snare_over_chord -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn snare_over_chord() {
        use std::f32::consts::{FRAC_PI_4, TAU};
        let rate = 48000u32;
        let (w, h) = (1600.0f32, 1000.0f32);
        let secs = 4.0;
        let n = (rate as f32 * secs) as usize;
        let th = (0.7 + 1.0) * FRAC_PI_4;
        let mut seed = 1u32;
        let samples: Vec<f32> = (0..n)
            .flat_map(|i| {
                let t = i as f32 / rate as f32;
                let chord: f32 = [220.0f32, 277.2, 329.6].iter().flat_map(|&f0| (1..=8).map(move |k| (TAU * f0 * k as f32 * t).sin() * 0.03 / k as f32)).sum();
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
                let since = (t - 1.0).rem_euclid(0.5);
                let snare = if t >= 1.0 { noise * 0.25 * (-since / 0.05).exp() } else { 0.0 };
                [chord * th.cos() + snare, chord * th.sin() + snare]
            })
            .collect();
        let track = crate::audio::Track::from_interleaved(&samples, rate);
        let scale = (h * 0.5).max(w * 0.5 / 1.5);
        // Only the rows of the chord's partials (the chord's own light).
        let p = params::get();
        let rows: Vec<f32> = [220.0f32, 277.2, 329.6].iter().flat_map(|&f0| (1..=8).map(move |k| f0 * k as f32)).map(|f| y_of(&p, f) * h).collect();
        let mut st = Stereo::new(rate);
        // [before hits, hit (0–60 ms), tail (100–400 ms)] × [chord, snare, between]
        let mut acc = [[0.0f64; 3]; 3];
        for i in 0..(secs * 60.0) as usize {
            let t = i as f32 / 60.0;
            let pf = st.draw(&mut Batch::default(), &*track, rate, t as f64 * rate as f64, 1.0 / 60.0, None, w, h);
            if t < 0.5 {
                continue;
            }
            let since = (t - 1.0).rem_euclid(0.5);
            let when = if t < 1.0 { 0 } else if since < 0.06 { 1 } else if (0.1..0.4).contains(&since) { 2 } else { continue };
            for step in &pf.steps {
                for (x, y, _, _, c) in step.glows() {
                    if !rows.iter().any(|r| (r - y).abs() < 2.0) {
                        continue;
                    }
                    let pos = (x - w * 0.5) / scale;
                    let light = (c[3] * (c[0] + c[1] + c[2])) as f64;
                    let zone = if (pos - 0.7).abs() < 0.15 { 0 } else if pos.abs() < 0.15 { 1 } else if pos > 0.15 && pos < 0.55 { 2 } else { continue };
                    acc[when][zone] += light;
                }
            }
        }
        for (name, a) in ["before", "hit", "tail"].iter().zip(acc) {
            let sum: f64 = a.iter().sum();
            println!("{name:>10}: chord {:5.1}%  snare {:5.1}%  between {:5.1}%", 100.0 * a[0] / sum, 100.0 * a[1] / sum, 100.0 * a[2] / sum);
        }
    }

    /// Lightness ripple along the trail of a tone panned across the field
    /// at constant speed: the strongest ripple's period and size (peak to
    /// peak, against the smoothed lightness), next to how far the tone moves
    /// per analysis step and per frame. Was up to 20 % with a period of one
    /// frame's travel when the phosphor took light once per frame.
    /// `cargo test --release trail_ripple -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn trail_ripple() {
        use std::f32::consts::{FRAC_PI_2, TAU};
        let rate = 48000u32;
        let (w, h) = (1600.0f32, 1000.0f32);
        println!("{:>6} {:>6} | {:>7} {:>7} | {:>6} {:>6}", "Hz", "sweep", "d_step", "d_frame", "period", "ripple");
        for &f in &[1000.0f32, 2000.0, 4000.0, 8000.0] {
            for &sweep in &[2.0f32, 1.0, 0.5] {
                // Pan from left to right over `sweep` seconds, after a still second.
                let n = (rate as f32 * (1.0 + sweep)) as usize;
                let samples: Vec<f32> = (0..n)
                    .flat_map(|i| {
                        let t = i as f32 / rate as f32;
                        let u = ((t - 1.0) / sweep).clamp(0.0, 1.0);
                        let th = 0.05 + u * (FRAC_PI_2 - 0.1);
                        let v = (TAU * f * t).sin() * 0.3;
                        [v * th.cos(), v * th.sin()]
                    })
                    .collect();
                let track = crate::audio::Track::from_interleaved(&samples, rate);
                let frames = ((1.0 + sweep * 0.66) * 60.0) as usize;
                let mut st = Stereo::new(rate);
                let mut sim = Sim::new(w, h);
                // The dot: the brightest point of the live light, each frame.
                let mut dots = Vec::new();
                for i in 0..frames {
                    sim.frame(&mut st, &*track, rate, i as f64 / 60.0 * rate as f64, h);
                    let lum: Vec<f32> = sim.live.iter().map(|c| c[0] + c[1] + c[2]).collect();
                    let m = (0..lum.len()).max_by(|a, b| lum[*a].total_cmp(&lum[*b])).unwrap();
                    dots.push((m % sim.cw, m / sim.cw));
                }
                let k = dots.len();
                let d_frame = (dots[k - 1].0 as f32 - dots[k - 11].0 as f32) / 10.0;
                let d_step = d_frame / ((rate as f64 / 60.0) / (SUB_STEP * rate as f64)).ceil() as f32;
                let (dot, row) = dots[k - 1];
                let lum: Vec<f32> = sim.shown().iter().map(|c| c[0] + c[1] + c[2]).collect();
                // Past the dot's own light (where the live light is under 2 % of
                // its peak), up to 12 frames back.
                let peak = sim.live[row * sim.cw + dot].iter().sum::<f32>();
                let mut b = dot;
                while b > 0 && sim.live[row * sim.cw + b].iter().sum::<f32>() > 0.02 * peak {
                    b -= 1;
                }
                let a = (b as f32 - 12.0 * d_frame).max(0.0) as usize;
                let (period, size) = ripple(&lum[row * sim.cw..(row + 1) * sim.cw], a, b, d_frame);
                println!("{f:>6} {sweep:>5}s | {d_step:>6.1}p {d_frame:>6.1}p | {period:>5.1}p {:>5.1}%", size * 100.0);
            }
        }
    }

    /// The strongest periodic ripple of `line[a..b]` relative to its smoothed
    /// self (over two frames' travel): (period in px, peak to peak).
    fn ripple(line: &[f32], a: usize, b: usize, d_frame: f32) -> (f32, f32) {
        use std::f32::consts::TAU;
        if b <= a + 8 {
            return (0.0, 0.0);
        }
        let prof = &line[a..b];
        let sig = (2.0 * d_frame).max(3.0);
        let k = (3.0 * sig) as i32;
        let res: Vec<f32> = (0..prof.len() as i32)
            .map(|i| {
                let (mut s, mut wsum) = (0.0, 0.0);
                for j in -k..=k {
                    let q = (i + j).clamp(0, prof.len() as i32 - 1) as usize;
                    let wj = (-(j * j) as f32 / (2.0 * sig * sig)).exp();
                    s += prof[q] * wj;
                    wsum += wj;
                }
                let smooth = s / wsum;
                if smooth > 1e-6 { prof[i as usize] / smooth - 1.0 } else { 0.0 }
            })
            .collect();
        let (mut best_p, mut best_a) = (0.0, 0.0);
        let mut period = 2.0f32;
        while period < sig * 2.0 {
            let (mut re, mut im) = (0.0, 0.0);
            for (x, r) in res.iter().enumerate() {
                let ph = TAU * x as f32 / period;
                re += r * ph.cos();
                im += r * ph.sin();
            }
            let amp = 2.0 * (re * re + im * im).sqrt() / res.len() as f32;
            if amp > best_a {
                (best_p, best_a) = (period, amp);
            }
            period *= 1.02;
        }
        (best_p, 2.0 * best_a)
    }

    #[test]
    fn timing_fades_out_between_500_and_700_hz() {
        assert_eq!(timing_weight(300.0), 1.0);
        assert_eq!(timing_weight(800.0), 0.0);
        let w = timing_weight(600.0);
        assert!(w > 0.3 && w < 0.7);
    }

    /// Compares the live background (measured as the audio arrives) with
    /// the analysed one over a real song:
    /// `MVIZ_FILE=song.mp3 cargo test --release live_matches_analysis -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_matches_analysis() {
        let path = std::env::var("MVIZ_FILE").expect("set MVIZ_FILE");
        let track = crate::audio::load(std::path::Path::new(&path)).unwrap();
        let tl = crate::analysis::analyze(&track, &Default::default());
        let rate = track.rate;
        let secs = track.length() as f64 / rate as f64;
        let (mut song, mut live) = (Stereo::new(rate), Stereo::new(rate));
        live.causal = true;
        let (mut same, mut notes, mut level_err, mut n) = (0, 0, 0.0f32, 0);
        let mut f = 1;
        while (f as f64) / 60.0 < secs {
            let t = f as f64 / 60.0;
            let frame = (t * rate as f64).floor();
            let a = song.step(&track, rate, frame, 1.0 / 60.0, Some(&tl), 1920.0, 1080.0).1.unwrap();
            let b = live.step(&track, rate, frame, 1.0 / 60.0, None, 1920.0, 1080.0).1.unwrap();
            if t > 20.0 {
                if let (Some(x), Some(y)) = (b.pc, a.pc) {
                    notes += 1;
                    let d = (x - y).rem_euclid(12.0);
                    if d.min(12.0 - d) < 0.5 {
                        same += 1;
                    }
                }
                // As the background sees it (capped at 1.2).
                let c = |v: f32| v.clamp(0.0, 1.2);
                level_err += (c(b.level.0) - c(a.level.0)).abs() + (c(b.level.1) - c(a.level.1)).abs();
                n += 2;
            }
            if f % 600 == 0 {
                println!("{t:6.1}s  live pc {:?} level ({:.2}, {:.2}) ref {:.5}  song pc {:?} level ({:.2}, {:.2}) ref {:.5}", b.pc.map(|p| p.round()), b.level.0, b.level.1, live.reference, a.pc.map(|p| p.round()), a.level.0, a.level.1, song.reference);
            }
            f += 1;
        }
        println!("same note {same} of {notes} frames ({:.0}%), mean level difference {:.3}", 100.0 * same as f32 / notes.max(1) as f32, level_err / n.max(1) as f32);
    }
}

