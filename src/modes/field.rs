//! The sound field: every FFT bin drawn as its own soft glow at its own
//! stereo position, on the GPU.
//!
//! Direction: every bin is treated as one sound at one place, its dominant
//! direction (the principal eigenvector of its smoothed left/right
//! covariance). Its position comes from the level ratio and polarity of the
//! two channels and, at low frequencies, from which side it arrives first
//! (see `timing_weight`). Content that is out of phase between the channels
//! lies beyond the speakers, up to fully out of phase at twice the speaker
//! distance (|position| = 2), which is where it is heard.
//!
//! Light: each bin gets its share of its band's light (shared by energy, so
//! a band's total light depends only on its loudness), drawn as a glow sized
//! for its pitch (`spot_size`: wide and flat for deep bass: the carpet), never smaller than
//! `MIN_SPOT`, and tall enough to cover the bin's span of pitch. Each glow
//! is also sized by how direct the bin is: the less its two channels move
//! together, the larger and dimmer (reverb and width become soft haze while
//! direct sounds stay crisp).
//!
//! Proportions: the in-phase part between the speakers is at least as wide
//! as the window is tall (square) and 2/3 of the whole field; each zone
//! beyond the speakers takes 1/6, at half the scale. Both parts map linearly
//! (nothing is stretched). Narrow windows crop the outer zones; wider windows
//! grow the whole field evenly until fully out of phase reaches the edges.
//! All sizes are relative to the window height, so the picture is the same
//! at any resolution.

use super::stereo::{spot_size, timing_weight, y_of, Stats, Stereo};
use crate::params::{self, Params};
use crate::dsp::{hsv, hz_to_midi, pitch_hue};
use crate::gfx::Batch;
use rustfft::num_complex::Complex32;
use std::f32::consts::{FRAC_PI_4, PI, TAU};

// All tunable values live in `params::Params` (stage width, outer zones,
// minimum spot, edge sharing, trust range, diffuse growth / dimming, ...).

/// Horizontal layout: stereo position ↔ screen x. Linear between the
/// speakers and, at a smaller scale (`OUTER`), beyond them; never stretched,
/// which would magnify small movements into long streaks.
struct Stage {
    half_w: f32,
    /// Pixels per position unit between the speakers.
    scale: f32,
    outer: f32,
}

impl Stage {
    fn new(p: &Params, w: f32, h: f32) -> Self {
        let half_w = w * 0.5;
        // The whole field (out to ±2) is (1 + outer) in-phase halves per side.
        // At least stage_width × height between the speakers; in wider windows
        // the whole field grows evenly until ±2 reaches the edges.
        let scale = (p.stage_width * h * 0.5).max(half_w / (1.0 + p.outer));
        Self { half_w, scale, outer: p.outer }
    }

    /// Screen x of a stereo position (-2 .. 2).
    fn x(&self, p: f32) -> f32 {
        let a = p.abs();
        let off = if a <= 1.0 { a * self.scale } else { self.scale * (1.0 + (a - 1.0) * self.outer) };
        self.half_w + off * p.signum()
    }

}

/// Draws every audible bin as a glow into `out` (the phosphor's live light).
pub fn draw(st: &Stereo, out: &mut Batch, w: f32, h: f32) {
    let p = params::get();
    let stage = Stage::new(&p, w, h);
    // A glow's radius is about three of its gaussian sigmas.
    let min_radius = 3.0 * p.min_spot * h;
    // Each band's distance to its neighbours (share of height), for `density`.
    let ys: Vec<f32> = st.bands.iter().map(|b| y_of(&p, b.fc)).collect();
    let dy = |i: usize| {
        let below = if i > 0 { (ys[i] - ys[i - 1]).abs() } else { 1.0 };
        let above = ys.get(i + 1).map_or(1.0, |y| (y - ys[i]).abs());
        below.min(above).max(1e-4)
    };
    for (b, (g, band)) in st.live.iter().zip(&st.bands).enumerate() {
        if g.bright < 0.004 || g.energy <= 0.0 {
            continue;
        }
        // Keep total light per area steady where neighbouring bands' glows overlap.
        let (_, band_ry) = spot_size(&p, band.fc, w, h);
        let density = (p.density * dy(b) * h / band_ry).min(1.0);
        for part in &band.parts {
            let [sl, sr] = &st.spec[part.res];
            let hz = st.bin_hz[part.res];
            for k in part.k0.max(1)..part.k1.min(sl.len()) {
                let share = part.weight * (sl[k].norm_sqr() + sr[k].norm_sqr()) / g.energy;
                let mut light = g.bright * share * density * p.exposure;
                if light < 1e-3 {
                    continue;
                }
                let f = k as f32 * hz;
                let stats = &st.bins[part.res][k];
                let pos = position(&p, stats, f);

                let (mut rx, mut ry) = spot_size(&p, f, w, h);
                // Tall enough to cover the bin's span of pitch (low bins are wide).
                let span = (y_of(&p, (f - hz * 0.5).max(1.0)) - y_of(&p, f + hz * 0.5)) * h;
                rx = rx.max(min_radius);
                ry = ry.max(min_radius).max(span * 0.75);
                // Directness: the less the channels move together, the larger and dimmer.
                let diffuse = ((p.diffuse_start - coherence(stats)) / p.diffuse_span.max(1e-3)).clamp(0.0, 1.0);
                let grow = 1.0 + p.diffuse_grow * diffuse;
                rx *= grow;
                ry *= grow;
                light /= 1.0 + p.diffuse_dim * diffuse;

                let c = hsv(pitch_hue(hz_to_midi(f)), p.saturation, 1.0, 1.0);
                let y = y_of(&p, f) * h;
                // Near-fully out of phase lies at both far edges at once (the
                // same direction): share it, so it cannot jump edge to edge.
                let mirror = 0.5 * smoothstep(p.edge_split.min(1.99), 2.0, pos.abs());
                for (x, s) in [(pos, 1.0 - mirror), (-pos, mirror)] {
                    if s > 0.0 {
                        out.glow(stage.x(x), y, rx, ry, [c[0], c[1], c[2], light * s]);
                    }
                }
            }
        }
    }
}

/// The dominant direction of a covariance: its principal eigenvector.
fn dominant(s: &Stats) -> (Complex32, Complex32) {
    let (a, b, c) = (s.l2, s.r2, s.cross);
    let power = a + b;
    if c.norm_sqr() > 1e-12 * power * power {
        let half_diff = (a - b) * 0.5;
        let lambda = (a + b) * 0.5 + (half_diff * half_diff + c.norm_sqr()).sqrt();
        (c, Complex32::new(lambda - a, 0.0))
    } else if a >= b {
        (Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0))
    } else {
        (Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0))
    }
}

/// Stereo position of a bin: -1 / 1 = hard left / right, beyond = out of
/// phase (heard outside the speakers), ±2 = fully out of phase.
pub(super) fn position(p: &Params, s: &Stats, f: f32) -> f32 {
    // Left/right level balance from the (fast) channel energies: responsive,
    // and stable even for unrelated channels.
    let level = s.r2.sqrt().atan2(s.l2.sqrt()) / FRAC_PI_4 - 1.0;

    // Polarity and timing are properties of how a sound was recorded or
    // mixed, so they are steady: take them from the slow average, where
    // phase noise cancels out. And use them only as far as they are clear:
    // a clear correlation (`TRUST`), and a phase that points clearly one way
    // (cos φ = ±1; near 90° the sign of the correlation says nothing).
    let slow = Stats { l2: s.slow_l2, r2: s.slow_r2, cross: s.slow_cross, ..*s };
    let trust = smoothstep(p.trust_lo, p.trust_hi.max(p.trust_lo + 1e-3), coherence(&slow));
    let cos_phi = if slow.cross.norm() > 0.0 { slow.cross.re / slow.cross.norm() } else { 1.0 };

    // Level and polarity: the angle whose (cos, sin) best matches the
    // dominant direction; beyond ±1 when the channels are inverted.
    let (v1, v2) = dominant(&slow);
    let mut theta = 0.5 * (2.0 * (v1 * v2.conj()).re).atan2(v1.norm_sqr() - v2.norm_sqr());
    if theta < -FRAC_PI_4 {
        theta += PI; // same direction, other sign: keep positions in -2 .. 2
    }
    // Decisive, not gradual: a bin is either between the speakers (level) or,
    // when clearly out of phase, beyond them. A gradual blend would let it
    // glide through the (often stretched) zone beyond the speakers, and the
    // afterglow would draw that path as a long thin line; a switch is a jump,
    // which leaves no trail.
    let polarity_weight = if trust * cos_phi * cos_phi > p.polarity_threshold { 1.0 } else { 0.0 };
    // Keep the responsive level; add only the polarity's push outwards.
    let slow_level = slow.r2.sqrt().atan2(slow.l2.sqrt()) / FRAC_PI_4 - 1.0;
    let polar = level + polarity_weight * (theta / FRAC_PI_4 - 1.0 - slow_level);

    let tf = timing_weight(f);
    if tf <= 0.0 {
        return polar.clamp(-2.0, 2.0);
    }
    // Low frequencies: the phase between the channels is a delay, not a
    // polarity: add how much earlier one side arrives. Clear near 0°,
    // ambiguous towards 180° (delayed or inverted?).
    let timing_trust = trust * 0.5 * (1.0 + cos_phi);
    let delay = slow.cross.arg() / (TAU * f); // > 0: left arrives first
    let timed = (level - delay / (p.max_delay_ms * 1e-3).max(1e-6) * timing_trust).clamp(-1.0, 1.0);
    (tf * timed + (1.0 - tf) * polar).clamp(-2.0, 2.0)
}

/// |correlation| of the two channels: 1 = one source, 0 = unrelated.
fn coherence(s: &Stats) -> f32 {
    if s.l2.min(s.r2) <= 1e-3 * s.l2.max(s.r2) {
        1.0 // one channel (nearly) silent: hard-panned, no ambiguity
    } else {
        (s.cross.norm() / (s.l2 * s.r2).sqrt()).min(1.0)
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Covariance of one source with left/right gains `l`, `r` (r < 0 = inverted).
    fn source(l: f32, r: f32) -> Stats {
        Stats::steady(l * l, r * r, Complex32::new(l * r, 0.0))
    }

    fn panned(p: f32) -> Stats {
        let t = (p + 1.0) * FRAC_PI_4;
        source(t.cos(), t.sin())
    }

    #[test]
    fn panned_sources_land_on_their_position() {
        for p in [-1.0, -0.6, 0.0, 0.3, 0.9] {
            for f in [300.0, 2000.0] {
                let found = position(&Params::DEFAULT, &panned(p), f);
                assert!((found - p).abs() < 0.06, "source at {p} ({f} Hz) placed at {found}");
            }
        }
    }

    #[test]
    fn out_of_phase_lies_beyond_the_speakers() {
        // Louder left, right channel inverted: beyond the left speaker.
        let p = position(&Params::DEFAULT, &source(1.0, -0.5), 2000.0);
        assert!(p < -1.2 && p > -2.0, "placed at {p}");
        // Mirror image.
        let p = position(&Params::DEFAULT, &source(-0.5, 1.0), 2000.0);
        assert!(p > 1.2 && p < 2.0, "placed at {p}");
        // Fully out of phase: at the far edge.
        let p = position(&Params::DEFAULT, &source(1.0, -1.0), 2000.0);
        assert!(p.abs() > 1.9, "placed at {p}");
    }

    #[test]
    fn earlier_arrival_leans_to_that_side() {
        // Equal levels, left leads by 0.35 ms (half the maximum delay).
        let f = 300.0;
        let s = Stats::steady(0.5, 0.5, Complex32::from_polar(0.5, TAU * f * 0.00035));
        let p = position(&Params::DEFAULT, &s, f);
        assert!(p < -0.1 && p > -0.7, "placed at {p}");
    }

    #[test]
    fn weak_correlation_does_not_flip_sides() {
        // Equal levels, tiny correlation of either sign (noise): stays near the
        // centre instead of jumping beyond the speakers and back.
        for re in [0.05, -0.05, 0.1, -0.1] {
            for f in [300.0, 2000.0] {
                let s = Stats::steady(1.0, 1.0, Complex32::new(re, 0.03));
                let p = position(&Params::DEFAULT, &s, f);
                assert!(p.abs() < 0.15, "cross {re} at {f} Hz placed at {p}");
            }
        }
        // Clearly anti-correlated: still beyond the speakers.
        let p = position(&Params::DEFAULT, &Stats::steady(1.0, 0.25, Complex32::new(-0.45, 0.0)), 2000.0);
        assert!(p < -1.2, "placed at {p}");
    }

    #[test]
    fn decorrelated_sound_is_placed_like_any_other() {
        let s = Stats::steady(0.5, 1.0, Complex32::new(0.2, 0.0));
        assert!(position(&Params::DEFAULT, &s, 2000.0) > 0.1);
    }

    #[test]
    fn stage_proportions() {
        let h = 1000.0;
        for w in [500.0, 1000.0, 1500.0, 2472.0, 3000.0] {
            let s = Stage::new(&Params::DEFAULT, w, h);
            let inner = s.x(1.0) - s.x(-1.0);
            let whole = s.x(2.0) - s.x(-2.0);
            // In-phase part at least square, and 2/3 of the whole field.
            assert!(inner >= Params::DEFAULT.stage_width * h - 1e-3, "w {w}: {inner}");
            assert!((inner / whole - 2.0 / 3.0).abs() < 1e-4);
            // Wide windows: fully out of phase exactly at the edges.
            if w >= (1.0 + Params::DEFAULT.outer) * Params::DEFAULT.stage_width * h {
                assert!(s.x(-2.0).abs() < 1e-2 && (s.x(2.0) - w).abs() < 1e-2, "w {w}");
            }
            // Continuous and monotonic across the speakers.
            assert!((s.x(1.0 + 1e-4) - s.x(1.0)).abs() < 1.0);
            let mut prev = s.x(-2.0);
            for i in 1..=400 {
                let x = s.x(-2.0 + i as f32 / 100.0);
                assert!(x > prev, "not monotonic at w {w}");
                prev = x;
            }
        }
    }
}
