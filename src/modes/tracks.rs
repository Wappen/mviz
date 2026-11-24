//! Partial tracks: the stable sinusoids of the spectrum (the partials of
//! notes, drones, whistles), followed from one analysis to the next.
//!
//! A bin only knows what passes through it: when a snare covers it for a
//! moment, or a partial glides on to the next bin (vibrato), its averaged
//! direction mixes in whatever else was there. A track belongs to one
//! partial: it learns its direction only while its peak clearly stands out
//! (a glimpse), holds it while the peak is covered, and carries it along as
//! the partial moves. The bins around a track take their position from it;
//! all other bins (noise, hits, reverb) keep their own.

use super::stereo::Stats;
use crate::params::Params;
use rustfft::num_complex::Complex32;

/// Bins either side of a track's peak that take its position: a Hann
/// window's main lobe.
const LOBE: usize = 2;

struct Track {
    /// Frequency in bins (interpolated between them).
    pos: f32,
    stats: Stats,
    /// How long it has been followed (s), and how long its peak has been
    /// missing (s).
    age: f32,
    missing: f32,
}

struct Peak {
    pos: f32,
    k: usize,
    /// How far it stands out above its surroundings (dB).
    db: f32,
}

/// The tracks of one FFT size.
#[derive(Default)]
pub(super) struct Tracks {
    tracks: Vec<Track>,
    /// Per bin: the track whose position it takes (`NONE` = its own).
    owner: Vec<u32>,
}

const NONE: u32 = u32::MAX;

impl Tracks {
    /// Follows the peaks of one analysis (`l`, `r`: its spectrum) within
    /// bins `k0 .. k1`; `bins`: every bin's own averaged stats (a new track
    /// starts from its bin's).
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, p: &Params, dt: f32, l: &[Complex32], r: &[Complex32], bins: &[Stats], k0: usize, k1: usize) {
        let k1 = k1.min(l.len());
        let power = |k: usize| l[k].norm_sqr() + r[k].norm_sqr();
        let mut peaks = Vec::new();
        for k in (k0 + 1).max(LOBE + 3)..k1.saturating_sub(LOBE + 3) {
            let (a, b, c) = (power(k - 1), power(k), power(k + 1));
            if !(b > a && b >= c) {
                continue;
            }
            // Above the higher of the two valleys either side (as `tonality`).
            let side = |ks: std::ops::RangeInclusive<usize>| ks.map(power).fold(f32::INFINITY, f32::min);
            let around = side(k - 5..=k - 2).max(side(k + 2..=k + 5));
            let db = 10.0 * ((b + 1e-20) / (around + 1e-20)).log10();
            if db < p.peak_db {
                continue;
            }
            // Parabola through the log powers: where the peak really is.
            let (la, lb, lc) = ((a + 1e-20).ln(), (b + 1e-20).ln(), (c + 1e-20).ln());
            let den = la - 2.0 * lb + lc;
            let d = if den < 0.0 { (0.5 * (la - lc) / den).clamp(-0.5, 0.5) } else { 0.0 };
            peaks.push(Peak { pos: k as f32 + d, k, db });
        }

        // Link: nearest pairs first, each track and peak once.
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, t) in self.tracks.iter().enumerate() {
            for (pi, pk) in peaks.iter().enumerate() {
                let d = (pk.pos - t.pos).abs();
                if d <= p.track_move {
                    pairs.push((d, ti, pi));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut track_peak: Vec<Option<usize>> = vec![None; self.tracks.len()];
        let mut taken = vec![false; peaks.len()];
        for (_, ti, pi) in pairs {
            if track_peak[ti].is_none() && !taken[pi] {
                track_peak[ti] = Some(pi);
                taken[pi] = true;
            }
        }

        let fast = 1.0 - (-dt / p.track_smooth.max(1e-3)).exp();
        let slow = 1.0 - (-dt / p.slow.max(1e-3)).exp();
        for (t, found) in self.tracks.iter_mut().zip(&track_peak) {
            t.age += dt;
            let Some(&pi) = found.as_ref() else {
                t.missing += dt;
                continue;
            };
            let pk = &peaks[pi];
            t.pos = pk.pos;
            t.missing = 0.0;
            // A glimpse: learn as far as the peak stands out (a snare
            // over it fills its valleys, so it learns little then).
            let glimpse = smoothstep(p.glimpse_lo_db, p.glimpse_hi_db.max(p.glimpse_lo_db + 1e-3), pk.db);
            t.stats.learn(l[pk.k], r[pk.k], fast * glimpse, slow * glimpse);
        }
        self.tracks.retain(|t| t.missing <= p.track_hold);
        for (pk, _) in peaks.iter().zip(&taken).filter(|(_, t)| !**t) {
            self.tracks.push(Track { pos: pk.pos, stats: bins[pk.k], age: 0.0, missing: 0.0 });
        }

        // Each bin near a grown track takes its position (the nearest one's).
        self.owner.clear();
        self.owner.resize(l.len(), NONE);
        let mut dist = vec![f32::INFINITY; l.len()];
        for (i, t) in self.tracks.iter().enumerate() {
            if t.age < p.track_birth {
                continue;
            }
            let c = t.pos.round() as usize;
            let lo = c.saturating_sub(LOBE);
            let hi = (c + LOBE).min(l.len() - 1);
            for (k, (best, owner)) in dist[lo..=hi].iter_mut().zip(&mut self.owner[lo..=hi]).enumerate() {
                let d = ((lo + k) as f32 - t.pos).abs();
                if d < *best {
                    *best = d;
                    *owner = i as u32;
                }
            }
        }
    }

    /// The stats that place bin `k`: its track's, if it belongs to one.
    pub fn placing(&self, k: usize) -> Option<&Stats> {
        match self.owner.get(k) {
            Some(&i) if i != NONE => Some(&self.tracks[i as usize].stats),
            _ => None,
        }
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
