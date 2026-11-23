//! The background's bass for live input, measured as the audio arrives
//! (songs get it from the full analysis, `analysis::Timeline`).
//!
//! Level: the same 80 Hz low-pass and 10 Hz envelope as the song analysis,
//! but run forwards only (no future to look at). Like a song's, it is
//! scaled to the field's loudness reference by `Stereo`.
//!
//! Note: every 0.05 s, the strongest bass frequency (30–160 Hz) in the last
//! ~0.34 s. A new note takes over once it has been heard twice in a row, so
//! the colour doesn't flicker between notes.

use super::Bass;
use crate::analysis::{LowPass, ENVELOPE_HZ, ENV_BLOCK};
use crate::audio::Samples;
use crate::dsp::{hz_to_midi, parabolic, Fft};

const PITCH_N: usize = 16384;
const PITCH_EVERY: f32 = 0.05;
/// Bass quieter than this (relative to loud bass) gives no note.
const PITCH_MIN_LEVEL: f32 = 0.15;

pub struct LiveBass {
    rate: u32,
    bass: [LowPass; 2],
    smooth: [LowPass; 2],
    /// Bass power being summed for the current block, and its length.
    acc: [f64; 2],
    n: usize,
    /// Smoothed bass power per channel.
    power: [f64; 2],
    /// The next frame to read.
    next: Option<i64>,
    fft: Fft,
    since_pitch: f32,
    /// The note shown, and the one heard last (with how often in a row).
    pc: Option<f32>,
    candidate: Option<(f32, u32)>,
    last: Bass,
}

fn filters(fc: f64, q: f64, rate: f64) -> [LowPass; 2] {
    [LowPass::new(fc, q, rate), LowPass::new(fc, q, rate)]
}

impl LiveBass {
    pub fn new(rate: u32) -> Self {
        let block_rate = rate as f64 / ENV_BLOCK as f64;
        Self {
            rate,
            bass: filters(80.0, 0.5, rate as f64),
            smooth: filters(ENVELOPE_HZ, std::f64::consts::FRAC_1_SQRT_2, block_rate),
            acc: [0.0; 2],
            n: 0,
            power: [0.0; 2],
            next: None,
            fft: Fft::new(PITCH_N),
            since_pitch: PITCH_EVERY,
            pc: None,
            candidate: None,
            last: Bass { pc: None, level: (0.0, 0.0) },
        }
    }

    /// Reads the audio up to `frame` (the newest) and returns the bass now
    /// (amplitudes, as measured). `loud` is the amplitude of loud bass, below
    /// which notes are not looked for.
    pub fn update(&mut self, s: &dyn Samples, frame: i64, dt: f32, loud: f32) -> &Bass {
        // Held still (paused): nothing new.
        if self.next == Some(frame) {
            return &self.last;
        }
        // After a gap (or at the start), settle on the last quarter second.
        let start = match self.next {
            Some(n) if n <= frame && frame - n < self.rate as i64 => n,
            _ => frame - self.rate as i64 / 4,
        };
        for i in start..frame {
            for ch in 0..2 {
                let y = self.bass[ch].run(s.sample(i, ch) as f64);
                self.acc[ch] += y * y;
            }
            self.n += 1;
            if self.n == ENV_BLOCK {
                for ch in 0..2 {
                    self.power[ch] = self.smooth[ch].run(self.acc[ch] / ENV_BLOCK as f64);
                }
                self.acc = [0.0; 2];
                self.n = 0;
            }
        }
        self.next = Some(frame);

        let (l, r) = (self.power[0].max(0.0).sqrt() as f32, self.power[1].max(0.0).sqrt() as f32);
        let mono = ((l * l + r * r) / 2.0).sqrt();

        self.since_pitch += dt;
        if self.since_pitch >= PITCH_EVERY {
            self.since_pitch = 0.0;
            if mono / loud > PITCH_MIN_LEVEL {
                let pc = self.pitch_class(s, frame);
                self.hear(pc);
            }
        }
        self.last = Bass { pc: self.pc, level: (l, r) };
        &self.last
    }

    /// Pitch class of the strongest bass frequency in the window ending at `frame`.
    fn pitch_class(&mut self, s: &dyn Samples, frame: i64) -> f32 {
        let start = frame - PITCH_N as i64;
        let spec = self.fft.run(|j| 0.5 * (s.sample(start + j as i64, 0) + s.sample(start + j as i64, 1)));
        let hz = self.rate as f32 / PITCH_N as f32;
        let (k0, k1) = (((30.0 / hz) as usize).max(1), (160.0 / hz) as usize);
        let best = (k0..k1).max_by(|&a, &b| spec[a].norm().total_cmp(&spec[b].norm())).unwrap_or(k0);
        let l = |k: usize| spec[k].norm().max(1e-12).ln();
        let kf = best as f32 + parabolic(l(best - 1), l(best), l(best + 1));
        hz_to_midi(kf * hz).rem_euclid(12.0)
    }

    /// A note heard now: it takes over once heard twice in a row.
    fn hear(&mut self, pc: f32) {
        let same = |a: f32, b: f32| (a.round() as i32).rem_euclid(12) == (b.round() as i32).rem_euclid(12);
        self.candidate = match self.candidate {
            Some((c, n)) if same(c, pc) => Some((pc, n + 1)),
            _ => Some((pc, 1)),
        };
        if let Some((c, n)) = self.candidate {
            if n >= 2 && !self.pc.is_some_and(|p| same(p, c)) {
                self.pc = Some(c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sine on both channels, `amp` on the left and `amp * right` on the right.
    struct Sine {
        hz: f32,
        rate: f32,
        amp: f32,
        right: f32,
    }

    impl Samples for Sine {
        fn sample(&self, i: i64, ch: usize) -> f32 {
            let v = self.amp * (i as f32 / self.rate * self.hz * std::f32::consts::TAU).sin();
            if ch == 0 { v } else { v * self.right }
        }
    }

    fn run(lb: &mut LiveBass, s: &Sine, secs: f32) -> Bass {
        let rate = s.rate as i64;
        let frames = (secs * 60.0) as i64;
        let mut out = Bass { pc: None, level: (0.0, 0.0) };
        for f in 1..=frames {
            let b = lb.update(s, f * rate / 60, 1.0 / 60.0, 0.2);
            out = Bass { pc: b.pc, level: b.level };
        }
        out
    }

    #[test]
    fn finds_the_note_and_each_sides_level() {
        let mut lb = LiveBass::new(48000);
        // A2 = 110 Hz: pitch class 9 (A). Louder on the left.
        let b = run(&mut lb, &Sine { hz: 110.0, rate: 48000.0, amp: 0.5, right: 0.5 }, 2.0);
        let pc = b.pc.expect("a note");
        assert!((pc - 9.0).abs() < 0.3, "pitch class {pc}");
        // The 80 Hz low-pass (Q 0.5) passes 1 / (1 + (110/80)²) ≈ 0.35 of
        // 110 Hz: RMS ≈ 0.5 · 0.35 / √2.
        assert!((b.level.0 - 0.122).abs() < 0.01, "left {}", b.level.0);
        assert!((b.level.1 / b.level.0 - 0.5).abs() < 0.05, "right / left {}", b.level.1 / b.level.0);
    }

    #[test]
    fn a_new_note_takes_over_and_silence_keeps_the_last() {
        let mut lb = LiveBass::new(48000);
        run(&mut lb, &Sine { hz: 110.0, rate: 48000.0, amp: 0.5, right: 1.0 }, 1.5);
        // D2 ≈ 73.4 Hz: pitch class 2.
        let b = run(&mut lb, &Sine { hz: 73.42, rate: 48000.0, amp: 0.5, right: 1.0 }, 1.5);
        assert!((b.pc.unwrap() - 2.0).abs() < 0.3, "pitch class {:?}", b.pc);
        let b = run(&mut lb, &Sine { hz: 73.42, rate: 48000.0, amp: 0.0, right: 1.0 }, 1.0);
        assert!((b.pc.unwrap() - 2.0).abs() < 0.3);
        assert!(b.level.0 < 0.01, "fades with the sound: {}", b.level.0);
    }
}

