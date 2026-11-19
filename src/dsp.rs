//! Small DSP helpers shared by the offline analysis and the live stereo mode.

use rustfft::{num_complex::Complex32, Fft as RFft, FftPlanner};
use std::f32::consts::TAU;
use std::sync::Arc;

/// Magnitudes below this (about -80 dBFS) are treated as silence.
pub const ABS_FLOOR: f32 = 1e-4;

/// A reusable Hann-windowed forward FFT.
pub struct Fft {
    pub n: usize,
    plan: Arc<dyn RFft<f32>>,
    window: Vec<f32>,
    norm: f32,
    buf: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        let plan = FftPlanner::new().plan_fft_forward(n);
        let window: Vec<f32> = (0..n)
            .map(|i| 0.5 - 0.5 * (TAU * i as f32 / n as f32).cos())
            .collect();
        // Scale so a full-scale sine peaks at ~1.0 regardless of window length.
        let norm = 2.0 / window.iter().sum::<f32>();
        let scratch = vec![Complex32::default(); plan.get_inplace_scratch_len()];
        Self { n, plan, window, norm, buf: vec![Complex32::default(); n], scratch }
    }

    /// Transforms `n` samples provided by `sample(j)`; returns bins `0..=n/2`.
    pub fn run(&mut self, mut sample: impl FnMut(usize) -> f32) -> &[Complex32] {
        for j in 0..self.n {
            self.buf[j] = Complex32::new(sample(j) * self.window[j], 0.0);
        }
        self.plan.process_with_scratch(&mut self.buf, &mut self.scratch);
        let half = self.n / 2;
        for c in &mut self.buf[..=half] {
            *c *= self.norm;
        }
        &self.buf[..=half]
    }
}

pub fn hz_to_midi(f: f32) -> f32 {
    69.0 + 12.0 * (f / 440.0).log2()
}

/// Every pitch class gets a fixed hue: C = 0, C# = 1/12, ...
pub fn pitch_hue(midi: f32) -> f32 {
    midi.rem_euclid(12.0) / 12.0
}

pub fn hsv(h: f32, s: f32, v: f32, a: f32) -> crate::gfx::Rgba {
    let h6 = h.rem_euclid(1.0) * 6.0;
    let c = v * s;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m, a]
}

/// p in 0..=1; reorders `v`.
pub fn percentile(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let i = ((v.len() - 1) as f32 * p).round() as usize;
    *v.select_nth_unstable_by(i, |a, b| a.total_cmp(b)).1
}

/// Fractional offset of a peak from three (log) magnitudes around it.
pub fn parabolic(a: f32, b: f32, c: f32) -> f32 {
    let d = a - 2.0 * b + c;
    if d.abs() < 1e-12 {
        0.0
    } else {
        (0.5 * (a - c) / d).clamp(-0.5, 0.5)
    }
}
