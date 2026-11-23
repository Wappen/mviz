//! Small DSP helpers shared by the offline analysis and the live stereo mode.

use rustfft::{num_complex::Complex32, Fft as RFft, FftPlanner};
use std::f32::consts::TAU;
use std::sync::Arc;

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
/// Display gamma: colour values are written to the screen as they are and
/// shown with roughly this gamma.
const GAMMA: f32 = 2.2;

/// The light a colour emits: linear luminance (Rec. 709 weights).
pub fn luminance(c: crate::gfx::Rgba) -> f32 {
    0.2126 * c[0].max(0.0).powf(GAMMA) + 0.7152 * c[1].max(0.0).powf(GAMMA) + 0.0722 * c[2].max(0.0).powf(GAMMA)
}

/// Saturation whose brightness every glow colour keeps, whatever its actual
/// saturation: changing saturation (or whitening noise) changes only the
/// colour, never the amount of light.
pub const LIGHT_SATURATION: f32 = 0.6;

/// The same for the bass background, which is more saturated.
pub const BACKGROUND_SATURATION: f32 = 0.85;

/// The average light of all hues at saturation `light_sat`: what every hue
/// emits with `even` = 1.
fn mean_hue_light(light_sat: f32) -> f32 {
    fn mean(s: f32) -> f32 {
        (0..360).map(|i| luminance(hsv(i as f32 / 360.0, s, 1.0, 1.0))).sum::<f32>() / 360.0
    }
    // Cached for the two saturations in use (it is needed for every glow).
    static GLOWS: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| mean(LIGHT_SATURATION));
    static BACKGROUND: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| mean(BACKGROUND_SATURATION));
    if light_sat == LIGHT_SATURATION {
        *GLOWS
    } else if light_sat == BACKGROUND_SATURATION {
        *BACKGROUND
    } else {
        mean(light_sat)
    }
}

/// The colour of `hue` at `saturation` for a glow (see `colour`).
pub fn glow_colour(hue: f32, saturation: f32, even: f32) -> crate::gfx::Rgba {
    colour(hue, saturation, LIGHT_SATURATION, even)
}

/// The colour of `hue` at `saturation`. Its light is the light the hue has
/// at `light_sat`, moved by `even` (0..1) towards the same light for every
/// hue (yellow is naturally far brighter than blue).
pub fn colour(hue: f32, saturation: f32, light_sat: f32, even: f32) -> crate::gfx::Rgba {
    let c = hsv(hue, saturation, 1.0, 1.0);
    let own = luminance(hsv(hue, light_sat, 1.0, 1.0));
    let target = own + (mean_hue_light(light_sat) - own) * even;
    // Scaling a colour by k scales its light by k^GAMMA.
    let k = (target / luminance(c).max(1e-9)).powf(1.0 / GAMMA);
    [c[0] * k, c[1] * k, c[2] * k, 1.0]
}

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

/// ISO 226:2003 equal-loudness contours: frequencies (Hz) and the parameters
/// αf, Lu, Tf of each.
const ISO226: [(f32, f32, f32, f32); 29] = [
    (20.0, 0.532, -31.6, 78.5),
    (25.0, 0.506, -27.2, 68.7),
    (31.5, 0.480, -23.0, 59.5),
    (40.0, 0.455, -19.1, 51.1),
    (50.0, 0.432, -15.9, 44.0),
    (63.0, 0.409, -13.0, 37.5),
    (80.0, 0.387, -10.3, 31.5),
    (100.0, 0.367, -8.1, 26.5),
    (125.0, 0.349, -6.2, 22.1),
    (160.0, 0.330, -4.5, 17.9),
    (200.0, 0.315, -3.1, 14.4),
    (250.0, 0.301, -2.0, 11.4),
    (315.0, 0.288, -1.1, 8.6),
    (400.0, 0.276, -0.4, 6.2),
    (500.0, 0.267, 0.0, 4.4),
    (630.0, 0.259, 0.3, 3.0),
    (800.0, 0.253, 0.5, 2.2),
    (1000.0, 0.250, 0.0, 2.4),
    (1250.0, 0.246, -2.7, 3.5),
    (1600.0, 0.244, -4.1, 1.7),
    (2000.0, 0.243, -1.0, -1.3),
    (2500.0, 0.243, 1.7, -4.2),
    (3150.0, 0.243, 2.5, -6.0),
    (4000.0, 0.242, 1.2, -5.4),
    (5000.0, 0.242, -2.1, -1.5),
    (6300.0, 0.245, -7.1, 6.0),
    (8000.0, 0.254, -11.2, 12.6),
    (10000.0, 0.271, -10.7, 13.9),
    (12500.0, 0.301, -3.1, 12.3),
];

/// Sound pressure level (dB) that sounds as loud as `phon` at the i-th
/// ISO 226 frequency.
fn iso226_spl(i: usize, phon: f32) -> f32 {
    let (_, af, lu, tf) = ISO226[i];
    let a = 4.47e-3 * (10f32.powf(0.025 * phon) - 1.15) + (0.4 * 10f32.powf((tf + lu) / 10.0 - 9.0)).powf(af);
    10.0 / af * a.log10() - lu + 94.0
}

/// How much louder (dB) the ear hears a tone at `f` than one of the same
/// level at 1 kHz, at a listening level of `phon`: negative for deep bass
/// and the highest treble, positive around 3 kHz. Interpolated in log
/// frequency; held constant beyond 20 Hz – 12.5 kHz.
pub fn ear_gain_db(f: f32, phon: f32) -> f32 {
    let f = f.clamp(ISO226[0].0, ISO226[ISO226.len() - 1].0);
    let i = ISO226.partition_point(|e| e.0 <= f).clamp(1, ISO226.len() - 1);
    let (f0, f1) = (ISO226[i - 1].0, ISO226[i].0);
    let t = ((f / f0).ln() / (f1 / f0).ln()).clamp(0.0, 1.0);
    let spl = iso226_spl(i - 1, phon) + (iso226_spl(i, phon) - iso226_spl(i - 1, phon)) * t;
    phon - spl
}

/// Below this the body feels bass fully (chest, floor); above it the felt
/// part falls off at `FELT_SLOPE` dB per octave.
const FELT_FROM: f32 = 60.0;
const FELT_SLOPE: f32 = 12.0;

/// Perceived loudness gain (dB, relative to 1 kHz): what is heard (ISO 226
/// at `phon`) plus what is felt. Deep bass also moves the body; that felt
/// part is `felt_db` (relative to hearing 1 kHz) below `FELT_FROM`, falls
/// off above it, and only exists when it is loud: it fades in from 55 to
/// 85 phon. This felt part is a simple model, not a standard.
pub fn perceived_gain_db(f: f32, phon: f32, felt_db: f32) -> f32 {
    let heard = ear_gain_db(f, phon);
    let loud_enough = ((phon - 55.0) / 30.0).clamp(0.0, 1.0);
    if loud_enough <= 0.0 {
        return heard;
    }
    let felt = felt_db - FELT_SLOPE * (f / FELT_FROM).log2().max(0.0);
    let power = 10f32.powf(heard / 10.0) + loud_enough * 10f32.powf(felt / 10.0);
    10.0 * power.log10()
}

#[cfg(test)]
mod ear_tests {
    use super::{ear_gain_db, perceived_gain_db};

    #[test]
    fn felt_bass_adds_to_heard_bass_when_loud() {
        // Loud: deep bass is felt; the mids and treble are unchanged.
        let heard = ear_gain_db(35.0, 90.0);
        let felt = perceived_gain_db(35.0, 90.0, -6.0);
        println!("35 Hz at 90 phon: heard {heard:.1} dB, heard + felt {felt:.1} dB");
        assert!(felt > heard + 3.0 && (felt - -6.0).abs() < 3.0, "{heard} {felt}");
        assert!((perceived_gain_db(1000.0, 90.0, -6.0) - ear_gain_db(1000.0, 90.0)).abs() < 0.1);
        // Quiet: nothing is felt.
        assert_eq!(perceived_gain_db(35.0, 50.0, -6.0), ear_gain_db(35.0, 50.0));
    }

    #[test]
    fn equal_loudness_matches_iso226() {
        // 1 kHz is the reference by definition.
        assert!(ear_gain_db(1000.0, 60.0).abs() < 0.2);
        // 60-phon levels: 100 Hz ≈ 79 dB; around 3–4 kHz the ear is most
        // sensitive, a few dB below 60 (gain = 60 − level).
        let g100 = ear_gain_db(100.0, 60.0);
        let g4k = ear_gain_db(4000.0, 60.0);
        println!("60 phon: 100 Hz {g100:.1} dB, 4 kHz {g4k:.1} dB, 30 Hz {:.1}, 10 kHz {:.1}", ear_gain_db(30.0, 60.0), ear_gain_db(10000.0, 60.0));
        assert!((g100 + 19.0).abs() < 1.5, "{g100}");
        assert!((1.0..5.0).contains(&g4k), "{g4k}");
        // Louder listening: flatter curve, less bass loss.
        assert!(ear_gain_db(40.0, 90.0) > ear_gain_db(40.0, 50.0));
    }
}

#[cfg(test)]
mod colour_tests {
    use super::*;

    #[test]
    fn saturation_changes_colour_not_light() {
        for hue in [0.0, 0.17, 0.33, 0.5, 0.66, 0.83] {
            let reference = luminance(hsv(hue, LIGHT_SATURATION, 1.0, 1.0));
            for s in [0.0, 0.3, 0.6, 0.93] {
                let l = luminance(glow_colour(hue, s, 0.0));
                assert!((l - reference).abs() < 1e-4, "hue {hue} at saturation {s}: {l} vs {reference}");
            }
            // The default saturation looks as before.
            let (a, b) = (glow_colour(hue, LIGHT_SATURATION, 0.0), hsv(hue, LIGHT_SATURATION, 1.0, 1.0));
            assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5), "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn even_hues_emit_the_same_light() {
        let lights: Vec<f32> = (0..12).map(|i| luminance(glow_colour(i as f32 / 12.0, 0.6, 1.0))).collect();
        let (lo, hi) = lights.iter().fold((f32::MAX, 0.0f32), |(lo, hi), &l| (lo.min(l), hi.max(l)));
        assert!(hi / lo < 1.001, "{lights:?}");
        // Without it, yellow is several times brighter than blue.
        let (y, b) = (luminance(glow_colour(1.0 / 6.0, 0.6, 0.0)), luminance(glow_colour(2.0 / 3.0, 0.6, 0.0)));
        assert!(y / b > 4.0, "{y} / {b}");
    }
}
