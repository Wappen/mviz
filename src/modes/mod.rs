//! The visualizer: background (`draw_background`), live stereo analysis
//! (`stereo`) and the sound field drawn from it (`field`).

pub mod field;
pub mod stereo;

use crate::analysis::Timeline;
use crate::dsp::hsv;
use crate::gfx::{Batch, Rgba};
use crate::params;

/// Low-passed bass amplitude (left, right) at time `t`, interpolated between hops.
fn bass_level(tl: &Timeline, t: f32) -> (f32, f32) {
    let x = (t / tl.hop).max(0.0);
    let i = (x as usize).min(tl.bass_env.len() - 1);
    let (a, b) = (tl.bass_env[i], tl.bass_env[(i + 1).min(tl.bass_env.len() - 1)]);
    let f = x - i as f32;
    (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f)
}

/// Colour of the empty background (`params::Params::base`).
pub fn base() -> Rgba {
    let b = params::get().base;
    [b[0], b[1], b[2], 1.0]
}

/// Dark background lit by the low-passed bass (each side by its own
/// channel), at `strength` (0..1). Its hue is the pitch class of the latest
/// bass note. Plain until the analysis (`tl`) is available.
pub fn draw_background(out: &mut Batch, tl: Option<&Timeline>, t: f32, strength: f32, w: f32, h: f32) {
    let Some(tl) = tl.filter(|_| strength > 0.0) else {
        let b = base();
        out.rect(0.0, 0.0, w, h, b);
        return;
    };
    let i = tl.bass.partition_point(|b| b.t <= t);
    let hue = i.checked_sub(1).map_or(0.6, |i| tl.bass[i].pc / 12.0);
    let c = hsv(hue, 0.85, 1.0, 1.0);
    let p = params::get();
    // A gentle curve: quiet bass stays near black, loud bass lights up.
    let shape = |a: f32| a.clamp(0.0, 1.2).powf(p.bass_curve);
    let (l, r) = bass_level(tl, t);
    let (gl, gr) = (shape(l), shape(r));
    let (b, m) = (p.base, p.flash_max * strength);
    let lit = |g: f32| [b[0] + c[0] * m * g, b[1] + c[1] * m * g, b[2] + c[2] * m * g, 1.0];
    out.gradient_h(0.0, 0.0, w, h, lit(gl), lit(gr));
}
