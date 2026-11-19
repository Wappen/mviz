//! The visualizer: background (`draw_background`), live stereo analysis
//! (`stereo`) and the sound field drawn from it (`field`).

pub mod field;
pub mod stereo;

use crate::analysis::Timeline;
use crate::dsp::{colour, BACKGROUND_SATURATION};
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
///
/// The dark base goes into `base`; the bass light goes into `light`, the
/// phosphor's live layer, so it gets the same afterglow as the field: a new
/// bass note's colour appears at once while the previous one fades out, and
/// the glow fades out after the bass instead of switching off.
/// `tonality` (0..1) is how pitched the bass is right now: unpitched bass
/// (rumble, a noisy kick) loses colour by `bg_noise_white`.
#[allow(clippy::too_many_arguments)]
pub fn draw_background(base_out: &mut Batch, light: &mut Batch, tl: Option<&Timeline>, t: f32, tonality: f32, strength: f32, w: f32, h: f32) {
    base_out.rect(0.0, 0.0, w, h, base());
    let Some(tl) = tl.filter(|_| strength > 0.0) else { return };
    let i = tl.bass.partition_point(|b| b.t <= t);
    let hue = i.checked_sub(1).map_or(0.6, |i| tl.bass[i].pc / 12.0);
    let p = params::get();
    let saturation = BACKGROUND_SATURATION * (1.0 - p.bg_noise_white * (1.0 - tonality));
    let c = colour(hue, saturation, BACKGROUND_SATURATION, p.even_hues);
    // A gentle curve: quiet bass stays near black, loud bass lights up.
    let shape = |a: f32| a.clamp(0.0, 1.2).powf(p.bass_curve);
    let (l, r) = bass_level(tl, t);
    let (gl, gr) = (shape(l), shape(r));
    let m = p.flash_max * strength;
    let lit = |g: f32| [c[0] * m * g, c[1] * m * g, c[2] * m * g, 1.0];
    light.gradient_h(0.0, 0.0, w, h, lit(gl), lit(gr));
}
