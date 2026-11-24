//! Every visual parameter in one place (tuned live and settled on these
//! values). Change them here to adjust the look.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    // --- Layout ---
    /// Minimum width from hard left to hard right, × window height (1 = square).
    pub stage_width: f32,
    /// Each zone beyond the speakers, relative to one in-phase half.
    pub outer: f32,
    /// Top of the pitch axis' main range (Hz); above it is the top margin.
    pub f_hi: f32,
    /// Space above `f_hi` and below the lowest pitch, × window height.
    pub top_margin: f32,
    pub bottom_margin: f32,

    // --- Glow size ---
    /// Base glow radius at 100 Hz, × window height.
    pub size_base: f32,
    /// Base radius ∝ (f / 100 Hz)^−size_exp: how fast glows shrink with pitch.
    pub size_exp: f32,
    /// Smallest base radius, × window height.
    pub size_floor: f32,
    /// Smallest glow (gaussian sigma), both directions, × window height.
    pub min_spot: f32,
    /// Lowest bass: horizontal radius × window width, and height / width.
    pub carpet_width: f32,
    pub carpet_flat: f32,
    /// How fast the carpet effect fades with pitch (higher = faster).
    pub carpet_power: f32,
    /// A fully diffuse glow is (1 + grow)× larger and (1 + dim)× dimmer.
    pub diffuse_grow: f32,
    pub diffuse_dim: f32,
    /// Coherence below which a glow starts counting as diffuse, and the
    /// range over which it becomes fully diffuse.
    pub diffuse_start: f32,
    pub diffuse_span: f32,

    // --- Brightness and colour ---
    /// Overall light multiplier.
    pub exposure: f32,
    /// Loudness range shown (dB) and the curve applied to it.
    pub db_range: f32,
    pub bright_curve: f32,
    /// Treble boost (dB per octave).
    pub tilt_db: f32,
    pub saturation: f32,
    /// Every hue emits the same light (0 = natural: yellow far brighter than
    /// blue; 1 = equal, so brightness shows only loudness, not the note).
    pub even_hues: f32,
    /// Evens out light where neighbouring bands' glows overlap.
    pub density: f32,
    /// Brightness follows perceived loudness (heard per ISO 226, plus deep
    /// bass felt in the body) instead of the plain treble boost:
    /// 0 = treble boost only, 1 = perceived loudness only.
    pub loudness: f32,
    /// Listening level (phon): quieter = less bass heard, and none felt.
    pub phon: f32,
    /// How strongly deep bass is felt (dB, relative to hearing 1 kHz).
    pub felt_bass: f32,
    /// Sounds without a clear pitch (noise, cymbals, breath) lose this much
    /// of their colour (0 = all coloured, 1 = noise fully white).
    pub noise_white: f32,
    /// How far (dB) a band's strongest peak must stand out above its
    /// surroundings: up to `tone_lo_db` it is noise, from `tone_hi_db` a
    /// clear tone (in between, partly pitched).
    pub tone_lo_db: f32,
    pub tone_hi_db: f32,
    /// Unpitched light is dimmed by this much (0 = not at all), so noise
    /// does not wash out the pitched light it overlaps.
    pub noise_dim: f32,
    /// Bright light rolls off softly above this level.
    pub knee: f32,

    // --- Afterglow ---
    /// Afterglow time constant (s). 0.25 when the phosphor took all light
    /// once per frame; taking the treble at every analysis step also catches
    /// its flicker within a frame, so 0.19 keeps the glow as bright as it was.
    pub fade: f32,
    /// Spread of fading light per 1/60 s (phosphor pixels, see `gfx::blur_offset`).
    pub fade_blur: f32,

    // --- Placement ---
    /// Averaging time (s) for level balance and diffuseness.
    pub smooth: f32,
    /// Averaging time (s) for polarity and timing.
    pub slow: f32,
    /// Level fall time (s); rises are instant.
    pub release: f32,
    /// Memory (s) of the level balance while a band's sound is far below
    /// its average (a hit dying away), so the hit's direction goes with it:
    /// from `fade_lo` times below (not yet) to `fade_hi` (fully).
    pub forget: f32,
    pub fade_lo: f32,
    pub fade_hi: f32,
    /// Arrival-time difference mapped to hard left/right (ms).
    pub max_delay_ms: f32,
    /// Timing used fully below / not at all above (Hz).
    pub delay_fade_lo: f32,
    pub delay_fade_hi: f32,
    /// Coherence range over which polarity and timing become trusted.
    pub trust_lo: f32,
    pub trust_hi: f32,
    /// How clearly out of phase a bin must be to go beyond the speakers.
    pub polarity_threshold: f32,
    /// Position from which out-of-phase light is shared with the other edge.
    pub edge_split: f32,

    // --- Background ---
    /// Peak brightness of the bass background glow, and its curve.
    pub flash_max: f32,
    pub bass_curve: f32,
    /// Unpitched bass (rumble, noisy kicks) loses this much of the
    /// background's colour (0 = always coloured, 1 = fully white).
    pub bg_noise_white: f32,
    /// Background colour without bass (RGB).
    pub base: [f32; 3],
}

impl Params {
    pub const DEFAULT: Params = Params {
        stage_width: 1.0,
        outer: 0.5,
        f_hi: 16000.0,
        top_margin: 0.05,
        bottom_margin: 0.0,
        size_base: 0.25,
        size_exp: 0.7,
        size_floor: 0.008,
        min_spot: 0.006,
        carpet_width: 0.8,
        carpet_flat: 0.25,
        carpet_power: 4.5,
        diffuse_grow: 2.0,
        diffuse_dim: 1.5,
        diffuse_start: 0.92,
        diffuse_span: 0.5,
        exposure: 1.0,
        db_range: 42.0,
        bright_curve: 1.6,
        tilt_db: 3.0,
        saturation: 0.9,
        even_hues: 1.0,
        density: 2.5,
        loudness: 0.5,
        phon: 100.0,
        felt_bass: 0.0,
        noise_white: 1.0,
        tone_lo_db: 3.0,
        tone_hi_db: 8.0,
        noise_dim: 0.5,
        knee: 0.8,
        fade: 0.19,
        fade_blur: 1.2,
        smooth: 0.1,
        slow: 0.4,
        release: 0.025,
        forget: 0.015,
        fade_lo: 1.3,
        fade_hi: 3.0,
        max_delay_ms: 0.7,
        delay_fade_lo: 500.0,
        delay_fade_hi: 700.0,
        trust_lo: 0.25,
        trust_hi: 0.6,
        polarity_threshold: 0.5,
        edge_split: 1.7,
        flash_max: 0.2,
        bass_curve: 1.6,
        bg_noise_white: 0.5,
        base: [0.012, 0.014, 0.03],
    };
}

impl Default for Params {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The parameters in use.
pub fn get() -> Params {
    Params::DEFAULT
}
