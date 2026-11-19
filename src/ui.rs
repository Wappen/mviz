//! Minimal controls:
//!   right-click / Space pause / resume
//!   wheel               volume
//!   left-drag           move the window (from fullscreen: leaves
//!                       fullscreen, then moves the window)
//!   double left-click   fullscreen on/off (and bring to front)
//!   waveform timeline   at the bottom while the cursor is visible over the
//!                       window (click or drag on it to seek)

use crate::audio::Player;
use crate::gfx::{Batch, Rgba, Text};
use winit::event::MouseButton;
use winit::window::{Fullscreen, Window};

const ZONE: f32 = 96.0;
const IDLE: f64 = 2.0;
const PAD: f32 = 16.0;
const WAVE_H: f32 = 36.0;
const DOUBLE_CLICK: f64 = 0.35;
/// The cursor must move this far with the button held before the window moves.
const DRAG_START: f32 = 4.0;
/// Touchpad scrolling: pixels per volume step.
const SCROLL_STEP_PX: f32 = 40.0;

#[derive(PartialEq)]
enum Drag {
    None,
    Seek,
    /// Left button held outside the timeline; becomes a window move once it travels.
    Pending { x: f32, y: f32 },
}

pub struct Ui {
    mouse: (f32, f32),
    mouse_in: bool,
    last_move: f64,
    bar_alpha: f32,
    drag: Drag,
    last_click: (f64, f32, f32),
    scroll_px: f32,
    vol_shown: f64,
    cursor_visible: bool,
}

fn white(a: f32) -> Rgba {
    [1.0, 1.0, 1.0, a]
}

impl Ui {
    pub fn new(now: f64) -> Self {
        Self {
            mouse: (0.0, 0.0),
            mouse_in: false,
            last_move: now,
            bar_alpha: 0.0,
            drag: Drag::None,
            last_click: (f64::NEG_INFINITY, 0.0, 0.0),
            scroll_px: 0.0,
            vol_shown: f64::NEG_INFINITY,
            cursor_visible: true,
        }
    }

    fn in_zone(&self, h: f32) -> bool {
        self.mouse_in && self.mouse.1 > h - ZONE
    }

    fn time_at(x: f32, w: f32, duration: f64) -> f64 {
        ((x - PAD) / (w - 2.0 * PAD)).clamp(0.0, 1.0) as f64 * duration
    }

    /// Cursor position in logical pixels.
    pub fn cursor_moved(&mut self, x: f32, y: f32, now: f64, window: &Window) {
        self.mouse = (x, y);
        self.mouse_in = true;
        self.last_move = now;
        if let Drag::Pending { x: px, y: py } = self.drag {
            if (x - px).hypot(y - py) > DRAG_START {
                // Dragging out of fullscreen: back to a window first, as a
                // double-click would, then move it.
                if window.fullscreen().is_some() {
                    window.set_fullscreen(None);
                }
                // Hand the move to the window manager (works on Wayland too).
                // It keeps the button, so forget it here.
                self.drag = Drag::None;
                let _ = window.drag_window();
            }
        }
    }

    pub fn cursor_left(&mut self) {
        self.mouse_in = false;
    }

    pub fn button(&mut self, button: MouseButton, pressed: bool, now: f64, window: &Window, player: &mut Player, size: (f32, f32)) {
        let (w, h) = size;
        match (button, pressed) {
            (MouseButton::Right, true) => player.toggle_pause(),
            (MouseButton::Left, true) => {
                let (x, y) = self.mouse;
                if self.in_zone(h) && self.bar_alpha > 0.3 {
                    self.drag = Drag::Seek;
                    player.scrub(Self::time_at(x, w, player.duration()));
                } else if now - self.last_click.0 < DOUBLE_CLICK && (x - self.last_click.1).hypot(y - self.last_click.2) < 6.0 {
                    let full = window.fullscreen().is_some();
                    window.set_fullscreen(if full { None } else { Some(Fullscreen::Borderless(None)) });
                    window.focus_window();
                    self.last_click.0 = f64::NEG_INFINITY;
                    self.drag = Drag::None;
                } else {
                    self.last_click = (now, x, y);
                    self.drag = Drag::Pending { x, y };
                }
            }
            (MouseButton::Left, false) => {
                if self.drag == Drag::Seek {
                    player.end_scrub();
                }
                self.drag = Drag::None;
            }
            _ => {}
        }
    }

    /// Mouse wheel: `lines` for notched wheels, `pixels` for touchpads.
    pub fn wheel(&mut self, lines: f32, pixels: f32, now: f64, player: &Player) {
        let mut steps = if lines > 0.0 { 1 } else if lines < 0.0 { -1 } else { 0 };
        if pixels != 0.0 {
            self.scroll_px += pixels;
            steps = (self.scroll_px / SCROLL_STEP_PX) as i32;
            self.scroll_px -= steps as f32 * SCROLL_STEP_PX;
        }
        if steps != 0 {
            player.set_volume(player.volume() + 0.05 * steps as f32);
            crate::settings::save_volume(player.volume());
            self.vol_shown = now;
        }
    }

    /// Per-frame bookkeeping: timeline fade, cursor hiding, live seeking.
    pub fn update(&mut self, now: f64, dt: f32, window: &Window, player: &mut Player, size: (f32, f32)) {
        let (w, _) = size;
        let recent = now - self.last_move < IDLE;
        let seeking = self.drag == Drag::Seek;
        // The timeline shows whenever the cursor is visible over the window.
        let target = if (self.mouse_in && recent) || seeking { 1.0 } else { 0.0 };
        self.bar_alpha += (target - self.bar_alpha) * (dt * 12.0).min(1.0);

        let want_cursor = recent || self.drag != Drag::None || !self.mouse_in;
        if want_cursor != self.cursor_visible {
            window.set_cursor_visible(want_cursor);
            self.cursor_visible = want_cursor;
        }
        if seeking {
            player.scrub(Self::time_at(self.mouse.0, w, player.duration()));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        ui: &mut Batch,
        texts: &mut Vec<Text>,
        measure: &mut dyn FnMut(&str, f32) -> f32,
        player: &Player,
        waveform: &[(f32, f32)],
        expected: usize,
        now: f64,
        size: (f32, f32),
    ) {
        let (w, h) = size;
        let mut text = |s: String, x: f32, y: f32, size: f32, color: Rgba| texts.push(Text { text: s, x, y, size, color });

        let a = self.bar_alpha;
        if a > 0.01 {
            // Soft shade behind the controls.
            // One smooth gradient (stacked strips overlapped into dark lines).
            ui.gradient_v(0.0, h - ZONE, w, ZONE, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.6 * a]);

            let dur = player.duration();
            let cur = player.position() / player.rate as f64;
            let mid = h - 12.0 - WAVE_H / 2.0;
            let bar_w = w - 2.0 * PAD;
            let played_x = PAD + bar_w * (cur / dur) as f32;

            // Waveform: one 2 px column every 3 px, mirrored around the centre line.
            // Bright = played, dim = still to come; outer part = peak, inner = loudness.
            let step = 3.0;
            let cols = (bar_w / step) as usize;
            // `expected` buckets make up the whole song; while it is still being
            // decoded, only the first `waveform.len()` exist yet (it fills in).
            let n = expected.max(waveform.len()).max(1);
            for c in 0..cols {
                let x = PAD + c as f32 * step;
                let b0 = ((x - PAD) / bar_w * n as f32) as usize;
                if b0 >= waveform.len() {
                    break;
                }
                let b1 = (((x + step - PAD) / bar_w * n as f32) as usize).clamp(b0 + 1, waveform.len());
                let (peak, rms) = waveform[b0..b1].iter().fold((0.0f32, 0.0f32), |acc, v| (acc.0.max(v.0), acc.1.max(v.1)));
                let played = x + 1.0 < played_x;
                let (pa, ra) = if played { (0.35, 0.9) } else { (0.12, 0.3) };
                let ph = (peak * WAVE_H / 2.0).max(0.5);
                let rh = (rms * WAVE_H / 2.0 * 1.4).min(ph).max(0.5);
                ui.rect(x, mid - ph, 2.0, ph * 2.0, white(pa * a));
                ui.rect(x, mid - rh, 2.0, rh * 2.0, white(ra * a));
            }
            ui.rect(played_x - 1.0, mid - WAVE_H / 2.0 - 3.0, 2.0, WAVE_H + 6.0, white(0.95 * a));

            let label_y = mid - WAVE_H / 2.0 - 30.0;
            text(format!("{} / {}", fmt_time(cur), fmt_time(dur)), PAD, label_y, 15.0, white(0.8 * a));
            if player.paused() {
                let tw = measure("paused", 15.0);
                text("paused".into(), w - PAD - tw, label_y, 15.0, white(0.5 * a));
            }

            if self.in_zone(h) && self.drag != Drag::Seek {
                let x = self.mouse.0.clamp(PAD, w - PAD);
                ui.rect(x - 0.5, mid - WAVE_H / 2.0, 1.0, WAVE_H, white(0.6 * a));
                let tip = fmt_time(Self::time_at(x, w, dur));
                let tw = measure(&tip, 14.0);
                let tx = (x - tw / 2.0).min(w - PAD - tw - 60.0).max(PAD + 90.0);
                text(tip, tx, label_y + 1.0, 14.0, white(0.9 * a));
            }
        }

        let fade = |since: f64| (1.0 - ((now - since - 1.2) / 0.5) as f32).clamp(0.0, 1.0);
        let va = fade(self.vol_shown);
        if va > 0.0 {
            let v = player.volume();
            let label = format!("Volume {:.0}%", v * 100.0);
            let tw = measure(&label, 15.0);
            text(label, w - PAD - tw, PAD - 4.0, 15.0, white(0.85 * va));
            ui.rect(w - PAD - 100.0, PAD + 20.0, 100.0, 2.0, white(0.2 * va));
            ui.rect(w - PAD - 100.0, PAD + 20.0, 100.0 * v, 2.0, white(0.85 * va));
        }
    }
}

fn fmt_time(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}
