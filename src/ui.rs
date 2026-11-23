//! Minimal controls:
//!   right-click / Space pause / resume
//!   wheel               volume (over the playlist: scroll it)
//!   left-drag           move the window (from fullscreen: leaves
//!                       fullscreen, then moves the window)
//!   double left-click   fullscreen on/off (and bring to front), or F11
//!   waveform timeline   at the bottom while the cursor is visible over the
//!                       window (click or drag on it to seek; live input
//!                       has none, just its name)
//!   playlist            playing a folder: at the right edge (or L pins it);
//!                       click a song to play it, or the order to change it

use crate::audio::Player;
use crate::capture::Capture;
use crate::gfx::{Batch, Rgba, Text};
use winit::event::MouseButton;
use winit::window::{Fullscreen, Window};

const ZONE: f32 = 96.0;
/// Height of the dark shade behind the controls (taller than `ZONE`, so it
/// also covers the title).
const SHADE: f32 = 128.0;
const IDLE: f64 = 2.0;
const PAD: f32 = 16.0;
const WAVE_H: f32 = 36.0;
const DOUBLE_CLICK: f64 = 0.35;
/// The cursor must move this far with the button held before the window moves.
const DRAG_START: f32 = 4.0;
/// Touchpad scrolling: pixels per volume step.
const SCROLL_STEP_PX: f32 = 40.0;

/// Playlist panel: width of the list, the shade behind it, the edge that
/// opens it, and its rows.
const PANEL_W: f32 = 340.0;
const PANEL_SHADE: f32 = 560.0;
const PANEL_EDGE: f32 = 48.0;
const PANEL_TOP: f32 = 32.0;
const ROWS_TOP: f32 = PANEL_TOP + 52.0;
const ROW_H: f32 = 36.0;

/// What plays: a song (`Player`) or live input (`Capture`).
pub trait Transport {
    fn paused(&self) -> bool;
    fn toggle_pause(&mut self);
    /// Length in seconds; `None` for live input (no seeking).
    fn duration(&self) -> Option<f64>;
    fn scrub(&mut self, _secs: f64) {}
    fn end_scrub(&mut self) {}
    /// `None`: no volume of its own (live input is heard elsewhere).
    fn volume(&self) -> Option<f32>;
    fn set_volume(&self, _v: f32) {}
}

impl Transport for Player {
    fn paused(&self) -> bool {
        Player::paused(self)
    }
    fn toggle_pause(&mut self) {
        Player::toggle_pause(self)
    }
    fn duration(&self) -> Option<f64> {
        Some(Player::duration(self))
    }
    fn scrub(&mut self, secs: f64) {
        Player::scrub(self, secs)
    }
    fn end_scrub(&mut self) {
        Player::end_scrub(self)
    }
    fn volume(&self) -> Option<f32> {
        Some(Player::volume(self))
    }
    fn set_volume(&self, v: f32) {
        Player::set_volume(self, v)
    }
}

impl Transport for Capture {
    fn paused(&self) -> bool {
        Capture::paused(self)
    }
    fn toggle_pause(&mut self) {
        Capture::toggle_pause(self)
    }
    fn duration(&self) -> Option<f64> {
        None
    }
    fn volume(&self) -> Option<f32> {
        None
    }
}

/// A folder being played, for the playlist panel.
pub struct PlaylistView<'a> {
    pub name: &'a str,
    pub songs: &'a [String],
    pub lengths: &'a [Option<f64>],
    pub current: usize,
    /// The order, as shown ("by name", …).
    pub sort: &'a str,
}

/// What the controls show this frame.
pub struct Status<'a> {
    pub title: &'a str,
    /// Dimmer, after the title (folder position, or the source's kind).
    pub subtitle: &'a str,
    pub paused: bool,
    /// (position, length) in seconds; `None` for live input.
    pub time: Option<(f64, f64)>,
    /// The overview: (peak, rms) buckets, and how many make up the whole
    /// song (it fills from the left as it decodes). Live input has none.
    pub waveform: &'a [(f32, f32)],
    pub expected: usize,
    pub volume: Option<f32>,
    pub playlist: Option<PlaylistView<'a>>,
}

/// Something the controls ask for that the app carries out.
pub enum Action {
    /// Play this song of the playlist.
    Play(usize),
    /// Play the folder in its next order.
    CycleSort,
}

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
    /// The timeline and playlist are available (not while the start screen is up).
    pub controls: bool,
    /// Songs in the playlist (0: none) and the one playing.
    songs: usize,
    current: usize,
    panel_alpha: f32,
    /// L: keep the playlist open.
    pub panel_pinned: bool,
    /// First row shown.
    panel_scroll: usize,
}

fn white(a: f32) -> Rgba {
    [1.0, 1.0, 1.0, a]
}

/// `s` shortened with "…" to fit `max_w`.
pub fn fit(s: &str, size: f32, max_w: f32, measure: &mut dyn FnMut(&str, f32) -> f32) -> String {
    if measure(s, size) <= max_w {
        return s.to_string();
    }
    let mut t = s.to_string();
    while !t.is_empty() && measure(&format!("{t}…"), size) > max_w {
        t.pop();
    }
    format!("{}…", t.trim_end())
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
            controls: false,
            songs: 0,
            current: 0,
            panel_alpha: 0.0,
            panel_pinned: false,
            panel_scroll: 0,
        }
    }

    pub fn mouse(&self) -> (f32, f32) {
        self.mouse
    }

    fn in_zone(&self, h: f32) -> bool {
        self.mouse_in && self.mouse.1 > h - ZONE
    }

    fn time_at(x: f32, w: f32, duration: f64) -> f64 {
        ((x - PAD) / (w - 2.0 * PAD)).clamp(0.0, 1.0) as f64 * duration
    }

    fn panel_x(w: f32) -> f32 {
        w - PAD - PANEL_W
    }

    /// Rows of the playlist that fit above the timeline.
    fn panel_rows(h: f32) -> usize {
        (((h - SHADE - ROWS_TOP) / ROW_H).floor().max(1.0)) as usize
    }

    /// The cursor is over the open playlist.
    fn over_panel(&self, w: f32) -> bool {
        self.songs > 0 && self.panel_alpha > 0.5 && self.mouse_in && self.mouse.0 > Self::panel_x(w) - 24.0
    }

    /// The cursor is on the playlist's order ("by name", …), right of the song count.
    fn on_sort(&self, w: f32) -> bool {
        let (x, y) = self.mouse;
        self.over_panel(w) && x > Self::panel_x(w) + PANEL_W / 2.0 && (PANEL_TOP + 20.0..PANEL_TOP + 46.0).contains(&y)
    }

    /// The playlist row under the cursor.
    fn panel_hit(&self, size: (f32, f32)) -> Option<usize> {
        let (w, h) = size;
        let (x, y) = self.mouse;
        if !self.over_panel(w) || x < Self::panel_x(w) || y < ROWS_TOP {
            return None;
        }
        let row = ((y - ROWS_TOP) / ROW_H) as usize;
        (row < Self::panel_rows(h)).then_some(self.panel_scroll + row).filter(|&i| i < self.songs)
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

    /// A mouse button. Window moves and fullscreen work everywhere; seeking,
    /// pausing and the playlist only with `transport` (something playing,
    /// start screen closed).
    pub fn button(&mut self, button: MouseButton, pressed: bool, now: f64, window: &Window, transport: Option<&mut dyn Transport>, size: (f32, f32)) -> Option<Action> {
        let (w, h) = size;
        let controls = self.controls;
        let mut transport = transport.filter(|_| controls);
        match (button, pressed) {
            (MouseButton::Right, true) => {
                if let Some(t) = transport {
                    t.toggle_pause();
                }
            }
            (MouseButton::Left, true) => {
                let (x, y) = self.mouse;
                let duration = transport.as_ref().and_then(|t| t.duration());
                if let Some(i) = self.panel_hit(size).filter(|_| controls) {
                    return Some(Action::Play(i));
                } else if controls && self.on_sort(w) {
                    return Some(Action::CycleSort);
                } else if controls && self.over_panel(w) {
                    // Inside the playlist but not on a song: nothing.
                } else if let (Some(t), Some(d), true) = (transport.as_mut(), duration, self.in_zone(h) && self.bar_alpha > 0.3) {
                    self.drag = Drag::Seek;
                    t.scrub(Self::time_at(x, w, d));
                } else if now - self.last_click.0 < DOUBLE_CLICK && (x - self.last_click.1).hypot(y - self.last_click.2) < 6.0 {
                    toggle_fullscreen(window);
                    self.last_click.0 = f64::NEG_INFINITY;
                    self.drag = Drag::None;
                } else {
                    self.last_click = (now, x, y);
                    self.drag = Drag::Pending { x, y };
                }
            }
            (MouseButton::Left, false) => {
                if self.drag == Drag::Seek {
                    if let Some(t) = transport {
                        t.end_scrub();
                    }
                }
                self.drag = Drag::None;
            }
            _ => {}
        }
        None
    }

    /// Mouse wheel: `lines` for notched wheels, `pixels` for touchpads.
    pub fn wheel(&mut self, lines: f32, pixels: f32, now: f64, transport: Option<&dyn Transport>, size: (f32, f32)) {
        let mut steps = if lines > 0.0 { 1 } else if lines < 0.0 { -1 } else { 0 };
        if pixels != 0.0 {
            self.scroll_px += pixels;
            steps = (self.scroll_px / SCROLL_STEP_PX) as i32;
            self.scroll_px -= steps as f32 * SCROLL_STEP_PX;
        }
        if steps == 0 || !self.controls {
            return;
        }
        if self.over_panel(size.0) {
            let max = self.songs.saturating_sub(Self::panel_rows(size.1));
            self.panel_scroll = (self.panel_scroll as i32 - steps).clamp(0, max as i32) as usize;
            return;
        }
        if let Some((t, v)) = transport.and_then(|t| Some((t, t.volume()?))) {
            t.set_volume(v + 0.05 * steps as f32);
            crate::settings::save_volume(t.volume().unwrap_or(v));
            self.vol_shown = now;
        }
    }

    /// Per-frame bookkeeping: timeline and playlist fade, cursor hiding,
    /// live seeking. `songs`/`current`: the playlist (0 songs: none).
    pub fn update(&mut self, now: f64, dt: f32, window: &Window, transport: Option<&mut dyn Transport>, size: (f32, f32), songs: usize, current: usize) {
        let (w, h) = size;
        let recent = now - self.last_move < IDLE;
        let seeking = self.drag == Drag::Seek;
        let over_panel = self.over_panel(w);
        // The timeline shows whenever the cursor is visible over the window.
        let target = if self.controls && ((self.mouse_in && recent) || seeking || over_panel) { 1.0 } else { 0.0 };
        self.bar_alpha += (target - self.bar_alpha) * (dt * 12.0).min(1.0);

        // The playlist: at the right edge, while over it, or pinned.
        if (songs, current) != (self.songs, self.current) || self.panel_alpha < 0.02 {
            // Opening (or the song changed): show the current song.
            let rows = Self::panel_rows(h);
            if current < self.panel_scroll || current >= self.panel_scroll + rows || self.panel_alpha < 0.02 {
                self.panel_scroll = current.saturating_sub(rows / 3).min(songs.saturating_sub(rows));
            }
        }
        self.songs = songs;
        self.current = current;
        let at_edge = self.mouse_in && recent && self.mouse.0 > w - PANEL_EDGE && self.mouse.1 < h - ZONE;
        let show = self.controls && songs > 0 && (self.panel_pinned || at_edge || over_panel);
        let target = if show { 1.0 } else { 0.0 };
        self.panel_alpha += (target - self.panel_alpha) * (dt * 12.0).min(1.0);

        let want_cursor = !self.controls || recent || over_panel || self.drag != Drag::None || !self.mouse_in;
        if want_cursor != self.cursor_visible {
            window.set_cursor_visible(want_cursor);
            self.cursor_visible = want_cursor;
        }
        if seeking {
            if let Some(t) = transport {
                if let Some(d) = t.duration() {
                    t.scrub(Self::time_at(self.mouse.0, w, d));
                }
            }
        }
    }

    pub fn draw(&self, ui: &mut Batch, texts: &mut Vec<Text>, measure: &mut dyn FnMut(&str, f32) -> f32, s: &Status, now: f64, size: (f32, f32)) {
        let (w, h) = size;
        let mut text = |t: String, x: f32, y: f32, size: f32, color: Rgba| texts.push(Text { text: t, x, y, size, color });

        if let Some(pl) = s.playlist.as_ref().filter(|_| self.panel_alpha > 0.01) {
            self.draw_playlist(ui, &mut text, measure, pl, size);
        }

        let a = self.bar_alpha;
        if a > 0.01 {
            // Soft shade behind the controls (lower for live input: no timeline).
            let shade = if s.time.is_some() { SHADE } else { SHADE * 0.6 };
            ui.fade_v(0.0, h - shade, w, shade, [0.0, 0.0, 0.0, 0.6 * a]);

            let mid = h - 12.0 - WAVE_H / 2.0;
            let bar_w = w - 2.0 * PAD;
            let played_x = s.time.map(|(cur, dur)| PAD + bar_w * (cur / dur.max(1e-9)) as f32);

            // Waveform: one 2 px column every 3 px, mirrored around the centre line.
            // Bright = played, dim = still to come; outer part = peak, inner = loudness.
            let step = 3.0;
            let cols = (bar_w / step) as usize;
            // `expected` buckets make up the whole song; while it is still being
            // decoded, only the first `waveform.len()` exist yet (it fills in).
            let n = s.expected.max(s.waveform.len()).max(1);
            for c in (0..cols).filter(|_| played_x.is_some()) {
                let x = PAD + c as f32 * step;
                let b0 = ((x - PAD) / bar_w * n as f32) as usize;
                if b0 >= s.waveform.len() {
                    break;
                }
                let b1 = (((x + step - PAD) / bar_w * n as f32) as usize).clamp(b0 + 1, s.waveform.len());
                let (peak, rms) = s.waveform[b0..b1].iter().fold((0.0f32, 0.0f32), |acc, v| (acc.0.max(v.0), acc.1.max(v.1)));
                let played = played_x.is_some_and(|px| x + 1.0 < px);
                let (pa, ra) = if played { (0.35, 0.9) } else { (0.12, 0.3) };
                let ph = (peak * WAVE_H / 2.0).max(0.5);
                let rh = (rms * WAVE_H / 2.0 * 1.4).min(ph).max(0.5);
                ui.rect(x, mid - ph, 2.0, ph * 2.0, white(pa * a));
                ui.rect(x, mid - rh, 2.0, rh * 2.0, white(ra * a));
            }
            if let Some(px) = played_x {
                ui.rect(px - 1.0, mid - WAVE_H / 2.0 - 3.0, 2.0, WAVE_H + 6.0, white(0.95 * a));
            }

            // One line: title (and subtitle) on the left; time or "live" on
            // the right, with a pause mark beside it while paused.
            // Above the timeline; live input (no timeline) sits at the bottom.
            let title_y = if s.time.is_some() { h - 88.0 } else { h - PAD - 18.0 * 1.3 - 4.0 };
            let small_y = title_y + 3.5;
            let right = match s.time {
                Some((cur, dur)) => format!("{} / {}", fmt_time(cur), fmt_time(dur)),
                None => "live".into(),
            };
            let rw = measure(&right, 15.0);
            let rx = w - PAD - rw;
            text(right, rx, small_y, 15.0, white(0.8 * a));
            let mark_cy = small_y + 15.0 * 1.3 / 2.0;
            let mut left_of_right = rx;
            if s.paused {
                let bx = rx - 10.0 - 7.0;
                ui.rect(bx, mark_cy - 6.0, 2.0, 12.0, white(0.8 * a));
                ui.rect(bx + 5.0, mark_cy - 6.0, 2.0, 12.0, white(0.8 * a));
                left_of_right = bx;
            } else if s.time.is_none() {
                // Live and running: a softly pulsing dot.
                let pulse = 0.35 + 0.65 * (0.5 + 0.5 * (now * std::f64::consts::PI).sin() as f32);
                ui.glow(rx - 12.0, mark_cy, 6.0, 6.0, white(pulse * a));
                left_of_right = rx - 18.0;
            }
            let room = left_of_right - PAD - 24.0;
            let tw = measure(s.title, 18.0);
            let sw = if s.subtitle.is_empty() { 0.0 } else { measure(s.subtitle, 15.0) };
            if !s.subtitle.is_empty() && tw + 12.0 + sw <= room {
                text(s.title.to_string(), PAD, title_y, 18.0, white(0.92 * a));
                text(s.subtitle.to_string(), PAD + tw + 12.0, small_y, 15.0, white(0.5 * a));
            } else {
                text(fit(s.title, 18.0, room, measure), PAD, title_y, 18.0, white(0.92 * a));
            }

            // Where a click would seek to, just above the waveform.
            if let Some((_, dur)) = s.time.filter(|_| self.in_zone(h) && self.drag != Drag::Seek && !self.over_panel(w)) {
                let x = self.mouse.0.clamp(PAD, w - PAD);
                ui.rect(x - 0.5, mid - WAVE_H / 2.0, 1.0, WAVE_H, white(0.6 * a));
                let tip = fmt_time(Self::time_at(x, w, dur));
                let tw = measure(&tip, 13.0);
                let tx = (x - tw / 2.0).clamp(PAD, w - PAD - tw);
                text(tip, tx, mid - WAVE_H / 2.0 - 19.0, 13.0, white(0.9 * a));
            }
        }

        let fade = |since: f64| (1.0 - ((now - since - 1.2) / 0.5) as f32).clamp(0.0, 1.0);
        let va = fade(self.vol_shown);
        if let Some(v) = s.volume.filter(|_| va > 0.0) {
            let label = format!("Volume {:.0}%", v * 100.0);
            let tw = measure(&label, 15.0);
            text(label, w - PAD - tw, PAD - 4.0, 15.0, white(0.85 * va));
            ui.rect(w - PAD - 100.0, PAD + 20.0, 100.0, 2.0, white(0.2 * va));
            ui.rect(w - PAD - 100.0, PAD + 20.0, 100.0 * v, 2.0, white(0.85 * va));
        }
    }

    fn draw_playlist(&self, ui: &mut Batch, text: &mut dyn FnMut(String, f32, f32, f32, Rgba), measure: &mut dyn FnMut(&str, f32) -> f32, pl: &PlaylistView, size: (f32, f32)) {
        let (w, h) = size;
        let a = self.panel_alpha;
        // Shade from the right edge, like the one behind the timeline.
        let solid = PANEL_SHADE * 0.55;
        ui.fade_h(w - PANEL_SHADE, 0.0, PANEL_SHADE - solid, h, [0.0, 0.0, 0.0, 0.62 * a]);
        ui.rect(w - solid, 0.0, solid, h, [0.0, 0.0, 0.0, 0.62 * a]);

        let x0 = Self::panel_x(w);
        let inner = x0 + 18.0;
        text(fit(pl.name, 18.0, PANEL_W - 18.0, measure), inner, PANEL_TOP, 18.0, white(0.92 * a));
        let count = format!("{} song{}", pl.songs.len(), if pl.songs.len() == 1 { "" } else { "s" });
        let sub = if pl.lengths.iter().all(Option::is_some) {
            format!("{count} · {}", fmt_time(pl.lengths.iter().flatten().sum()))
        } else {
            count
        };
        text(sub, inner, PANEL_TOP + 26.0, 13.0, white(0.45 * a));
        // The order, on the right of the same line: click to change it.
        let sw = measure(pl.sort, 13.0);
        let sa = if self.on_sort(w) { 0.95 } else { 0.45 };
        text(pl.sort.to_string(), x0 + PANEL_W - sw, PANEL_TOP + 26.0, 13.0, white(sa * a));

        let hover = self.panel_hit(size);
        let rows = Self::panel_rows(h);
        for (row, i) in (self.panel_scroll..pl.songs.len()).take(rows).enumerate() {
            let y = ROWS_TOP + row as f32 * ROW_H;
            let cur = i == pl.current;
            let lit = cur || hover == Some(i);
            let ta = if lit { 0.95 } else if i < pl.current { 0.4 } else { 0.55 };
            let (na, da) = if cur { (0.7, 0.7) } else { (0.32, 0.38) };
            if cur {
                ui.rect(x0, y + 7.0, 2.0, 22.0, white(0.95 * a));
            }
            let ty = y + (ROW_H - 15.0 * 1.3) / 2.0;
            let sy = y + (ROW_H - 13.0 * 1.3) / 2.0;
            let n = format!("{}", i + 1);
            text(n.clone(), inner + 20.0 - measure(&n, 13.0), sy, 13.0, white(na * a));
            let len = pl.lengths.get(i).copied().flatten().map(fmt_time).unwrap_or_default();
            let lw = if len.is_empty() { 0.0 } else { measure(&len, 13.0) };
            if !len.is_empty() {
                text(len, x0 + PANEL_W - lw, sy, 13.0, white(da * a));
            }
            let tx = inner + 34.0;
            let title = fit(&pl.songs[i], 15.0, x0 + PANEL_W - lw - 14.0 - tx, measure);
            text(title, tx, ty, 15.0, white(ta * a));
        }
    }
}

/// Fullscreen on/off, and bring the window to the front.
pub fn toggle_fullscreen(window: &Window) {
    let full = window.fullscreen().is_some();
    window.set_fullscreen(if full { None } else { Some(Fullscreen::Borderless(None)) });
    window.focus_window();
}

pub fn fmt_time(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}
