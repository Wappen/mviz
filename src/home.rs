//! The start screen: shown when mviz starts without a file, and over the
//! picture when Tab is pressed. Open a file or a folder, capture a source
//! (what's playing, or an input), or reopen the last file or folder.
//!
//! The rows look like the rest of the controls: white text at a few
//! strengths, and the 2 px bar of the timeline's playhead beside the row
//! under the cursor. Resting on a capture source shows its input level.

use crate::capture::{self, Capture, SourceInfo};
use crate::gfx::{Batch, Rgba, Text};
use crate::ui::fit;
use std::path::PathBuf;
use std::sync::mpsc;

const COL_W: f32 = 560.0;
const ROW_H: f32 = 44.0;
const GAP: f32 = 36.0;
const TITLE: f32 = 40.0;
/// Rest this long on a capture source before listening to it (level meter).
const PREVIEW_AFTER: f64 = 0.3;
const METER_BARS: usize = 24;

/// What the person picked.
pub enum Choice {
    OpenFile,
    OpenFolder,
    Capture(SourceInfo),
    Recent(PathBuf),
}

#[derive(Clone, Copy, PartialEq)]
enum Row {
    OpenFile,
    OpenFolder,
    Source(usize),
    Recent,
}

pub struct Home {
    sources: Vec<SourceInfo>,
    listing: Option<mpsc::Receiver<Vec<SourceInfo>>>,
    recent: Option<PathBuf>,
    /// The highlighted row (cursor or arrow keys), and since when.
    selected: Option<Row>,
    since: f64,
    /// Listening to the highlighted source (for its level meter).
    preview: Option<Capture>,
    /// A source that could not be opened (not retried while highlighted).
    failed: Option<String>,
    /// Rows as last laid out: (row, top y), and the column's left edge.
    rows: Vec<(Row, f32)>,
    x0: f32,
}

impl Home {
    pub fn new() -> Self {
        let mut home = Self {
            sources: Vec::new(),
            listing: None,
            recent: None,
            selected: None,
            since: 0.0,
            preview: None,
            failed: None,
            rows: Vec::new(),
            x0: 0.0,
        };
        home.refresh();
        home
    }

    /// Reads the sources (in the background) and the recent entry again.
    pub fn refresh(&mut self) {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(capture::list());
        });
        self.listing = Some(rx);
        self.recent = crate::settings::load_recent();
        self.failed = None;
    }

    /// Stops listening (the screen is closing).
    pub fn close(&mut self) {
        self.preview = None;
        self.selected = None;
    }

    /// The running preview of `source`, to keep using it instead of opening
    /// the source again.
    pub fn take_preview(&mut self, source: &SourceInfo) -> Option<Capture> {
        self.preview.take_if(|c| c.source.id == source.id)
    }

    fn select(&mut self, row: Option<Row>, now: f64) {
        if row != self.selected {
            self.selected = row;
            self.since = now;
        }
    }

    fn choice(&self, row: Row) -> Option<Choice> {
        Some(match row {
            Row::OpenFile => Choice::OpenFile,
            Row::OpenFolder => Choice::OpenFolder,
            Row::Source(i) => Choice::Capture(self.sources.get(i)?.clone()),
            Row::Recent => Choice::Recent(self.recent.clone()?),
        })
    }

    fn row_at(&self, x: f32, y: f32) -> Option<Row> {
        if x < self.x0 - 8.0 || x > self.x0 + COL_W + 8.0 {
            return None;
        }
        self.rows.iter().find(|(_, top)| y >= *top && y < top + ROW_H).map(|(r, _)| *r)
    }

    /// The cursor moved (logical pixels).
    pub fn hover(&mut self, x: f32, y: f32, now: f64) {
        if let Some(row) = self.row_at(x, y) {
            self.select(Some(row), now);
        }
    }

    /// A left click; a row picks it.
    pub fn click(&mut self, x: f32, y: f32) -> Option<Choice> {
        self.row_at(x, y).and_then(|r| self.choice(r))
    }

    /// Up / down move the highlight; Enter picks it.
    pub fn key(&mut self, key: &winit::keyboard::Key, now: f64) -> Option<Choice> {
        use winit::keyboard::{Key, NamedKey};
        let order: Vec<Row> = self.rows.iter().map(|(r, _)| *r).collect();
        if order.is_empty() {
            return None;
        }
        let at = self.selected.and_then(|s| order.iter().position(|r| *r == s));
        match key {
            Key::Named(NamedKey::ArrowDown) => {
                let i = at.map_or(0, |i| (i + 1).min(order.len() - 1));
                self.select(Some(order[i]), now);
            }
            Key::Named(NamedKey::ArrowUp) => {
                let i = at.map_or(0, |i| i.saturating_sub(1));
                self.select(Some(order[i]), now);
            }
            Key::Named(NamedKey::Enter) => return self.selected.and_then(|r| self.choice(r)),
            _ => {}
        }
        None
    }

    /// Per frame: picks up the source list, and listens to the highlighted
    /// source once the highlight has rested on it.
    pub fn update(&mut self, now: f64) {
        if let Some(list) = self.listing.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.sources = list;
            self.listing = None;
            // Highlight the source used last time.
            if self.selected.is_none() {
                let last = crate::settings::load_capture();
                if let Some(i) = self.sources.iter().position(|s| Some(&s.id) == last.as_ref()) {
                    self.select(Some(Row::Source(i)), now);
                }
            }
            // Keep the preview only if its source is still there.
            if let Some(p) = &self.preview {
                if !self.sources.contains(&p.source) {
                    self.preview = None;
                }
            }
        }
        let source = match self.selected {
            Some(Row::Source(i)) => self.sources.get(i),
            _ => None,
        };
        match source {
            None => self.preview = None,
            Some(s) if self.preview.as_ref().is_some_and(|p| p.source.id == s.id) => {}
            Some(s) if now - self.since >= PREVIEW_AFTER && self.failed.as_ref() != Some(&s.id) => {
                self.preview = None;
                match Capture::start(s) {
                    Ok(c) => self.preview = Some(c),
                    Err(e) => {
                        eprintln!("mviz: {e}");
                        self.failed = Some(s.id.clone());
                    }
                }
            }
            Some(_) => {}
        }
    }

    /// Draws the screen at strength `a`. `title`: "mviz", or what is
    /// playing. `glows`: no picture behind (started without a file): soft
    /// drifting light, into the phosphor layer. `overlay`: over the picture.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        ui: &mut Batch,
        texts: &mut Vec<Text>,
        glows: Option<&mut Batch>,
        measure: &mut dyn FnMut(&str, f32) -> f32,
        title: &str,
        overlay: bool,
        a: f32,
        now: f64,
        size: (f32, f32),
    ) {
        let (w, h) = size;
        let white = |al: f32| -> Rgba { [1.0, 1.0, 1.0, al * a] };
        let mut text = |t: String, x: f32, y: f32, size: f32, color: Rgba| texts.push(Text { text: t, x, y, size, color });

        if let Some(g) = glows {
            let t = now as f32;
            let drift = |phase: f32| ((t / 9.0 + phase).sin() * 40.0, (t / 9.0 + phase).cos() * -24.0);
            let lights: [(f32, f32, f32, f32, [f32; 3], f32); 4] = [
                (0.30, 0.72, 0.17, 0.08, [0.47, 0.24, 0.78], 0.0),
                (0.70, 0.23, 0.12, 0.06, [0.16, 0.67, 0.55], 2.0),
                (0.79, 0.75, 0.10, 0.05, [0.86, 0.43, 0.24], 4.0),
                (0.31, 0.19, 0.08, 0.04, [0.82, 0.27, 0.55], 1.0),
            ];
            for (cx, cy, rx, ry, c, phase) in lights {
                let (dx, dy) = drift(phase);
                g.glow(cx * w + dx, cy * h + dy, rx * w, ry * h * 1.6, [c[0], c[1], c[2], 0.22]);
            }
        }

        // Lay out: title, open rows, capture, recent.
        let has_recent = self.recent.is_some();
        let n_sources = self.sources.len().max(1);
        let label = 12.0 * 1.3 + 6.0;
        let total = TITLE * 1.3 + GAP + 2.0 * ROW_H + GAP + label + n_sources as f32 * ROW_H + if has_recent { GAP + label + ROW_H } else { 0.0 };
        let mut y = ((h - total) / 2.0 - 30.0).clamp(16.0, 92.0);
        let x0 = ((w - COL_W) / 2.0).max(16.0);
        let col_w = COL_W.min(w - 32.0);
        self.x0 = x0;
        self.rows.clear();

        let tw = measure(title, TITLE);
        let shown = fit(title, TITLE, col_w, measure);
        text(shown, x0 + ((col_w - tw).max(0.0)) / 2.0, y, TITLE, white(0.95));
        y += TITLE * 1.3 + GAP;

        self.rows.push((Row::OpenFile, y));
        self.rows.push((Row::OpenFolder, y + ROW_H));
        y += 2.0 * ROW_H + GAP;

        let spaced = |s: &str| s.chars().map(String::from).collect::<Vec<_>>().join("\u{2009}");
        text(spaced("CAPTURE"), x0 + 20.0, y, 12.0, white(0.4));
        y += label;
        let capture_top = y;
        for i in 0..self.sources.len() {
            self.rows.push((Row::Source(i), y + i as f32 * ROW_H));
        }
        y += n_sources as f32 * ROW_H;
        if has_recent {
            y += GAP;
            text(spaced("RECENT"), x0 + 20.0, y, 12.0, white(0.4));
            y += label;
            self.rows.push((Row::Recent, y));
        }

        for &(row, top) in &self.rows {
            let lit = self.selected == Some(row);
            if lit {
                ui.rect(x0, top + 10.0, 2.0, 24.0, white(0.95));
            }
            let ty = top + (ROW_H - 16.0 * 1.3) / 2.0;
            let sy = top + (ROW_H - 14.0 * 1.3) / 2.0;
            let ky = top + (ROW_H - 13.0 * 1.3) / 2.0;
            let name_a = if lit { 0.95 } else { 0.55 };
            let right = x0 + col_w - 16.0;
            let (name, sub, key) = match row {
                Row::OpenFile => ("Open file…".to_string(), String::new(), "Ctrl+O"),
                Row::OpenFolder => ("Open folder…".to_string(), String::new(), "Ctrl+Shift+O"),
                Row::Source(i) => (self.sources[i].name.clone(), self.sources[i].kind().to_string(), ""),
                Row::Recent => {
                    let p = self.recent.as_ref().expect("recent row");
                    (crate::song_name_or_dir(p), home_relative(p.parent().unwrap_or(p)), "")
                }
            };
            let mut room_right = right;
            if !key.is_empty() {
                let kw = measure(key, 13.0);
                text(key.into(), right - kw, ky, 13.0, white(0.32));
                room_right = right - kw - 16.0;
            }
            // The level of the source being listened to.
            if let (Row::Source(i), Some(p)) = (row, &self.preview) {
                if lit && self.sources.get(i).is_some_and(|s| s.id == p.source.id) {
                    let peaks = p.ring.peaks(METER_BARS);
                    let mx = right - METER_BARS as f32 * 3.0;
                    let cy = top + ROW_H / 2.0;
                    for (k, peak) in peaks.iter().enumerate() {
                        let x = mx + (METER_BARS - peaks.len() + k) as f32 * 3.0;
                        let bh = 3.0 + 17.0 * peak.sqrt().min(1.0);
                        ui.rect(x, cy - bh / 2.0, 2.0, bh, white(0.85));
                    }
                    room_right = mx - 16.0;
                }
            }
            let nx = x0 + 20.0;
            let name = fit(&name, 16.0, room_right - nx, measure);
            let nw = measure(&name, 16.0);
            text(name, nx, ty, 16.0, white(name_a));
            if !sub.is_empty() && room_right - (nx + nw + 16.0) > 40.0 {
                text(fit(&sub, 14.0, room_right - (nx + nw + 16.0), measure), nx + nw + 16.0, sy, 14.0, white(0.38));
            }
        }
        if self.sources.is_empty() {
            let msg = if self.listing.is_some() { "Looking for sources…" } else { "Nothing to capture (no PulseAudio or PipeWire)" };
            text(msg.into(), x0 + 20.0, capture_top + (ROW_H - 14.0 * 1.3) / 2.0, 14.0, white(0.38));
        }

        if overlay {
            let hint = "Tab closes";
            let hw = measure(hint, 14.0);
            text(hint.into(), w - 16.0 - hw, h - 16.0 - 14.0 * 1.3, 14.0, white(0.45));
        } else {
            let hint = "or drop a file or folder anywhere";
            let hw = measure(hint, 14.0);
            text(hint.into(), (w - hw) / 2.0, h - 40.0 - 14.0 * 1.3, 14.0, white(0.38));
        }
    }
}

/// `path` with the home directory shown as "~".
fn home_relative(path: &std::path::Path) -> String {
    let s = path.display().to_string();
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() && s.starts_with(&home) => format!("~{}", &s[home.len()..]),
        _ => s,
    }
}
