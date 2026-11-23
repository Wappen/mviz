//! mviz: a looping music player with an audio-driven visualizer.
//!
//! Usage: mviz [<audio file> | <folder>]
//!          without one, a start screen opens: open a file or folder, or
//!          capture what's playing (or an input)
//!        mviz --render <audio file> <out.mp4> [--start S] [--seconds N] [--size WxH]
//!          renders a video of the visualizer (with the audio) instead of playing
//!   right-click   pause / resume (or Space)
//!   scroll        volume
//!   left-drag     move the window
//!   double-click  fullscreen on/off (or F11)
//!   bottom edge   waveform timeline (click or drag to seek)
//!   right edge    the playlist, when playing a folder (L keeps it open)
//!   ← / →         previous / next song of the folder
//!   Tab           the start screen, over the picture
//!   Ctrl+O        open a file (Ctrl+Shift+O: a folder)
//!   drop a file   play that file (or folder) instead
//!   Esc           quit (or close the start screen)

mod analysis;
mod audio;
mod capture;
mod dnd;
mod dsp;
mod gfx;
mod home;
mod modes;
mod mpris;
mod params;
mod picker;
mod playlist;
mod render;
mod settings;
mod ui;

use audio::Player;
use capture::Capture;
use gfx::{Gfx, PhosphorFrame, Scene, Text};
use modes::stereo::Stereo;
use playlist::Playlist;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU32;
use std::sync::{mpsc, Arc};
use std::time::Instant;
use ui::Transport;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

/// Messages from a loader thread.
enum Loaded {
    /// Header read: play this track (it fills while playing). `out`: the
    /// output config chosen for it (none when it joins a running player).
    Ready { track: Arc<audio::Track>, out: Option<(cpal::StreamConfig, cpal::SampleFormat)>, info: audio::TrackInfo },
    /// The full analysis, once the whole song is decoded.
    Analysed(analysis::Timeline),
    Failed(String),
}

/// Opens the file and sends the (still empty) track at once; then decodes
/// it and sends the analysis. Playback starts after the first moments of
/// decoding, not after all of it. With `rate`, the track is made for a
/// running player at that rate; without, an output is chosen for it.
fn load(path: PathBuf, rate: Option<u32>, tx: mpsc::Sender<Loaded>) {
    let run = || -> Result<(), String> {
        let source = audio::Source::open(&path)?;
        let (track, out) = match rate {
            Some(r) => (source.track(r), None),
            None => {
                let (config, format) = audio::pick_config(source.rate)?;
                (source.track(config.sample_rate), Some((config, format)))
            }
        };
        let _ = tx.send(Loaded::Ready { track: track.clone(), out, info: source.info.clone() });
        source.decode_into(track.clone())?;
        let _ = tx.send(Loaded::Analysed(analysis::analyze(&track, &AtomicU32::new(0))));
        Ok(())
    };
    if let Err(e) = run() {
        let _ = tx.send(Loaded::Failed(e));
    }
}

fn spawn_load(path: PathBuf, rate: Option<u32>) -> mpsc::Receiver<Loaded> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || load(path, rate, tx));
    rx
}

/// The name shown for a song: its file name without the extension.
fn song_name(path: &Path) -> String {
    path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// A song's name, or a folder's.
fn song_name_or_dir(path: &Path) -> String {
    if path.is_dir() {
        path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string())
    } else {
        song_name(path)
    }
}

/// Where a song comes from: a folder (and its place in it), or a lone file.
#[derive(Clone)]
struct Origin {
    playlist: Option<(Arc<Playlist>, usize)>,
}

/// A song being opened; what plays keeps playing until it is ready.
struct Pending {
    rx: mpsc::Receiver<Loaded>,
    name: String,
    origin: Origin,
}

/// The next song of the folder, decoded ahead to follow without a gap.
struct Queued {
    rx: mpsc::Receiver<Loaded>,
    index: usize,
    name: String,
    info: Option<audio::TrackInfo>,
    tl: Option<analysis::Timeline>,
}

struct Song {
    player: Player,
    /// The full analysis, once it has arrived.
    tl: Option<analysis::Timeline>,
    rx: mpsc::Receiver<Loaded>,
    name: String,
    origin: Origin,
    queued: Option<Queued>,
    /// Song names and lengths of the folder (for the playlist).
    names: Vec<String>,
}

enum Feed {
    Song(Song),
    Live(Capture),
}

impl Feed {
    fn transport(&mut self) -> &mut dyn Transport {
        match self {
            Feed::Song(s) => &mut s.player,
            Feed::Live(c) => c,
        }
    }
}

struct Playing {
    feed: Feed,
    stereo: Stereo,
    dim: f32,
}

/// The state last reported to the media controls.
struct Told {
    playing: bool,
    volume: f32,
    /// Position (s) and when it was reported (app time).
    position: f64,
    at: f64,
}

struct App {
    playing: Option<Playing>,
    pending: Option<Pending>,
    /// System media controls (while songs play), and what they were last told.
    mpris: Option<mpris::Mpris>,
    told: Told,
    ui: ui::Ui,
    home: home::Home,
    home_open: bool,
    home_alpha: f32,
    /// An open file dialog.
    picker: Option<mpsc::Receiver<Option<picker::Picked>>>,
    /// A short message (something could not be opened) and when it appeared.
    notice: Option<(String, f64)>,
    modifiers: ModifiersState,
    gfx: Option<Gfx>,
    /// Files dropped onto the window (Wayland; X11 drops come as events).
    drops: Option<mpsc::Receiver<PathBuf>>,
    start: Instant,
    last_frame: Instant,
}

impl App {
    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn notify(&mut self, msg: String) {
        eprintln!("mviz: {msg}");
        self.notice = Some((msg, self.now()));
    }

    /// Plays a file, or every song of a folder.
    fn open_path(&mut self, path: PathBuf) {
        if path.is_dir() {
            match Playlist::scan(&path) {
                Ok(pl) => {
                    settings::save_recent(&path);
                    self.play(Arc::new(pl), 0);
                }
                Err(e) => self.notify(e),
            }
        } else {
            settings::save_recent(&path);
            let name = song_name(&path);
            self.pending = Some(Pending { rx: spawn_load(path, None), name, origin: Origin { playlist: None } });
        }
    }

    /// Plays song `index` of `pl`.
    fn play(&mut self, pl: Arc<Playlist>, index: usize) {
        let path = pl.songs[index].clone();
        let name = song_name(&path);
        self.pending = Some(Pending { rx: spawn_load(path, None), name, origin: Origin { playlist: Some((pl, index)) } });
    }

    /// The next (`step` 1) or previous (-1) song of the folder being played.
    fn skip(&mut self, step: i32) {
        let Some(Playing { feed: Feed::Song(song), .. }) = &self.playing else { return };
        let Some((pl, i)) = song.origin.playlist.clone() else { return };
        let j = if step > 0 { pl.next(i) } else { pl.prev(i) };
        self.play(pl, j);
    }

    fn open_dialog(&mut self, folder: bool) {
        if self.picker.is_some() {
            return;
        }
        let Some(gfx) = &self.gfx else { return };
        self.picker = Some(picker::open(gfx.window(), folder, settings::load_recent()));
    }

    fn show_home(&mut self, open: bool) {
        if open == self.home_open {
            return;
        }
        if open {
            self.home.refresh();
        } else {
            self.home.close();
        }
        self.home_open = open;
    }

    /// Carries out a choice on the start screen.
    fn choose(&mut self, choice: home::Choice) {
        match choice {
            home::Choice::OpenFile => self.open_dialog(false),
            home::Choice::OpenFolder => self.open_dialog(true),
            home::Choice::Recent(p) => self.open_path(p),
            home::Choice::Capture(source) => {
                let capture = match self.home.take_preview(&source) {
                    Some(c) => Ok(c),
                    None => Capture::start(&source),
                };
                match capture {
                    Ok(c) => {
                        settings::save_capture(&source.id);
                        let mut stereo = Stereo::new(c.ring.rate);
                        stereo.causal = true;
                        let dim = self.playing.as_ref().map_or(0.0, |p| p.dim);
                        self.pending = None;
                        self.mpris = None;
                        self.playing = Some(Playing { feed: Feed::Live(c), stereo, dim });
                        self.set_title(&source.name);
                        self.show_home(false);
                    }
                    Err(e) => self.notify(e),
                }
            }
        }
    }

    fn set_title(&self, name: &str) {
        if let Some(gfx) = &self.gfx {
            gfx.window().set_title(&if name.is_empty() { "mviz".into() } else { format!("{name} — mviz") });
        }
    }

    /// Switches to the song being opened once it is ready to play.
    fn poll_pending(&mut self) {
        let Some(pending) = &self.pending else { return };
        let msg = match pending.rx.try_recv() {
            Err(mpsc::TryRecvError::Empty) => return,
            Ok(m) => m,
            Err(mpsc::TryRecvError::Disconnected) => Loaded::Failed("could not load the file".into()),
        };
        let pending = self.pending.take().expect("checked above");
        self.start_song(pending, msg);
    }

    fn start_song(&mut self, pending: Pending, msg: Loaded) {
        let started = match msg {
            Loaded::Ready { track, out: Some((config, format)), info } => {
                let volume = match &self.playing {
                    Some(Playing { feed: Feed::Song(s), .. }) => s.player.volume(),
                    _ => settings::load_volume().unwrap_or(1.0),
                };
                Player::start(track, config, format, volume).map(|p| (p, info))
            }
            Loaded::Failed(e) => Err(e),
            _ => Err("could not load the file".into()),
        };
        let (player, info) = match started {
            Ok(x) => x,
            Err(e) => {
                self.notify(format!("{}: {e}", pending.name));
                if self.playing.is_none() {
                    self.show_home(true);
                }
                return;
            }
        };
        let name = pending.name;
        match &mut self.mpris {
            Some(m) => m.set_track(&info, &name, player.duration()),
            None => self.mpris = Some(mpris::Mpris::start(&info, &name, player.duration(), player.volume())),
        }
        self.told = Told { playing: true, volume: player.volume(), position: 0.0, at: self.now() };
        let names = pending.origin.playlist.as_ref().map(|(pl, _)| pl.names()).unwrap_or_default();
        let song = Song { player, tl: None, rx: pending.rx, name: name.clone(), origin: pending.origin, queued: None, names };
        // Fresh analysis state; the afterglow on screen fades over.
        let stereo = Stereo::new(song.player.rate);
        let dim = self.playing.as_ref().map_or(0.0, |p| p.dim);
        // The old player (and its audio stream) is dropped here.
        self.playing = Some(Playing { feed: Feed::Song(song), stereo, dim });
        self.set_title(&name);
        self.show_home(false);
    }

    /// Analysis arriving, the next song of a folder decoding ahead, and the
    /// switch to it.
    fn poll_song(&mut self) {
        let Some(Playing { feed: Feed::Song(song), .. }) = &mut self.playing else { return };
        match song.rx.try_recv() {
            Ok(Loaded::Analysed(tl)) => {
                song.tl = Some(tl);
                // Decoding has finished: the exact length is known now.
                if let Some(m) = &self.mpris {
                    m.send(mpris::Update::Length(song.player.duration()));
                }
            }
            Ok(Loaded::Failed(e)) => {
                let name = song.name.clone();
                self.notify(format!("{name}: {e}"));
                return;
            }
            _ => {}
        }
        // Decode the next song once this one is done (one at a time).
        if let Some((pl, i)) = song.origin.playlist.clone().filter(|(pl, _)| pl.len() > 1) {
            if song.queued.is_none() && song.tl.is_some() {
                let j = pl.next(i);
                let path = pl.songs[j].clone();
                song.queued = Some(Queued { rx: spawn_load(path.clone(), Some(song.player.rate)), index: j, name: song_name(&path), info: None, tl: None });
            }
        }
        if let Some(q) = &mut song.queued {
            while let Ok(msg) = q.rx.try_recv() {
                match msg {
                    Loaded::Ready { track, info, .. } => {
                        song.player.queue(track);
                        q.info = Some(info);
                    }
                    Loaded::Analysed(tl) => q.tl = Some(tl),
                    // The song after it loops instead (the queue stays, so it is not retried).
                    Loaded::Failed(e) => eprintln!("mviz: {}: {e}", q.name),
                }
            }
        }
        if song.player.poll_switch() {
            if let Some(q) = song.queued.take() {
                song.tl = q.tl;
                song.rx = q.rx;
                song.name = q.name;
                if let Some((_, i)) = &mut song.origin.playlist {
                    *i = q.index;
                }
                if let (Some(m), Some(info)) = (&mut self.mpris, &q.info) {
                    m.set_track(info, &song.name, song.player.duration());
                }
                let name = song.name.clone();
                self.set_title(&name);
            }
        }
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        let now = self.now();
        let dt = self.last_frame.elapsed().as_secs_f32();
        self.last_frame = Instant::now();
        if self.gfx.is_none() {
            return;
        }

        if let Some(path) = self.drops.as_ref().and_then(|d| d.try_iter().last()) {
            self.open_path(path);
        }
        if let Some(picked) = self.picker.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.picker = None;
            match picked {
                Some(picker::Picked::File(p)) | Some(picker::Picked::Folder(p)) => self.open_path(p),
                None => {}
            }
        }
        self.poll_pending();
        self.poll_song();
        let commands = self.sync_media_controls(event_loop, now);
        for step in commands {
            self.skip(step);
        }
        if self.playing.is_none() {
            self.home_open = true;
        }
        if self.home_open {
            self.home.update(now);
        }

        let gfx = self.gfx.as_mut().expect("checked above");
        let (w, h) = gfx.size();
        let window = gfx.window().clone();
        let mut scene = Scene { clear: modes::base(), ..Default::default() };
        self.ui.controls = !self.home_open && self.playing.is_some();
        let target = if self.home_open { 1.0 } else { 0.0 };
        self.home_alpha += (target - self.home_alpha) * (dt * 10.0).min(1.0);

        // What the timeline and playlist show.
        let mut title = String::new();
        let mut subtitle = String::new();
        let mut waveform = (Vec::new(), 0usize);
        let mut time = None;
        let mut paused = false;
        let mut volume = None;
        let mut lengths = Vec::new();
        let mut playlist_info: Option<(String, usize)> = None;
        let mut home_title = "mviz".to_string();

        if let Some(p) = &mut self.playing {
            let (songs, current) = match &p.feed {
                Feed::Song(s) => s.origin.playlist.as_ref().map_or((0, 0), |(pl, i)| (pl.len(), *i)),
                Feed::Live(_) => (0, 0),
            };
            self.ui.update(now, dt, &window, Some(p.feed.transport()), (w, h), songs, current);
            let (frame, rate) = match &p.feed {
                Feed::Song(s) => (s.player.position(), s.player.rate),
                Feed::Live(c) => (c.position(), c.ring.rate),
            };
            let (samples, tl): (&dyn audio::Samples, _) = match &p.feed {
                Feed::Song(s) => (&s.player.track, s.tl.as_ref()),
                Feed::Live(c) => (&c.ring, None),
            };
            scene.phosphor = Some(p.stereo.draw(&mut scene.background, samples, rate, frame, dt, tl, w, h));
            paused = p.feed.transport().paused();
            let target = if paused { 1.0 } else { 0.0 };
            p.dim += (target - p.dim) * (dt * 10.0).min(1.0);
            match &p.feed {
                Feed::Song(s) => {
                    title = s.name.clone();
                    home_title = s.name.clone();
                    waveform = s.player.track.waveform();
                    time = Some((frame / rate as f64, s.player.duration()));
                    volume = Some(s.player.volume());
                    if let Some((pl, i)) = &s.origin.playlist {
                        subtitle = format!("{} / {} · {}", i + 1, pl.len(), pl.name);
                        home_title = pl.name.clone();
                        lengths = pl.lengths.lock().unwrap().clone();
                        playlist_info = Some((pl.name.clone(), *i));
                    }
                }
                Feed::Live(c) => {
                    title = c.source.name.clone();
                    home_title = c.source.name.clone();
                    subtitle = c.source.kind().to_string();
                    let n = (ui::LIVE_SECS * c.ring.rate as f64 / capture::WAVE_FRAMES as f64) as usize;
                    waveform = (c.ring.waveform(n), n);
                }
            }
        } else {
            self.ui.update(now, dt, &window, None, (w, h), 0, 0);
            // Nothing playing: the start screen's light drifts in the phosphor.
            scene.phosphor = Some(PhosphorFrame { live: Default::default(), advance: Some(dt.min(0.1)) });
        }

        // Paused or start screen over the picture: dim it.
        let pause_dim = self.playing.as_ref().map_or(0.0, |p| 0.6 * p.dim);
        let home_dim = if self.playing.is_some() { 0.72 * self.home_alpha } else { 0.0 };
        let dim = pause_dim.max(home_dim);
        if dim > 0.005 {
            scene.ui.rect(0.0, 0.0, w, h, [0.0, 0.0, 0.0, dim]);
        }

        let mut measure = |s: &str, size: f32| gfx.text_width(s, size);
        let songs: Vec<String> = match &self.playing {
            Some(Playing { feed: Feed::Song(s), .. }) => s.names.clone(),
            _ => Vec::new(),
        };
        let status = ui::Status {
            title: &title,
            subtitle: &subtitle,
            paused,
            time,
            waveform: &waveform.0,
            expected: waveform.1,
            volume,
            playlist: playlist_info.as_ref().map(|(name, i)| ui::PlaylistView { name, songs: &songs, lengths: &lengths, current: *i }),
        };
        self.ui.draw(&mut scene.ui, &mut scene.texts, &mut measure, &status, now, (w, h));

        if self.home_alpha > 0.01 {
            let standalone = self.playing.is_none();
            let glows = if standalone { scene.phosphor.as_mut().map(|p| &mut p.live) } else { None };
            self.home.draw(&mut scene.ui, &mut scene.texts, glows, &mut measure, &home_title, !standalone, self.home_alpha, now, (w, h));
        }

        if let Some((msg, at)) = &self.notice {
            let a = (1.0 - ((now - at - 3.0) / 0.5) as f32).clamp(0.0, 1.0);
            if a <= 0.0 {
                self.notice = None;
            } else {
                let tw = measure(msg, 15.0);
                scene.texts.push(Text { text: msg.clone(), x: (w - tw) / 2.0, y: 16.0, size: 15.0, color: [1.0, 1.0, 1.0, 0.85 * a] });
            }
        }

        gfx.render(scene);
    }

    /// Carries out what the system's media controls ask, and reports back
    /// play state, volume and position (flagging jumps from seeks and loops).
    /// Returns songs to skip by (next / previous).
    fn sync_media_controls(&mut self, event_loop: &ActiveEventLoop, now: f64) -> Vec<i32> {
        use mpris::{Command, Update};
        let mut skips = Vec::new();
        let (Some(m), Some(Playing { feed: Feed::Song(song), .. })) = (&self.mpris, &mut self.playing) else { return skips };
        let player = &mut song.player;
        while let Ok(cmd) = m.commands.try_recv() {
            let position = player.position() / player.rate as f64;
            match cmd {
                Command::PlayPause => player.toggle_pause(),
                Command::Play if player.paused() => player.toggle_pause(),
                Command::Pause if !player.paused() => player.toggle_pause(),
                Command::Play | Command::Pause => {}
                Command::Stop => {
                    if !player.paused() {
                        player.toggle_pause();
                    }
                    player.seek(0.0);
                }
                Command::SeekBy(d) => player.seek((position + d).clamp(0.0, player.duration())),
                Command::SeekTo(t) => player.seek(t.clamp(0.0, player.duration())),
                Command::Volume(v) => {
                    player.set_volume(v);
                    settings::save_volume(v);
                }
                Command::Next => skips.push(1),
                Command::Previous => skips.push(-1),
                Command::Raise => {
                    if let Some(g) = &self.gfx {
                        g.window().focus_window();
                    }
                }
                Command::Quit => event_loop.exit(),
            }
        }

        let told = &mut self.told;
        let playing = !player.paused();
        if playing != told.playing {
            told.playing = playing;
            m.send(Update::Playing(playing));
        }
        let volume = player.volume();
        if (volume - told.volume).abs() > 1e-4 {
            told.volume = volume;
            m.send(Update::Volume(volume));
        }
        let position = player.position() / player.rate as f64;
        let expected = told.position + if playing { now - told.at } else { 0.0 };
        if (position - expected).abs() > 0.5 {
            m.send(Update::Seeked(position));
        } else if now - told.at > 0.25 {
            m.send(Update::Position(position));
        } else {
            return skips;
        }
        told.position = position;
        told.at = now;
        skips
    }

    fn key(&mut self, event_loop: &ActiveEventLoop, key: Key) {
        let now = self.now();
        let ctrl = self.modifiers.control_key();
        let shift = self.modifiers.shift_key();
        match &key {
            Key::Character(c) if ctrl && c.eq_ignore_ascii_case("o") => self.open_dialog(shift),
            Key::Named(NamedKey::Escape) if self.home_open && self.playing.is_some() => self.show_home(false),
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Named(NamedKey::Tab) if self.playing.is_some() => self.show_home(!self.home_open),
            Key::Named(NamedKey::F11) => {
                if let Some(g) = &self.gfx {
                    ui::toggle_fullscreen(g.window());
                }
            }
            _ if self.home_open => {
                if let Some(choice) = self.home.key(&key, now) {
                    self.choose(choice);
                }
            }
            Key::Named(NamedKey::Space) => {
                if let Some(p) = &mut self.playing {
                    p.feed.transport().toggle_pause();
                }
            }
            Key::Named(NamedKey::ArrowRight) => self.skip(1),
            Key::Named(NamedKey::ArrowLeft) => self.skip(-1),
            Key::Character(c) if c.eq_ignore_ascii_case("l") => self.ui.panel_pinned = !self.ui.panel_pinned,
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let title = match &self.playing {
            Some(Playing { feed: Feed::Song(s), .. }) => format!("{} — mviz", s.name),
            _ => "mviz".into(),
        };
        let attrs = Window::default_attributes().with_title(title).with_inner_size(LogicalSize::new(1280.0, 800.0));
        // Declare the app id "mviz" (Wayland) / class "mviz" (X11), so the
        // desktop matches the window to mviz.desktop and shows its icon.
        let attrs = winit::platform::wayland::WindowAttributesExtWayland::with_name(attrs, "mviz", "mviz");
        let attrs = winit::platform::x11::WindowAttributesExtX11::with_name(attrs, "mviz", "mviz");
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.drops = dnd::listen(&window);
        self.gfx = Some(Gfx::new(window, event_loop.owned_display_handle()));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let now = self.now();
        let Some(gfx) = self.gfx.as_mut() else { return };
        let size = gfx.size();
        let scale = gfx.scale();
        let window = gfx.window().clone();
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => gfx.resize(s.width, s.height),
            WindowEvent::RedrawRequested => self.frame(event_loop),
            WindowEvent::DroppedFile(path) => self.open_path(path),
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed && !event.repeat {
                    self.key(event_loop, event.logical_key);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32 / scale, position.y as f32 / scale);
                self.ui.cursor_moved(x, y, now, &window);
                if self.home_open {
                    self.home.hover(x, y, now);
                }
            }
            WindowEvent::CursorLeft { .. } => self.ui.cursor_left(),
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                if self.home_open && pressed && button == MouseButton::Left {
                    let (x, y) = self.ui.mouse();
                    if let Some(choice) = self.home.click(x, y) {
                        self.choose(choice);
                        return;
                    }
                }
                let transport = self.playing.as_mut().map(|p| p.feed.transport());
                if let Some(ui::Action::Play(i)) = self.ui.button(button, pressed, now, &window, transport, size) {
                    if let Some(Playing { feed: Feed::Song(s), .. }) = &self.playing {
                        if let Some((pl, cur)) = s.origin.playlist.clone() {
                            if i != cur {
                                self.play(pl, i);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let transport = self.playing.as_mut().map(|p| &*p.feed.transport());
                match delta {
                    MouseScrollDelta::LineDelta(_, y) => self.ui.wheel(y, 0.0, now, transport, size),
                    MouseScrollDelta::PixelDelta(d) => self.ui.wheel(0.0, d.y as f32 / scale, now, transport, size),
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Animate continuously; presentation is paced by vsync.
        if let Some(gfx) = &self.gfx {
            gfx.window().request_redraw();
        }
    }
}

fn main() {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    if args.iter().any(|a| a == "--render") {
        if let Err(e) = render::run(&args) {
            eprintln!("mviz: {e}");
            std::process::exit(1);
        }
        return;
    }
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let now = Instant::now();
    let mut app = App {
        playing: None,
        pending: None,
        mpris: None,
        told: Told { playing: true, volume: 1.0, position: 0.0, at: 0.0 },
        ui: ui::Ui::new(0.0),
        home: home::Home::new(),
        home_open: false,
        home_alpha: 0.0,
        picker: None,
        notice: None,
        modifiers: ModifiersState::empty(),
        gfx: None,
        drops: None,
        start: now,
        last_frame: now,
    };
    if let Some(path) = args.first().map(PathBuf::from) {
        if !path.exists() {
            eprintln!("mviz: {} does not exist", path.display());
            std::process::exit(1);
        }
        app.open_path(path);
        // Start playing before anything else (the window, the GPU): sound
        // comes first. Decoding and analysis continue in the background.
        if let Some(pending) = app.pending.take() {
            match pending.rx.recv() {
                Ok(msg) => app.start_song(pending, msg),
                Err(_) => app.notify(format!("{}: could not load the file", pending.name)),
            }
        }
    }
    if app.playing.is_none() {
        app.home_open = true;
        app.home_alpha = 1.0;
    }
    event_loop.run_app(&mut app).expect("run");
}
