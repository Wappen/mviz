//! mviz: a looping music player with an audio-driven visualizer.
//!
//! Usage: mviz <audio file>
//!        mviz --render <audio file> <out.mp4> [--start S] [--seconds N] [--size WxH]
//!          renders a video of the visualizer (with the audio) instead of playing
//!   right-click   pause / resume (or Space)
//!   scroll        volume
//!   left-drag     move the window
//!   double-click  fullscreen on/off (or F11)
//!   bottom edge   waveform timeline (click or drag to seek)
//!   Esc           quit

mod analysis;
mod audio;
mod dsp;
mod gfx;
mod modes;
mod mpris;
mod params;
mod render;
mod settings;
mod ui;

use gfx::{Gfx, Scene};
use modes::stereo::Stereo;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::{mpsc, Arc};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Messages from the loader thread.
enum Loaded {
    /// Header read and output chosen: play this track (it fills while playing).
    Ready(Arc<audio::Track>, cpal::StreamConfig, cpal::SampleFormat, audio::TrackInfo),
    /// The full analysis, once the whole song is decoded.
    Analysed(analysis::Timeline),
    Failed(String),
}

/// Opens the file, picks the output, and sends the (still empty) track at
/// once; then decodes it and sends the analysis. Playback starts after the
/// first moments of decoding, not after all of it.
fn load(path: PathBuf, tx: mpsc::Sender<Loaded>) {
    let run = || -> Result<(), String> {
        let source = audio::Source::open(&path)?;
        let (config, format) = audio::pick_config(source.rate)?;
        let track = source.track(config.sample_rate);
        let _ = tx.send(Loaded::Ready(track.clone(), config, format, source.info.clone()));
        source.decode_into(track.clone())?;
        let _ = tx.send(Loaded::Analysed(analysis::analyze(&track, &AtomicU32::new(0))));
        Ok(())
    };
    if let Err(e) = run() {
        let _ = tx.send(Loaded::Failed(e));
    }
}

struct Playing {
    player: audio::Player,
    /// System media controls, and what they were last told.
    mpris: mpris::Mpris,
    told: Told,
    /// The full analysis, once it has arrived.
    tl: Option<analysis::Timeline>,
    rx: mpsc::Receiver<Loaded>,
    stereo: Stereo,
    ui: ui::Ui,
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
    playing: Playing,
    gfx: Option<Gfx>,
    start: Instant,
    last_frame: Instant,
}

impl App {
    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        let now = self.now();
        let dt = self.last_frame.elapsed().as_secs_f32();
        self.last_frame = Instant::now();
        let Some(gfx) = self.gfx.as_mut() else { return };
        let (w, h) = gfx.size();
        let mut scene = Scene { clear: modes::base(), ..Default::default() };

        let p = &mut self.playing;
        match p.rx.try_recv() {
            Ok(Loaded::Analysed(tl)) => {
                p.tl = Some(tl);
                // Decoding has finished: the exact length is known now.
                p.mpris.send(mpris::Update::Length(p.player.duration()));
            }
            Ok(Loaded::Failed(e)) => fail(event_loop, &e),
            _ => {}
        }
        sync_media_controls(p, gfx.window(), event_loop, now);
        p.ui.update(now, dt, gfx.window(), &mut p.player, (w, h));
        let frame = p.player.position();
        scene.phosphor = Some(p.stereo.draw(&mut scene.background, &p.player.track, p.player.rate, frame, dt, p.tl.as_ref(), w, h));
        // Paused: dim everything.
        let target = if p.player.paused() { 1.0 } else { 0.0 };
        p.dim += (target - p.dim) * (dt * 10.0).min(1.0);
        if p.dim > 0.005 {
            scene.ui.rect(0.0, 0.0, w, h, [0.0, 0.0, 0.0, 0.6 * p.dim]);
        }
        let (waveform, expected) = p.player.track.waveform();
        let mut measure = |s: &str, size: f32| gfx.text_width(s, size);
        p.ui.draw(&mut scene.ui, &mut scene.texts, &mut measure, &p.player, &waveform, expected, now, (w, h));

        gfx.render(scene);
    }
}

/// Carries out what the system's media controls ask, and reports back
/// play state, volume and position (flagging jumps from seeks and loops).
fn sync_media_controls(p: &mut Playing, window: &Window, event_loop: &ActiveEventLoop, now: f64) {
    use mpris::{Command, Update};
    while let Ok(cmd) = p.mpris.commands.try_recv() {
        let player = &mut p.player;
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
            Command::Raise => window.focus_window(),
            Command::Quit => event_loop.exit(),
        }
    }

    let player = &p.player;
    let playing = !player.paused();
    if playing != p.told.playing {
        p.told.playing = playing;
        p.mpris.send(Update::Playing(playing));
    }
    let volume = player.volume();
    if (volume - p.told.volume).abs() > 1e-4 {
        p.told.volume = volume;
        p.mpris.send(Update::Volume(volume));
    }
    let position = player.position() / player.rate as f64;
    let expected = p.told.position + if playing { now - p.told.at } else { 0.0 };
    if (position - expected).abs() > 0.5 {
        p.mpris.send(Update::Seeked(position));
    } else if now - p.told.at > 0.25 {
        p.mpris.send(Update::Position(position));
    } else {
        return;
    }
    p.told.position = position;
    p.told.at = now;
}

fn fail(event_loop: &ActiveEventLoop, msg: &str) {
    eprintln!("mviz: {msg}");
    event_loop.exit();
    std::process::exit(1);
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let title = format!("{} — mviz", self.playing.ui.title);
        let attrs = Window::default_attributes().with_title(title).with_inner_size(LogicalSize::new(1280.0, 800.0));
        // Declare the app id "mviz" (Wayland) / class "mviz" (X11), so the
        // desktop matches the window to mviz.desktop and shows its icon.
        let attrs = winit::platform::wayland::WindowAttributesExtWayland::with_name(attrs, "mviz", "mviz");
        let attrs = winit::platform::x11::WindowAttributesExtX11::with_name(attrs, "mviz", "mviz");
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.gfx = Some(Gfx::new(window, event_loop.owned_display_handle()));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let now = self.now();
        let Some(gfx) = self.gfx.as_mut() else { return };
        let size = gfx.size();
        let scale = gfx.scale();
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => gfx.resize(s.width, s.height),
            WindowEvent::RedrawRequested => self.frame(event_loop),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed && !event.repeat {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::Space) => self.playing.player.toggle_pause(),
                        Key::Named(NamedKey::F11) => ui::toggle_fullscreen(gfx.window()),
                        _ => {}
                    }
                }
            }
            event => {
                let p = &mut self.playing;
                let window = gfx.window().clone();
                match event {
                    WindowEvent::CursorMoved { position, .. } => {
                        p.ui.cursor_moved(position.x as f32 / scale, position.y as f32 / scale, now, &window)
                    }
                    WindowEvent::CursorLeft { .. } => p.ui.cursor_left(),
                    WindowEvent::MouseInput { state, button, .. } => {
                        p.ui.button(button, state == ElementState::Pressed, now, &window, &mut p.player, size);
                    }
                    WindowEvent::MouseWheel { delta, .. } => match delta {
                        MouseScrollDelta::LineDelta(_, y) => p.ui.wheel(y, 0.0, now, &p.player),
                        MouseScrollDelta::PixelDelta(d) => p.ui.wheel(0.0, d.y as f32 / scale, now, &p.player),
                    },
                    _ => {}
                }
            }
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
    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: mviz <audio file>");
        std::process::exit(2);
    };
    // Read the header, pick the output and start playing before anything
    // else (the window, the GPU): sound comes first. Decoding and analysis
    // continue in the background.
    let (tx, rx) = mpsc::channel();
    let name = path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    std::thread::spawn(move || load(path, tx));
    let started = match rx.recv() {
        Ok(Loaded::Ready(track, config, format, info)) => {
            audio::Player::start(track, config, format, settings::load_volume().unwrap_or(1.0)).map(|p| (p, info))
        }
        Ok(Loaded::Failed(e)) => Err(e),
        _ => Err("could not load the file".into()),
    };
    let (player, info) = started.unwrap_or_else(|e| {
        eprintln!("mviz: {e}");
        std::process::exit(1);
    });
    let mpris = mpris::Mpris::start(&info, &name, player.duration(), player.volume());
    let told = Told { playing: true, volume: player.volume(), position: 0.0, at: 0.0 };

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let now = Instant::now();
    let ui = ui::Ui::new(0.0, name.clone());
    let playing = Playing { stereo: Stereo::new(player.rate), player, mpris, told, tl: None, rx, ui, dim: 0.0 };
    let mut app = App { playing, gfx: None, start: now, last_frame: now };
    event_loop.run_app(&mut app).expect("run");
}
