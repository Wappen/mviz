//! System media controls (MPRIS over D-Bus): the Plasma media widget, lock
//! screen, media keys, Bluetooth headphone buttons and KDE Connect all talk
//! to mviz through this.
//!
//! The D-Bus service runs on its own thread with a small single-threaded
//! executor. Requests from the system arrive as `Command`s on `commands`;
//! mviz reports its state with `send(Update)`. Nothing here can block audio
//! or drawing, and if D-Bus is unavailable mviz simply runs without it.

use crate::audio::TrackInfo;
use futures::task::LocalSpawnExt;
use mpris_server::{LoopStatus, Metadata, PlaybackStatus, Player, Time, TrackId};
use std::path::PathBuf;
use std::sync::mpsc;

/// What the system asks mviz to do.
pub enum Command {
    PlayPause,
    Play,
    Pause,
    /// Pause and go back to the start.
    Stop,
    /// Move by this many seconds.
    SeekBy(f64),
    /// Go to this position (seconds).
    SeekTo(f64),
    Volume(f32),
    Raise,
    Quit,
}

/// What mviz tells the system.
pub enum Update {
    Playing(bool),
    /// Current position (s); clients read it when they need it.
    Position(f64),
    /// Playback jumped (seek or loop) to this position (s).
    Seeked(f64),
    Volume(f32),
    /// Song length (s), once known exactly.
    Length(f64),
}

pub struct Mpris {
    updates: async_channel::Sender<Update>,
    pub commands: mpsc::Receiver<Command>,
    cover: Option<PathBuf>,
}

fn micros(secs: f64) -> Time {
    Time::from_micros((secs.max(0.0) * 1e6) as i64)
}

impl Mpris {
    /// Publishes mviz as a media player. `name` is shown when the file has no title tag.
    pub fn start(info: &TrackInfo, name: &str, length: f64, volume: f32) -> Mpris {
        let (updates, update_rx) = async_channel::unbounded::<Update>();
        let (cmd_tx, commands) = mpsc::channel();
        let cover = info.cover.as_ref().and_then(|(mime, data)| write_cover(mime, data));

        let mut meta = Metadata::builder()
            .trackid(TrackId::try_from("/org/mviz/Track").unwrap_or(TrackId::NO_TRACK))
            .title(info.title.clone().unwrap_or_else(|| name.to_string()))
            .length(micros(length));
        if let Some(a) = &info.artist {
            meta = meta.artist([a.clone()]);
        }
        if let Some(a) = &info.album {
            meta = meta.album(a.clone());
        }
        if let Some(p) = &cover {
            meta = meta.art_url(format!("file://{}", p.display()));
        }
        let meta = meta.build();

        std::thread::Builder::new()
            .name("mpris".into())
            .spawn(move || {
                let mut pool = futures::executor::LocalPool::new();
                let spawner = pool.spawner();
                pool.run_until(async move {
                    // One name per process, so several open songs can coexist.
                    let bus_name = format!("mviz.instance{}", std::process::id());
                    let player = match Player::builder(&bus_name)
                        .identity("mviz")
                        .desktop_entry("mviz")
                        .can_play(true)
                        .can_pause(true)
                        .can_seek(true)
                        .can_raise(true)
                        .can_quit(true)
                        .can_go_next(false)
                        .can_go_previous(false)
                        .loop_status(LoopStatus::Track)
                        .playback_status(PlaybackStatus::Playing)
                        .volume(volume as f64)
                        .metadata(meta)
                        .build()
                        .await
                    {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("mviz: media controls unavailable: {e}");
                            return;
                        }
                    };
                    // Each request becomes a Command for the main loop.
                    macro_rules! on {
                        ($connect:ident, $cmd:expr) => {{
                            let tx = cmd_tx.clone();
                            player.$connect(move |_| {
                                let _ = tx.send($cmd);
                            });
                        }};
                    }
                    on!(connect_play_pause, Command::PlayPause);
                    on!(connect_play, Command::Play);
                    on!(connect_pause, Command::Pause);
                    on!(connect_stop, Command::Stop);
                    on!(connect_raise, Command::Raise);
                    on!(connect_quit, Command::Quit);
                    let tx = cmd_tx.clone();
                    player.connect_seek(move |_, offset| {
                        let _ = tx.send(Command::SeekBy(offset.as_micros() as f64 / 1e6));
                    });
                    let tx = cmd_tx.clone();
                    player.connect_set_position(move |_, _, at| {
                        let _ = tx.send(Command::SeekTo(at.as_micros() as f64 / 1e6));
                    });
                    let tx = cmd_tx.clone();
                    player.connect_set_volume(move |_, v| {
                        let _ = tx.send(Command::Volume(v.clamp(0.0, 1.0) as f32));
                    });
                    let _ = spawner.spawn_local(player.run());

                    // Mirror mviz's state into the published properties.
                    while let Ok(update) = update_rx.recv().await {
                        let _ = match update {
                            Update::Playing(on) => {
                                let status = if on { PlaybackStatus::Playing } else { PlaybackStatus::Paused };
                                player.set_playback_status(status).await
                            }
                            Update::Position(s) => {
                                player.set_position(micros(s));
                                Ok(())
                            }
                            Update::Seeked(s) => {
                                player.set_position(micros(s));
                                player.seeked(micros(s)).await
                            }
                            Update::Volume(v) => player.set_volume(v as f64).await,
                            Update::Length(s) => {
                                let mut m = player.metadata().clone();
                                m.set_length(Some(micros(s)));
                                player.set_metadata(m).await
                            }
                        };
                    }
                });
            })
            .ok();
        Mpris { updates, commands, cover }
    }

    pub fn send(&self, update: Update) {
        let _ = self.updates.try_send(update);
    }
}

impl Drop for Mpris {
    fn drop(&mut self) {
        if let Some(p) = &self.cover {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// MPRIS refers to cover art by file URL: write it to the cache directory.
fn write_cover(mime: &str, data: &[u8]) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    let ext = match mime {
        "image/png" => "png",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "jpg",
    };
    let dir = base.join("mviz");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("cover-{}.{ext}", std::process::id()));
    std::fs::write(&path, data).ok()?;
    Some(path)
}
