//! Video export: `mviz --render <song> <out.mp4> [--start S] [--seconds N] [--size WxH]`
//!
//! Renders the visualizer offscreen with the same GPU pipeline as the window,
//! at exactly 60 frames per second of song time (so picture and sound stay in
//! sync however fast the machine is), and pipes the frames into ffmpeg, which
//! adds the matching stretch of audio.

use crate::analysis;
use crate::audio::Source;
use crate::gfx::{Gfx, Scene};
use crate::modes::{self, stereo::Stereo};
use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicU32;

const FPS: u32 = 60;
/// Rendered before the excerpt starts, so the afterglow has settled.
const WARM_UP: f64 = 1.5;

struct Options {
    song: PathBuf,
    out: PathBuf,
    start: f64,
    seconds: f64,
    size: (u32, u32),
}

fn parse(args: &[OsString]) -> Result<Options, String> {
    let usage = "usage: mviz --render <song> <out.mp4> [--start SECONDS] [--seconds N] [--size WxH]";
    let mut positional = Vec::new();
    let (mut start, mut seconds, mut size) = (0.0, 10.0, (1280, 720));
    let mut it = args.iter().map(|a| a.to_string_lossy().into_owned());
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{a} needs a value\n{usage}"));
        match a.as_str() {
            "--render" => {}
            "--start" => start = value()?.parse().map_err(|_| "--start: not a number")?,
            "--seconds" => seconds = value()?.parse().map_err(|_| "--seconds: not a number")?,
            "--size" => {
                let v = value()?;
                let (w, h) = v.split_once('x').ok_or("--size: expected WxH, e.g. 1280x720")?;
                size = (w.parse().map_err(|_| "--size: bad width")?, h.parse().map_err(|_| "--size: bad height")?);
            }
            _ => positional.push(PathBuf::from(a)),
        }
    }
    let [song, out]: [PathBuf; 2] = positional.try_into().map_err(|_| usage.to_string())?;
    Ok(Options { song, out, start, seconds, size })
}

pub fn run(args: &[OsString]) -> Result<(), String> {
    let o = parse(args)?;
    eprintln!("decoding and analysing {} …", o.song.display());
    let source = Source::open(&o.song)?;
    let rate = source.rate;
    let track = source.track(rate);
    source.decode_into(track.clone())?;
    let tl = analysis::analyze(&track, &AtomicU32::new(0));

    let (w, h) = o.size;
    let mut gfx = Gfx::new_offscreen(w, h);
    let mut stereo = Stereo::new(rate);

    let mut ffmpeg = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &format!("{w}x{h}"), "-r", &FPS.to_string(), "-i", "-"])
        .args(["-ss", &o.start.to_string(), "-t", &o.seconds.to_string(), "-i"])
        .arg(&o.song)
        .args(["-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"])
        .args(["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart"])
        .arg(&o.out)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start ffmpeg (is it installed?): {e}"))?;
    let mut pipe = ffmpeg.stdin.take().ok_or("no pipe to ffmpeg")?;

    let dt = 1.0 / FPS as f64;
    let warm = (WARM_UP.min(o.start) / dt) as i64;
    let frames = (o.seconds / dt).round() as i64;
    for i in -warm..frames {
        let t = o.start + i as f64 * dt;
        let mut scene = Scene { clear: modes::base(), ..Default::default() };
        scene.phosphor = Some(stereo.draw(&mut scene.background, &track, rate, t * rate as f64, dt as f32, Some(&tl), w as f32, h as f32));
        gfx.render(scene);
        if i >= 0 {
            pipe.write_all(&gfx.read_frame()).map_err(|e| format!("ffmpeg stopped: {e}"))?;
            if i % FPS as i64 == 0 {
                eprint!("\rrendering {:.0} / {:.0} s", i as f64 * dt, o.seconds);
            }
        }
    }
    drop(pipe);
    let status = ffmpeg.wait().map_err(|e| e.to_string())?;
    eprintln!();
    if !status.success() {
        return Err(format!("ffmpeg failed ({status})"));
    }
    eprintln!("wrote {}", o.out.display());
    Ok(())
}

#[cfg(test)]
mod preview {
    use super::*;

    fn save(gfx: &Gfx, w: u32, h: u32, path: &str) {
        let rgba = gfx.read_frame();
        let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
        for p in rgba.chunks_exact(4) {
            out.extend_from_slice(&p[..3]);
        }
        std::fs::write(path, out).unwrap();
    }

    /// `MVIZ_OUT=dir [MVIZ_FILE=song] cargo test --release preview -- --ignored`
    #[test]
    #[ignore]
    fn start_screen() {
        let dir = std::env::var("MVIZ_OUT").unwrap();
        let (w, h) = (1600u32, 1000u32);
        let mut gfx = Gfx::new_offscreen(w, h);
        let mut home = crate::home::Home::new();
        std::thread::sleep(std::time::Duration::from_millis(500));
        home.update(0.0);
        for t in [0.0, 4.0, 8.0, 16.0] {
            let mut scene = Scene { clear: modes::base(), blur: 1.0, ..Default::default() };
            let mut measure = |s: &str, size: f32| gfx.text_width(s, size);
            home.draw(&mut scene.ui, &mut scene.texts, Some(&mut scene.background), &mut measure, "mviz", false, 1.0, t, (w as f32, h as f32));
            gfx.render(scene);
            save(&gfx, w, h, &format!("{dir}/home_{t}.ppm"));
        }
        let Ok(song) = std::env::var("MVIZ_FILE") else { return };
        let source = Source::open(std::path::Path::new(&song)).unwrap();
        let rate = source.rate;
        let track = source.track(rate);
        source.decode_into(track.clone()).unwrap();
        let tl = analysis::analyze(&track, &AtomicU32::new(0));
        let mut stereo = Stereo::new(rate);
        let dt = 1.0 / 60.0;
        for i in 0..240 {
            let t = 30.0 + i as f64 * dt;
            let blur = if i == 239 { 1.0 } else { 0.0 };
            let mut scene = Scene { clear: modes::base(), blur, ..Default::default() };
            scene.phosphor = Some(stereo.draw(&mut scene.background, &track, rate, t * rate as f64, dt as f32, Some(&tl), w as f32, h as f32));
            if i >= 238 {
                if blur > 0.0 {
                    scene.ui.rect(0.0, 0.0, w as f32, h as f32, [0.0, 0.0, 0.0, 0.5]);
                    let mut measure = |s: &str, size: f32| gfx.text_width(s, size);
                    home.draw(&mut scene.ui, &mut scene.texts, None, &mut measure, "Song", true, 1.0, t, (w as f32, h as f32));
                }
                gfx.render(scene);
                save(&gfx, w, h, &format!("{dir}/song_{}.ppm", if blur > 0.0 { "menu" } else { "plain" }));
            } else {
                gfx.render(scene);
            }
        }
    }
}
