//! Settings that persist between runs, in a small text file:
//! `$XDG_CONFIG_HOME/mviz/settings` (usually `~/.config/mviz/settings`),
//! one `key=value` per line. Currently only the volume.

use std::path::PathBuf;

fn path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("mviz").join("settings"))
}

/// The saved volume (0..=1), if there is one.
pub fn load_volume() -> Option<f32> {
    let text = std::fs::read_to_string(path()?).ok()?;
    parse_volume(&text)
}

fn parse_volume(text: &str) -> Option<f32> {
    text.lines()
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| k.trim() == "volume")
        .and_then(|(_, v)| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, 1.0))
}

/// Saves the volume; failures are reported once and otherwise ignored
/// (playback must never depend on it).
pub fn save_volume(volume: f32) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let result = path().ok_or_else(|| "no config directory".to_string()).and_then(|p| {
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        // Write to a temporary file, then rename: never leaves a half-written file.
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, format!("volume={volume:.2}\n")).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &p).map_err(|e| e.to_string())
    });
    if let Err(e) = result {
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("mviz: could not save the volume: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_volume;

    #[test]
    fn reads_volume() {
        assert_eq!(parse_volume("volume=0.65\n"), Some(0.65));
        assert_eq!(parse_volume("other=1\n volume = 0.3 \n"), Some(0.3));
        assert_eq!(parse_volume("volume=2.0"), Some(1.0));
        assert_eq!(parse_volume("volume=abc"), None);
        assert_eq!(parse_volume(""), None);
    }
}
