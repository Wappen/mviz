//! Settings that persist between runs, in a small text file:
//! `$XDG_CONFIG_HOME/mviz/settings` (usually `~/.config/mviz/settings`),
//! one `key=value` per line: the volume, the last file or folder opened
//! (`recent`) and the last capture source (`capture`).

use std::path::PathBuf;

fn path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("mviz").join("settings"))
}

fn get(text: &str, key: &str) -> Option<String> {
    text.lines()
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| k.trim() == key)
        .map(|(_, v)| v.trim().to_string())
}

fn load(key: &str) -> Option<String> {
    get(&std::fs::read_to_string(path()?).ok()?, key)
}

/// Sets `key` to `value`, keeping the other lines.
fn set(text: &str, key: &str, value: &str) -> String {
    let mut out: String = text
        .lines()
        .filter(|l| l.split_once('=').is_none_or(|(k, _)| k.trim() != key) && !l.trim().is_empty())
        .map(|l| format!("{l}\n"))
        .collect();
    out.push_str(&format!("{key}={value}\n"));
    out
}

/// Saves one setting; failures are reported once and otherwise ignored
/// (playback must never depend on it).
fn save(key: &str, value: &str) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let result = path().ok_or_else(|| "no config directory".to_string()).and_then(|p| {
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let old = std::fs::read_to_string(&p).unwrap_or_default();
        // Write to a temporary file, then rename: never leaves a half-written file.
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, set(&old, key, value)).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &p).map_err(|e| e.to_string())
    });
    if let Err(e) = result {
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("mviz: could not save settings: {e}");
        }
    }
}

/// The saved volume (0..=1), if there is one.
pub fn load_volume() -> Option<f32> {
    parse_volume(&load("volume")?)
}

fn parse_volume(v: &str) -> Option<f32> {
    v.parse::<f32>().ok().filter(|v| v.is_finite()).map(|v| v.clamp(0.0, 1.0))
}

pub fn save_volume(volume: f32) {
    save("volume", &format!("{volume:.2}"));
}

/// The last file or folder opened, if it still exists.
pub fn load_recent() -> Option<PathBuf> {
    load("recent").map(PathBuf::from).filter(|p| p.exists())
}

pub fn save_recent(path: &std::path::Path) {
    if let Some(p) = path.canonicalize().ok().and_then(|p| p.to_str().map(str::to_string)) {
        save("recent", &p);
    }
}

/// The last capture source (its PulseAudio name).
pub fn load_capture() -> Option<String> {
    load("capture").filter(|s| !s.is_empty())
}

pub fn save_capture(id: &str) {
    save("capture", id);
}

#[cfg(test)]
mod tests {
    use super::{get, parse_volume, set};

    #[test]
    fn reads_volume() {
        assert_eq!(get("volume=0.65\n", "volume").as_deref().and_then(parse_volume), Some(0.65));
        assert_eq!(get("other=1\n volume = 0.3 \n", "volume").as_deref().and_then(parse_volume), Some(0.3));
        assert_eq!(parse_volume("2.0"), Some(1.0));
        assert_eq!(parse_volume("abc"), None);
        assert_eq!(get("", "volume"), None);
    }

    #[test]
    fn set_keeps_other_keys() {
        let t = set("volume=0.50\nrecent=/a\n", "volume", "0.70");
        assert_eq!(get(&t, "volume").as_deref(), Some("0.70"));
        assert_eq!(get(&t, "recent").as_deref(), Some("/a"));
        assert_eq!(t.lines().count(), 2);
    }
}
