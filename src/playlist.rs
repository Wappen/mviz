//! A folder of songs, played in order (and from the start again after the
//! last). Song lengths are read from the files' headers in the background.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Extensions mviz can play (what Symphonia decodes).
pub const AUDIO: &[&str] = &["mp3", "flac", "ogg", "oga", "wav", "wave", "aif", "aiff", "aifc", "caf", "m4a", "mp4", "aac", "alac", "mka", "mkv", "webm"];

pub fn is_audio(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| AUDIO.contains(&e.to_ascii_lowercase().as_str()))
}

pub struct Playlist {
    /// The folder's name.
    pub name: String,
    pub songs: Vec<PathBuf>,
    /// Song lengths in seconds as they are read (`None`: not yet, or unknown).
    pub lengths: Arc<Mutex<Vec<Option<f64>>>>,
}

impl Playlist {
    /// The audio files directly in `dir`, in natural order ("2" before "10").
    pub fn scan(dir: &Path) -> Result<Playlist, String> {
        let entries = std::fs::read_dir(dir).map_err(|e| format!("cannot open {}: {e}", dir.display()))?;
        let mut songs: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file() && is_audio(p)).collect();
        songs.sort_by(|a, b| natural(&file_name(a), &file_name(b)));
        let name = file_name(dir);
        if songs.is_empty() {
            return Err(format!("no audio files in {name}"));
        }
        let lengths = Arc::new(Mutex::new(vec![None; songs.len()]));
        let (paths, out) = (songs.clone(), lengths.clone());
        std::thread::spawn(move || {
            for (i, p) in paths.iter().enumerate() {
                let len = crate::audio::Source::open(p).ok().and_then(|s| Some(s.frames? as f64 / s.rate as f64));
                out.lock().unwrap()[i] = len;
            }
        });
        Ok(Playlist { name, songs, lengths })
    }

    pub fn len(&self) -> usize {
        self.songs.len()
    }

    pub fn next(&self, i: usize) -> usize {
        (i + 1) % self.len()
    }

    pub fn prev(&self, i: usize) -> usize {
        (i + self.len() - 1) % self.len()
    }

    /// Song names (file names without extension).
    pub fn names(&self) -> Vec<String> {
        self.songs.iter().map(|p| crate::song_name(p)).collect()
    }
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
}

/// Compares names with runs of digits as numbers, ignoring case.
pub fn natural(a: &str, b: &str) -> Ordering {
    let (mut a, mut b) = (a.chars().peekable(), b.chars().peekable());
    loop {
        match (a.peek().copied(), b.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, _) => return Ordering::Less,
            (_, None) => return Ordering::Greater,
            (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let num = |it: &mut std::iter::Peekable<std::str::Chars>| {
                    let mut s = String::new();
                    while let Some(c) = it.peek().copied().filter(char::is_ascii_digit) {
                        s.push(c);
                        it.next();
                    }
                    s
                };
                let (na, nb) = (num(&mut a), num(&mut b));
                let (ta, tb) = (na.trim_start_matches('0'), nb.trim_start_matches('0'));
                let o = ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb));
                if o != Ordering::Equal {
                    return o;
                }
            }
            (Some(x), Some(y)) => {
                let o = x.to_lowercase().cmp(y.to_lowercase());
                if o != Ordering::Equal {
                    return o;
                }
                a.next();
                b.next();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::natural;

    #[test]
    fn sorts_numbers_naturally() {
        let mut v = vec!["10 Ten.mp3", "2 Two.mp3", "1 one.mp3", "B.mp3", "a.mp3", "02b.mp3"];
        v.sort_by(|a, b| natural(a, b));
        assert_eq!(v, ["1 one.mp3", "2 Two.mp3", "02b.mp3", "10 Ten.mp3", "a.mp3", "B.mp3"]);
    }
}
