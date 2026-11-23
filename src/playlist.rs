//! A folder of songs, played in order (and from the start again after the
//! last). The order is by name, newest first, or shuffled. Song lengths are
//! read from the files' headers in the background.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Extensions mviz can play (what Symphonia decodes).
pub const AUDIO: &[&str] = &["mp3", "flac", "ogg", "oga", "wav", "wave", "aif", "aiff", "aifc", "caf", "m4a", "mp4", "aac", "alac", "mka", "mkv", "webm"];

pub fn is_audio(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| AUDIO.contains(&e.to_ascii_lowercase().as_str()))
}

/// What dropping some files or folders onto the window would play: the
/// first song, or folder with songs in it, among them.
#[derive(Clone, Debug, PartialEq)]
pub enum Droppable {
    Song(PathBuf),
    /// A folder and how many songs it has.
    Folder(PathBuf, usize),
    /// Nothing mviz can play.
    Nothing,
    /// Not known yet (the drag's source tells only on the drop).
    Unknown,
}

impl Droppable {
    pub fn of(paths: &[PathBuf]) -> Droppable {
        if paths.is_empty() {
            return Droppable::Unknown;
        }
        for p in paths {
            if p.is_dir() {
                let n = std::fs::read_dir(p).map_or(0, |d| d.filter_map(|e| e.ok()).filter(|e| is_audio(&e.path()) && e.path().is_file()).count());
                if n > 0 {
                    return Droppable::Folder(p.clone(), n);
                }
            } else if p.is_file() && is_audio(p) {
                return Droppable::Song(p.clone());
            }
        }
        Droppable::Nothing
    }

    /// The file or folder to open, if any.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Droppable::Song(p) | Droppable::Folder(p, _) => Some(p),
            _ => None,
        }
    }
}

/// The order a folder plays in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sort {
    /// By file name, numbers as numbers ("2" before "10").
    Name,
    /// Most recently changed files first.
    Newest,
    Shuffle,
}

impl Sort {
    pub const ALL: [Sort; 3] = [Sort::Name, Sort::Newest, Sort::Shuffle];

    /// As shown in the playlist.
    pub fn label(self) -> &'static str {
        match self {
            Sort::Name => "by name",
            Sort::Newest => "newest first",
            Sort::Shuffle => "shuffled",
        }
    }

    /// The next mode (the playlist cycles through them).
    pub fn next(self) -> Sort {
        let i = Self::ALL.iter().position(|&s| s == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// Name in the settings file.
    pub fn key(self) -> &'static str {
        match self {
            Sort::Name => "name",
            Sort::Newest => "newest",
            Sort::Shuffle => "shuffle",
        }
    }

    pub fn from_key(k: &str) -> Option<Sort> {
        Self::ALL.into_iter().find(|s| s.key() == k)
    }
}

#[derive(Clone)]
pub struct Playlist {
    /// The folder's name.
    pub name: String,
    /// The songs by name; `order` says which plays when.
    songs: Vec<PathBuf>,
    modified: Vec<SystemTime>,
    /// Song lengths in seconds as they are read (`None`: not yet, or
    /// unknown), indexed like `songs`.
    lengths: Arc<Mutex<Vec<Option<f64>>>>,
    /// Play position → song.
    order: Vec<usize>,
    pub sort: Sort,
}

impl Playlist {
    /// The audio files directly in `dir`, in `sort` order.
    pub fn scan(dir: &Path, sort: Sort) -> Result<Playlist, String> {
        let entries = std::fs::read_dir(dir).map_err(|e| format!("cannot open {}: {e}", dir.display()))?;
        let mut songs: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file() && is_audio(p)).collect();
        songs.sort_by(|a, b| natural(&file_name(a), &file_name(b)));
        let name = file_name(dir);
        if songs.is_empty() {
            return Err(format!("no audio files in {name}"));
        }
        let modified = songs.iter().map(|p| p.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH)).collect();
        let lengths = Arc::new(Mutex::new(vec![None; songs.len()]));
        let (paths, out) = (songs.clone(), lengths.clone());
        std::thread::spawn(move || {
            for (i, p) in paths.iter().enumerate() {
                let len = crate::audio::Source::open(p).ok().and_then(|s| Some(s.frames? as f64 / s.rate as f64));
                out.lock().unwrap()[i] = len;
            }
        });
        let pl = Playlist { name, songs, modified, lengths, order: Vec::new(), sort: Sort::Name };
        Ok(pl.sorted(sort))
    }

    /// The same songs in `sort` order (shuffling again each time).
    pub fn sorted(&self, sort: Sort) -> Playlist {
        let mut order: Vec<usize> = (0..self.songs.len()).collect();
        match sort {
            Sort::Name => {}
            Sort::Newest => order.sort_by(|&a, &b| self.modified[b].cmp(&self.modified[a])),
            Sort::Shuffle => shuffle(&mut order),
        }
        Playlist { order, sort, ..self.clone() }
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn next(&self, pos: usize) -> usize {
        (pos + 1) % self.len()
    }

    pub fn prev(&self, pos: usize) -> usize {
        (pos + self.len() - 1) % self.len()
    }

    /// The song at play position `pos`.
    pub fn path(&self, pos: usize) -> &Path {
        &self.songs[self.order[pos]]
    }

    /// Where `other`'s position `pos` (same folder, other order) is here.
    pub fn position_of(&self, other: &Playlist, pos: usize) -> usize {
        let song = other.order[pos];
        self.order.iter().position(|&s| s == song).unwrap_or(0)
    }

    /// Song names (file names without extension), in play order.
    pub fn names(&self) -> Vec<String> {
        self.order.iter().map(|&i| crate::song_name(&self.songs[i])).collect()
    }

    /// Song lengths as far as known, in play order.
    pub fn lengths(&self) -> Vec<Option<f64>> {
        let l = self.lengths.lock().unwrap();
        self.order.iter().map(|&i| l[i]).collect()
    }
}

/// Fisher–Yates with a small xorshift generator seeded from the clock
/// (good enough to mix a playlist).
fn shuffle(v: &mut [usize]) {
    let mut x = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0x9e37_79b9, |d| d.as_nanos() as u64) | 1;
    for i in (1..v.len()).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v.swap(i, (x % (i as u64 + 1)) as usize);
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
    use super::*;

    #[test]
    fn a_drop_plays_the_first_song_or_folder_with_songs() {
        let dir = std::env::temp_dir().join(format!("mviz-drop-{}", std::process::id()));
        let songs = dir.join("songs");
        let empty = dir.join("empty");
        std::fs::create_dir_all(&songs).unwrap();
        std::fs::create_dir_all(&empty).unwrap();
        for f in ["songs/a.mp3", "songs/b.flac", "songs/notes.txt", "c.ogg", "d.pdf"] {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        assert_eq!(Droppable::of(&[]), Droppable::Unknown);
        assert_eq!(Droppable::of(&[dir.join("d.pdf"), empty.clone()]), Droppable::Nothing);
        assert_eq!(Droppable::of(&[dir.join("d.pdf"), dir.join("c.ogg")]), Droppable::Song(dir.join("c.ogg")));
        assert_eq!(Droppable::of(&[empty, songs.clone(), dir.join("c.ogg")]), Droppable::Folder(songs, 2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sorts_numbers_naturally() {
        let mut v = vec!["10 Ten.mp3", "2 Two.mp3", "1 one.mp3", "B.mp3", "a.mp3", "02b.mp3"];
        v.sort_by(|a, b| natural(a, b));
        assert_eq!(v, ["1 one.mp3", "2 Two.mp3", "02b.mp3", "10 Ten.mp3", "a.mp3", "B.mp3"]);
    }

    fn sample() -> Playlist {
        let t = |s| SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(s);
        Playlist {
            name: "x".into(),
            songs: ["a", "b", "c", "d"].iter().map(PathBuf::from).collect(),
            modified: vec![t(10), t(40), t(20), t(30)],
            lengths: Arc::new(Mutex::new(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)])),
            order: vec![0, 1, 2, 3],
            sort: Sort::Name,
        }
    }

    #[test]
    fn newest_first_and_positions_carry_over() {
        let by_name = sample();
        let newest = by_name.sorted(Sort::Newest);
        assert_eq!(newest.names(), ["b", "d", "c", "a"]);
        assert_eq!(newest.lengths(), [Some(2.0), Some(4.0), Some(3.0), Some(1.0)]);
        // "c" is third by name and third newest; "a" moves from first to last.
        assert_eq!(newest.position_of(&by_name, 2), 2);
        assert_eq!(newest.position_of(&by_name, 0), 3);
    }

    #[test]
    fn shuffle_keeps_every_song_once() {
        let mut names = sample().sorted(Sort::Shuffle).names();
        names.sort();
        assert_eq!(names, ["a", "b", "c", "d"]);
    }

    #[test]
    fn sort_modes_cycle_and_round_trip() {
        assert_eq!(Sort::Name.next().next().next(), Sort::Name);
        for s in Sort::ALL {
            assert_eq!(Sort::from_key(s.key()), Some(s));
        }
    }
}
