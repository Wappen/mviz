//! The desktop's own file dialog (through the XDG desktop portal), run on
//! its own thread so the window keeps animating while it is open.

use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use winit::window::Window;

pub enum Picked {
    File(PathBuf),
    Folder(PathBuf),
}

/// Opens a dialog for a file or a folder; the answer (`None`: cancelled)
/// arrives on the returned channel.
pub fn open(window: &Arc<Window>, folder: bool, start: Option<PathBuf>) -> mpsc::Receiver<Option<Picked>> {
    let (tx, rx) = mpsc::channel();
    let window = window.clone();
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new().set_parent(&*window);
        if let Some(dir) = start.as_ref().and_then(|p| if p.is_dir() { Some(p.as_path()) } else { p.parent() }) {
            dialog = dialog.set_directory(dir);
        }
        let picked = if folder {
            dialog.set_title("Open folder").pick_folder().map(Picked::Folder)
        } else {
            dialog
                .set_title("Open file")
                .add_filter("Audio", crate::playlist::AUDIO)
                .pick_file()
                .map(Picked::File)
        };
        let _ = tx.send(picked);
    });
    rx
}
