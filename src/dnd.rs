//! Files dropped onto the window, on Wayland.
//!
//! winit only reports dropped files on X11, so on Wayland mviz listens for
//! them itself: it shares the window's Wayland connection, gets its own data
//! device for the seat, accepts drags that carry files (`text/uri-list`) and
//! reads the dropped paths. This runs on its own thread with its own event
//! queue; winit's handling of the window is untouched.

use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Read;
use std::os::unix::ffi::OsStringExt;
use std::os::fd::AsFd;
use std::path::PathBuf;
use std::sync::mpsc;
use wayland_client::backend::{Backend, ObjectId};
use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::wl_data_device::{self, WlDataDevice};
use wayland_client::protocol::wl_data_device_manager::{DndAction, WlDataDeviceManager};
use wayland_client::protocol::wl_data_offer::{self, WlDataOffer};
use wayland_client::protocol::{wl_registry, wl_seat::WlSeat};
use wayland_client::{event_created_child, Connection, Dispatch, Proxy, QueueHandle};
use winit::window::Window;

const URI_LIST: &str = "text/uri-list";

/// Starts listening for drops on `window`. Returns `None` when the window is
/// not on Wayland (X11 reports drops through winit) or setup fails.
pub fn listen(window: &Window) -> Option<mpsc::Receiver<PathBuf>> {
    let RawDisplayHandle::Wayland(d) = window.display_handle().ok()?.as_raw() else { return None };
    // SAFETY: the display stays valid while the window exists, i.e. for the
    // whole run of mviz (the window is never closed before exit).
    let backend = unsafe { Backend::from_foreign_display(d.display.as_ptr().cast()) };
    let conn = Connection::from_backend(backend);
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("dnd".into())
        .spawn(move || {
            if let Err(e) = run(conn, tx) {
                eprintln!("mviz: drag and drop unavailable: {e}");
            }
        })
        .ok()?;
    Some(rx)
}

fn run(conn: Connection, tx: mpsc::Sender<PathBuf>) -> Result<(), String> {
    let (globals, mut queue) = registry_queue_init::<State>(&conn).map_err(|e| e.to_string())?;
    let qh = queue.handle();
    let manager: WlDataDeviceManager = globals.bind(&qh, 1..=3, ()).map_err(|e| e.to_string())?;
    let seat: WlSeat = globals.bind(&qh, 1..=1, ()).map_err(|e| e.to_string())?;
    manager.get_data_device(&seat, &qh, ());
    let mut state = State { conn: conn.clone(), tx, mimes: HashMap::new(), current: None };
    loop {
        queue.blocking_dispatch(&mut state).map_err(|e| e.to_string())?;
    }
}

struct State {
    conn: Connection,
    tx: mpsc::Sender<PathBuf>,
    /// The types each pending offer can provide.
    mimes: HashMap<ObjectId, Vec<String>>,
    /// The offer being dragged over the window, if it carries files.
    current: Option<WlDataOffer>,
}

impl Dispatch<WlDataDevice, ()> for State {
    fn event(state: &mut Self, _: &WlDataDevice, event: wl_data_device::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        use wl_data_device::Event;
        match event {
            Event::DataOffer { id } => {
                state.mimes.insert(id.id(), Vec::new());
            }
            Event::Enter { serial, id: Some(offer), .. } => {
                let files = state.mimes.get(&offer.id()).is_some_and(|m| m.iter().any(|m| m == URI_LIST));
                if files {
                    offer.accept(serial, Some(URI_LIST.into()));
                    if offer.version() >= 3 {
                        offer.set_actions(DndAction::Copy, DndAction::Copy);
                    }
                    state.current = Some(offer);
                } else {
                    offer.accept(serial, None);
                    state.forget(offer);
                }
            }
            Event::Leave => {
                if let Some(offer) = state.current.take() {
                    state.forget(offer);
                }
            }
            Event::Drop => {
                if let Some(offer) = state.current.take() {
                    for path in state.receive(&offer) {
                        let _ = state.tx.send(path);
                    }
                    if offer.version() >= 3 {
                        offer.finish();
                    }
                    state.forget(offer);
                }
            }
            Event::Selection { id: Some(offer) } => state.forget(offer),
            _ => {}
        }
    }

    event_created_child!(State, WlDataDevice, [
        wl_data_device::EVT_DATA_OFFER_OPCODE => (WlDataOffer, ()),
    ]);
}

impl State {
    fn forget(&mut self, offer: WlDataOffer) {
        self.mimes.remove(&offer.id());
        offer.destroy();
    }

    /// Reads the dropped file list from the drag source.
    fn receive(&self, offer: &WlDataOffer) -> Vec<PathBuf> {
        let Ok((mut reader, writer)) = std::io::pipe() else { return Vec::new() };
        offer.receive(URI_LIST.into(), writer.as_fd());
        drop(writer);
        let _ = self.conn.flush();
        let mut list = String::new();
        let _ = reader.read_to_string(&mut list);
        parse_uri_list(&list)
    }
}

impl Dispatch<WlDataOffer, ()> for State {
    fn event(state: &mut Self, offer: &WlDataOffer, event: wl_data_offer::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        if let wl_data_offer::Event::Offer { mime_type } = event {
            state.mimes.entry(offer.id()).or_default().push(mime_type);
        }
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
    fn event(_: &mut Self, _: &wl_registry::WlRegistry, _: wl_registry::Event, _: &GlobalListContents, _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<WlSeat, ()> for State {
    fn event(_: &mut Self, _: &WlSeat, _: <WlSeat as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<WlDataDeviceManager, ()> for State {
    fn event(_: &mut Self, _: &WlDataDeviceManager, _: <WlDataDeviceManager as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

/// The local files in a `text/uri-list` (`file:///…`, percent-encoded).
fn parse_uri_list(list: &str) -> Vec<PathBuf> {
    list.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.strip_prefix("file://"))
        // Skip a host name ("file://host/path"); local paths start with '/'.
        .filter_map(|l| l.find('/').map(|i| &l[i..]))
        .map(|p| PathBuf::from(OsString::from_vec(percent_decode(p))))
        .collect()
}

fn percent_decode(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_file_uris() {
        let list = "# comment\r\nfile:///home/me/Music/My%20Song%20%C3%A4.mp3\r\nhttps://example.com/x.mp3\r\nfile://host/tmp/a.flac\r\nfile:///b%2\r\n";
        let paths = parse_uri_list(list);
        assert_eq!(paths, [PathBuf::from("/home/me/Music/My Song ä.mp3"), PathBuf::from("/tmp/a.flac"), PathBuf::from("/b%2")]);
    }
}
