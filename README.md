<p align="center">
  <img src="assets/mviz.svg" width="112" alt="mviz icon">
</p>

<h1 align="center">mviz</h1>

<p align="center">
  A looping music player that shows the <b>stereo sound field</b>:<br>
  every frequency of the music as light, placed where you hear it.
</p>

<p align="center">
  <img src="docs/demo.webp" width="800" alt="mviz playing a song: coloured glows across the stereo field, a warm bass carpet at the bottom">
</p>


## What you see

You are sitting in front of the speakers:

- **Left ↔ right** is where a sound sits between the speakers. Sound that is out
  of phase between the channels (stereo wideners, some reverbs) appears
  **beyond the speakers**, towards the window edges, which is where you hear it.
- **Bottom ↔ top** is pitch, on a musical (logarithmic) scale: every octave gets
  the same height.
- **Colour** is the note: every C has the same hue, every C♯ the next one, and
  so on.
- **Size and softness** grow with depth. The deepest bass spreads into a wide,
  flat **carpet** along the bottom, and the highs are small, round sparks.
- **Direct sounds are crisp; diffuse ones (reverb, wide pads) are larger and
  softer.** Every frequency bin is placed on its own, so instruments that share
  a range still appear side by side.
- **Afterglow:** light fades like a phosphor screen: fast at first, then slowly.
  Moving sounds leave trails, and overlapping light adds up like light.
- **Background:** the bass (below about 80 Hz) lights the background, each side
  with its own channel, tinted by the bass note.

The light is computed live from the audio: each frame analyses the music at the
playback position with FFTs at three resolutions (long windows for the bass,
short ones for the treble), crossfaded between, and places each frequency bin
by its left/right level, polarity and, at low frequencies, arrival time.

<p align="center">
  <img src="docs/screenshot.jpg" width="800" alt="A still frame of the sound field">
</p>

## Using it

```sh
mviz song.mp3
```

The song starts at once (about 40 ms after launch) and loops without a gap.

| Input | Action |
| --- | --- |
| Right-click or Space | Pause / resume |
| Scroll | Volume (remembered between runs) |
| Left-drag | Move the window (from fullscreen: back to a window, then move) |
| Double-click or F11 | Fullscreen on / off |
| Timeline at the bottom | Shown while the cursor is visible; click or drag to seek |
| Drop a file on the window | Play that file instead |
| Esc | Quit |

mviz also appears in the desktop's **media controls** (MPRIS): the Plasma media
widget, the lock screen, media keys, Bluetooth headphone buttons and KDE Connect
can play, pause, seek and change the volume. It shows the title, artist, album
and cover from the file's tags.

Supported formats: MP3, FLAC, Ogg Vorbis, WAV, AIFF, CAF, and AAC / ALAC in M4A
or MP4.

### Exporting a video

```sh
mviz --render song.mp3 clip.mp4 --start 88 --seconds 12 --size 1280x720
```

This renders the visualizer offscreen (same GPU pipeline as the window) at
exactly 60 frames per second of song time and lets `ffmpeg` add the matching
audio. The preview above was made this way.

## Building and installing

Linux only. You need:

- a recent stable **Rust** (tested with 1.94)
- **ALSA** development files (`alsa-devel` / `libasound2-dev`) and `pkg-config`.
  PipeWire and PulseAudio work through their ALSA plugins.
- a GPU with **Vulkan** or OpenGL ES support (rendering uses wgpu)
- `ffmpeg`, only for `--render`

```sh
cargo build --release

# install for your user
install -Dm755 target/release/mviz ~/.local/bin/mviz
install -Dm644 assets/mviz.svg ~/.local/share/icons/hicolor/scalable/apps/mviz.svg
install -Dm644 assets/mviz.desktop ~/.local/share/applications/mviz.desktop

# optional: make it the default player for common audio files
for t in audio/mpeg audio/flac audio/ogg audio/x-vorbis+ogg audio/x-wav audio/mp4 audio/aac; do
  xdg-mime default mviz.desktop "$t"
done
```

All visual settings (sizes, colours, fade time, stereo width, …) live in one
place, [`src/params.rs`](src/params.rs), each with a short description.

## License

mviz is released under the [MIT License](LICENSE). It includes the
[Noto Sans](https://notofonts.github.io/) font, which is under the
[SIL Open Font License 1.1](assets/NotoSans-LICENSE.txt).
