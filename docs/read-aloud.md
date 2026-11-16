# Read Aloud (local text-to-speech)

Handy can read text back to you with a fully local neural voice. Select text in
any app and press the **Read Aloud** shortcut (default `Option+Shift+A` on
macOS, `Ctrl+Shift+A` on Windows, `Ctrl+Alt+A` on Linux), or use the speaker
button on any History entry.

## How it works

1. On macOS it reads `AXSelectedText` from the focused element through the
   Accessibility API, so the clipboard is never touched. Windows and Linux use
   a temporary copy chord and restore the previous clipboard. Either way it
   only reads an actual selection — it never speaks whatever is already sitting
   on the clipboard. If the macOS build has not been granted Accessibility
   permission, it falls back to the temporary-copy path until permission is
   granted.
2. The text is synthesized by [Supertonic 3](https://huggingface.co/Supertone/supertonic-3)
   through ONNX Runtime, in-process. Nothing is uploaded.
3. The 44.1 kHz mono result plays through the same output device as Handy's
   sound effects (Settings → Sound → Output Device).

Pressing the shortcut again while Handy is speaking stops playback. Audio is
synthesized on a worker thread, so the UI and the recording shortcuts stay
responsive.

### Latency

The engine is prewarmed in the background shortly after startup (and when Read
Aloud is enabled or the model is downloaded). Synthesis runs at denoising
quality 5, and the text is split into 160-character chunks: the first chunk is
played while the rest is still being synthesized. The output device is opened
concurrently with the first chunk. In a debug `tauri dev` build this brings the
first audio down to roughly 0.7–1.2 s on an M2 Pro, instead of ~2 s; a release
build is faster still. `Cargo.toml` pins `opt-level = 3` for the inference
crates in the dev profile so `tauri dev` does not pay unoptimized Rust inference
costs. While audio plays, the dictation pill reappears with a speaker
glyph in place of the microphone dot, an animated waveform, and the same close
button (which stops playback). It follows the existing Overlay style and
position settings.

## Model

The model is not bundled. The first use downloads about 401 MB from the pinned
Hugging Face revision:

```
          onnx/vector_estimator.onnx   245 MB
          onnx/vocoder.onnx             97 MB
          onnx/text_encoder.onnx        35 MB
          onnx/duration_predictor.onnx 3.5 MB
          onnx/tts.json, unicode_indexer.json
          voice_styles/{F1..F5,M1..M5}.json
```

Files land in `<app data>/models/supertonic3/`. Downloads use the pinned
Hugging Face revision over HTTPS and can be cancelled.

### Licenses

- Rust inference code: MIT (vendored at `src-tauri/vendor/supertonic3-tts`).
- Model weights: **BigScience Open RAIL-M**. The Settings page shows the
  license and requires acceptance before downloading. Attachment A of that
  license lists use-based restrictions; redistribution of the weights must
  carry the license.

VoiceStudio (AGPL-3.0) was used as a reference for the engine and license-gate
design, but its code is not copied into Handy (MIT); the inference crate is a
separate MIT project.

## Implementation

- `src-tauri/src/tts/mod.rs` — model download, engine lifecycle, synthesis and
  playback. The engine is loaded lazily on the first speak and unloaded when the
  model is deleted.
- `src-tauri/src/commands/tts.rs` — Tauri commands used by the settings page
  and History.
- `src-tauri/src/actions.rs` — `ReadAloudAction` (shortcut, selection capture,
  toggle-stop).
- `src/components/settings/tts/TtsSettings.tsx` — Settings → Read Aloud.
- `src-tauri/vendor/supertonic3-tts` — vendored MIT inference crate. Upstream
  depends on `cpal 0.15` for CLI playback, which conflicts with Handy's
  `cpal 0.18` (`alsa-sys` links conflict); the vendored copy drops the CLI,
  playback and microphone-capture code and keeps inference only.

## Building

> **macOS:** the prebuilt ONNX Runtime static library used by `ort` reports a
> minimum deployment target of macOS 13.4. Handy's other features still support
> older macOS releases, but a build that includes Read Aloud will need 13.4+.
> Linux and Windows build normally; ONNX Runtime is linked statically.

No new system dependency beyond Handy's existing CMake/Xcode toolchain. The
`ort` crate downloads a prebuilt ONNX Runtime static library at build time. In a
sandbox or offline build, point it at a local runtime:

```bash
curl -L -o ort.tar.lzma2 \
  "https://cdn.pyke.io/0/pyke:ort-rs/ms@1.24.2/aarch64-apple-darwin.tar.lzma2"
python3 - <<'PY'
import lzma, pathlib
with lzma.open('ort.tar.lzma2', format=lzma.FORMAT_RAW,
               filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 1 << 26}]) as f:
    pathlib.Path('ort.tar').write_bytes(f.read())
PY
mkdir -p ort && tar -xf ort.tar -C ort
ORT_LIB_LOCATION="$PWD/ort" cargo build
```
