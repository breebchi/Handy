# Read Aloud (local text-to-speech)

On macOS, grant Handy Accessibility permission, select text in an app that
exposes its selection, and press **Read Aloud** (default `Option+Shift+A`).
Handy reads the selection through Accessibility and leaves the clipboard
untouched. You can also use the speaker button on a History entry. Selection,
History, and voice previews use the Supertonic 3 model.

The shared playback session uses Handy’s selected output device and volume.
The speaking pill provides a Stop button when the dictation overlay is free.
The Settings → Read Aloud Stop button also works while dictation owns the
overlay. Stop discards the session’s queued audio.

## Stream an agent reply

On Apple Silicon, turn on **Allow external agent speech** in Settings →
Read Aloud → External Agent Speech. This setting defaults to off. Agent
replies use Phonon’s built-in English voice. The Supertonic 3 voice, language,
and speed controls apply to Selection and History. Any process running under your Mac account can send speech while
it is enabled. Incoming text stays local and is not written to Handy’s log.

Pass UTF-8 text to `handy speak --stdin`, once per reply. Supply a space after
sentence punctuation or a newline to make a phrase ready before EOF. Handy
also splits long input at roughly 160 UTF-8 bytes, and reads the final phrase
when the input closes. The command returns after playback drains. It returns
an error if the request is rejected, fails, or is stopped. The IPC and audio
queues are bounded, so writes can block during longer replies.

```bash
# A paced transport check. Speech should begin before the last write.
{ printf 'The first phrase is ready. '; sleep 3; printf 'This is the final phrase'; } |
  handy speak --stdin

# From another terminal, or use the speaking pill's Stop button.
handy stop-speaking
```

### Claude Code example

Claude Code emits text deltas in its streaming JSON output. The filter forwards
only the incremental user-facing text blocks.

```bash
claude -p 'Explain how tides work in five short sentences.' \
  --output-format stream-json --verbose --include-partial-messages |
  jq --unbuffered -j '
    select(.type == "stream_event" and
           .event.type == "content_block_delta" and
           .event.delta.type == "text_delta") |
    .event.delta.text' |
  handy speak --stdin
```

The agent prototype uses [Phonon 31M English](https://huggingface.co/gradium/phonon-31m-en)
via [ptts](https://github.com/gradium-ai/xn-ptts/tree/main/ptts). The first
reply downloads the pinned 64.5 MB Q8 checkpoint and accompanying tokenizer
and default voice. Phonon streams PCM as it synthesizes each phrase.
Supertonic 3 remains the baseline and synthesizes one bounded phrase at a
time. Handy’s development logs report first phrase to first PCM render.

## Prototype measurements

These development-build measurements were collected on an M2 Pro with the
built-in speakers selected. The start point for engine comparisons was the
first complete phrase, and the endpoint was the first PCM pulled by the audio
renderer.

| Trial                                                   | Observed                                                                                                                                                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Warm Phonon, paced input                                | 101 ms from phrase to first render, with first render 2.88 s before EOF. The final phrase drained after EOF. The fixture’s three-second input pause produced a 1.66 s render gap.          |
| Warm Phonon, live Claude Code                           | 82 text deltas. First render came 427 ms after the first delta, 109 ms after the first complete phrase, and 3.61 s before the last delta. One renderer gap over 50 ms was logged (102 ms). |
| Warm Supertonic 3, same opening phrase in Voice Preview | 634 ms from phrase to first render.                                                                                                                                                        |
| First Phonon reply, model download and load included    | 13.34 s from phrase to first render, after the fixture’s EOF.                                                                                                                              |
| Flooded stdin, then Stop                                | Peak of eight queued PCM buffers. The output stream closed 206 ms after launching the Stop command. The speech client exited with a stopped result.                                        |

These timings use the software renderer and output stream. Acoustic latency
depends on the output device.

## Supertonic 3 baseline model

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
> This draft is being validated on Apple Silicon. The pinned ONNX Runtime does
> not supply an Intel Mac prebuilt library.

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
