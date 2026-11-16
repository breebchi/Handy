# Vendored: supertonic3-tts 1.3.0

Upstream: https://github.com/DavidValin/supertonic3-tts (crates.io 1.3.0)
License: MIT (see `LICENSE`); the Supertonic 3 model weights are distributed
separately under BigScience Open RAIL-M.

Handy vendors the inference crate because upstream's CLI playback depends on
`cpal 0.15`, which links ALSA 0.3 (`alsa-sys`), while Handy's `rodio` uses
`cpal 0.18` / ALSA 0.4. Cargo permits only one package linking the native
`alsa` library, so the two cannot coexist on Linux.

This copy keeps `onnx`, `features` (log-mel helpers used by cloning), `helper`,
`clone`, `speaker` and the `TtsEngine` synthesis API. It removes the CLI
(`main.rs`), CPAL playback (`play_samples` / `play_wav`), microphone capture
(`record_audio`), and the examples/scripts that used them. Handy owns playback
through `rodio` in `src/tts/mod.rs`.
