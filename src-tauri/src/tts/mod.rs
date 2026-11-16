//! Local text-to-speech ("Read Aloud").
//!
//! The engine is [Supertonic 3](https://huggingface.co/Supertone/supertonic-3),
//! run in-process through ONNX Runtime. The Rust inference crate is MIT; the
//! model weights are BigScience Open RAIL-M and are downloaded on demand, the
//! same way Handy's transcription models are. Nothing leaves the machine.
//!
//! The model is a directory of files rather than a single GGUF, so this module
//! owns its downloader instead of reusing the transcription `ModelManager`.

use crate::audio_feedback::open_output_sink;
use crate::notices;
use crate::settings;
use log::{debug, info, warn};
use serde::Serialize;
use specta::Type;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use supertonic3_tts::helper::{chunk_text, load_voice_style};
use supertonic3_tts::{Style, TtsEngine};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::Mutex as AsyncMutex;

/// HF repository and immutable revision. The revision is the Supertonic 3
/// release commit also pinned by VoiceStudio's Supertonic engine.
const MODEL_REPO: &str = "Supertone/supertonic-3";
const MODEL_REVISION: &str = "724fb5abbf5502583fb520898d45929e62f02c0b";

/// Files that must exist for the engine to load. `LICENSE` is downloaded too,
/// but a missing license file does not make the model unusable.
const MODEL_FILES: &[&str] = &[
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
    "voice_styles/F1.json",
    "voice_styles/F2.json",
    "voice_styles/F3.json",
    "voice_styles/F4.json",
    "voice_styles/F5.json",
    "voice_styles/M1.json",
    "voice_styles/M2.json",
    "voice_styles/M3.json",
    "voice_styles/M4.json",
    "voice_styles/M5.json",
];

const LICENSE_FILE: &str = "LICENSE";

/// Total payload of the 16 model files, used for the download progress bar.
/// (The license is ~15 KB and folded in there.)
const MODEL_TOTAL_BYTES: u64 = 401_291_751;

/// Preset voices shipped with the model. The frontend shows these as `F1`…
/// `M5`; the engine treats the name as the file stem in `voice_styles/`.
pub const VOICES: &[&str] = &["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"];

/// Languages Supertonic 3 speaks. `na` lets the model infer the language.
pub const LANGUAGES: &[&str] = &[
    "na", "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hi", "hr", "hu", "id",
    "it", "ja", "ko", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
];

/// Longest text accepted in one read-aloud request. Supertonic chunks long
/// input itself; this only stops a stray multi-megabyte selection from
/// allocating an hour of audio.
const MAX_TEXT_CHARS: usize = 20_000;

/// Flow-matching denoising steps. 8 is the model default; 5 keeps the voice
/// close enough while cutting the first-audio latency noticeably.
const VOICE_QUALITY: usize = 5;

/// Characters per synthesis chunk. The first chunk is played while the next
/// one is still being synthesized, so a smaller chunk means faster first audio.
const CHUNK_CHARS: usize = 160;

/// One-shot model download bookkeeping.
#[derive(Debug, Clone, Serialize, Type)]
pub struct TtsStatus {
    /// The Read Aloud setting is on.
    pub enabled: bool,
    /// The user accepted the Open RAIL-M model license.
    pub license_accepted: bool,
    /// Every required model file is present.
    pub model_downloaded: bool,
    pub model_downloading: bool,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    /// The ONNX sessions are loaded and ready.
    pub model_loaded: bool,
    /// A read is in progress (loading, synthesizing, or playing).
    pub speaking: bool,
}

struct Playback {
    /// Keeping the device sink alive keeps the output stream open while the
    /// player drains.
    _sink: rodio::MixerDeviceSink,
    player: Arc<rodio::Player>,
    generation: u64,
}

/// One streaming playback request: the engine and voice are resolved before
/// the worker starts, and the generation lets a stop cancel every stage.
struct PlaybackJob {
    engine: Arc<TtsEngine>,
    chunks: Vec<String>,
    style_path: PathBuf,
    language: String,
    speed: f32,
    volume: f32,
    generation: u64,
    request_started: std::time::Instant,
}

pub struct TtsManager {
    app: AppHandle,
    /// Absolute path to `<app data>/models/supertonic3`.
    model_dir: PathBuf,
    engine: AsyncMutex<Option<Arc<TtsEngine>>>,
    /// Serializes `speak` calls so two shortcuts can't interleave synthesis.
    speak_lock: AsyncMutex<()>,
    playback: Mutex<Option<Playback>>,
    engine_loaded: AtomicBool,
    downloading: AtomicBool,
    download_cancelled: AtomicBool,
    speaking: AtomicBool,
    /// The speaking pill is currently on screen.
    overlay_active: AtomicBool,
    /// Bumped by every stop/new request; a completed synthesis whose
    /// generation is stale is thrown away instead of played.
    generation: AtomicU64,
}

impl TtsManager {
    pub fn new(app: &AppHandle) -> Result<Self, String> {
        let model_dir = crate::portable::app_data_dir(app)
            .map_err(|e| format!("Failed to resolve app data directory: {e}"))?
            .join("models")
            .join("supertonic3");
        Ok(Self {
            app: app.clone(),
            model_dir,
            engine: AsyncMutex::new(None),
            speak_lock: AsyncMutex::new(()),
            playback: Mutex::new(None),
            engine_loaded: AtomicBool::new(false),
            downloading: AtomicBool::new(false),
            download_cancelled: AtomicBool::new(false),
            speaking: AtomicBool::new(false),
            overlay_active: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn status(&self) -> TtsStatus {
        let settings = settings::get_settings(&self.app);
        TtsStatus {
            enabled: settings.tts_enabled,
            license_accepted: settings.tts_license_accepted,
            model_downloaded: self.is_model_downloaded(),
            model_downloading: self.downloading.load(Ordering::SeqCst),
            downloaded_bytes: self.downloaded_bytes(),
            total_bytes: MODEL_TOTAL_BYTES,
            model_loaded: self.engine_loaded.load(Ordering::SeqCst),
            speaking: self.speaking.load(Ordering::SeqCst),
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    pub fn is_model_downloaded(&self) -> bool {
        MODEL_FILES.iter().all(|relative| {
            std::fs::metadata(self.model_dir.join(relative))
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
        })
    }

    fn downloaded_bytes(&self) -> u64 {
        let mut total = 0;
        for relative in MODEL_FILES.iter().chain(std::iter::once(&LICENSE_FILE)) {
            let path = self.model_dir.join(relative);
            if let Ok(metadata) = std::fs::metadata(path) {
                total += metadata.len();
            }
            // A file still in flight is counted through its `.part` sibling so
            // the progress bar moves during the 245 MB vector estimator.
            let part = self.model_dir.join(format!("{relative}.part"));
            if let Ok(metadata) = std::fs::metadata(part) {
                total += metadata.len();
            }
        }
        total
    }

    /// Stop playback and invalidate any synthesis in flight.
    pub fn stop(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        if let Some(playback) = self
            .playback
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            playback.player.stop();
        }
        self.set_speaking(false);
        if self.overlay_active.swap(false, Ordering::SeqCst) {
            crate::overlay::hide_speaking_overlay(&self.app);
        }
    }

    fn set_speaking(&self, speaking: bool) {
        if self.speaking.swap(speaking, Ordering::SeqCst) != speaking {
            if let Err(error) = self.app.emit("tts-status", self.status()) {
                warn!("Failed to emit TTS status: {error}");
            }
        }
    }

    /// Run one read-aloud request. `stop()` can cancel both synthesis and
    /// playback; a stale synthesis is never played.
    ///
    /// The overlay appears as soon as the request is accepted, the engine is
    /// loaded ahead of time by [`TtsManager::prewarm`], and the first sentence
    /// is played while the rest of the text is still being synthesized.
    pub async fn speak(
        self: &Arc<Self>,
        text: String,
        voice: String,
        language: String,
        speed: f32,
    ) -> Result<(), String> {
        log::debug!("TTS speak requested: {} chars", text.chars().count());
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err("There is no text to read.".to_string());
        }
        if text.chars().count() > MAX_TEXT_CHARS {
            return Err(format!(
                "The text is too long to read aloud (limit {MAX_TEXT_CHARS} characters)."
            ));
        }
        if !VOICES.contains(&voice.as_str()) {
            return Err(format!("Unknown voice '{voice}'."));
        }
        if !LANGUAGES.contains(&language.as_str()) {
            return Err(format!("Unsupported language '{language}'."));
        }
        if !self.is_model_downloaded() {
            return Err("The Read Aloud model is not downloaded yet.".to_string());
        }

        let _speak_guard = self.speak_lock.lock().await;
        self.stop();
        // `stop` bumped the generation; this request owns the new one. A stop
        // that arrives while synthesis runs bumps it again and discards the
        // result.
        let generation = self.generation.load(Ordering::SeqCst);
        let settings = settings::get_settings(&self.app);
        let request_started = std::time::Instant::now();

        // Give the shortcut immediate feedback even while the engine loads or
        // the first sentence synthesizes.
        self.begin_speaking(generation, &settings);

        info!(
            "TTS: synthesizing {} chars (voice={voice}, lang={language}, speed={speed:.2}, quality={VOICE_QUALITY})",
            text.chars().count()
        );
        let engine = match self.ensure_engine().await {
            Ok(engine) => engine,
            Err(error) => {
                self.fail_start(generation);
                return Err(error);
            }
        };
        if generation != self.generation.load(Ordering::SeqCst) {
            return Ok(());
        }

        let style_path = self
            .model_dir
            .join("voice_styles")
            .join(format!("{voice}.json"));
        if !style_path.is_file() {
            self.fail_start(generation);
            return Err(format!("Voice '{voice}' is missing from the model."));
        }
        let chunks: Vec<String> = chunk_text(&text, Some(CHUNK_CHARS))
            .into_iter()
            .filter(|chunk| !chunk.trim().is_empty())
            .collect();
        if chunks.is_empty() {
            self.fail_start(generation);
            return Err("There is no text to read.".to_string());
        }

        let manager = self.clone();
        std::thread::spawn(move || {
            if let Err(error) = manager.play_chunks(PlaybackJob {
                engine,
                chunks,
                style_path,
                language,
                speed,
                volume: settings.tts_volume,
                generation,
                request_started,
            }) {
                manager.fail_playback(generation, error);
            }
        });
        Ok(())
    }

    /// Show the reading pill and mark the request in flight before the model
    /// work starts. `is_speaking` therefore doubles as "a read is in progress"
    /// for the shortcut toggle.
    fn begin_speaking(&self, generation: u64, settings: &crate::settings::AppSettings) {
        self.set_speaking(true);
        if generation != self.generation.load(Ordering::SeqCst) {
            return;
        }
        let recording = self
            .app
            .try_state::<Arc<crate::managers::audio::AudioRecordingManager>>()
            .is_some_and(|manager| manager.is_recording());
        if !recording && settings.overlay_style != crate::settings::OverlayStyle::None {
            crate::overlay::show_speaking_overlay(&self.app);
            self.overlay_active.store(true, Ordering::SeqCst);
        }
    }

    /// Roll back the optimistic speaking state when setup fails before playback.
    fn fail_start(&self, generation: u64) {
        if generation != self.generation.load(Ordering::SeqCst) {
            return;
        }
        self.set_speaking(false);
        if self.overlay_active.swap(false, Ordering::SeqCst) {
            crate::overlay::hide_speaking_overlay(&self.app);
        }
    }

    async fn ensure_engine(&self) -> Result<Arc<TtsEngine>, String> {
        let mut guard = self.engine.lock().await;
        if let Some(engine) = guard.as_ref() {
            return Ok(engine.clone());
        }
        if !self.is_model_downloaded() {
            return Err("The Read Aloud model is not downloaded yet.".to_string());
        }

        let onnx_dir = self.model_dir.join("onnx");
        let base_dir = self.model_dir.clone();
        info!("TTS: loading Supertonic 3 from {}", onnx_dir.display());
        let engine = tauri::async_runtime::spawn_blocking(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| format!("Failed to create TTS runtime: {e}"))?;
            runtime
                .block_on(TtsEngine::new(onnx_dir, base_dir, false))
                .map_err(|e| format!("Failed to load the Read Aloud model: {e}"))
        })
        .await
        .map_err(|e| format!("TTS load task failed: {e}"))??;

        let engine = Arc::new(engine);
        *guard = Some(engine.clone());
        self.engine_loaded.store(true, Ordering::SeqCst);
        info!("TTS: engine ready");
        Ok(engine)
    }

    /// Synthesize chunks in order and feed each one to the player as soon as it
    /// is ready. The player starts on the first append, so the first sentence
    /// is heard while later chunks are still in the model.
    fn play_chunks(self: &Arc<Self>, job: PlaybackJob) -> Result<(), String> {
        let PlaybackJob {
            engine,
            chunks,
            style_path,
            language,
            speed,
            volume,
            generation,
            request_started,
        } = job;
        if generation != self.generation.load(Ordering::SeqCst) {
            return Ok(());
        }
        let Some(first_chunk) = chunks.first().cloned() else {
            return Err("There is no text to read.".to_string());
        };

        let settings = settings::get_settings(&self.app);
        let style_path = style_path.to_string_lossy().to_string();
        let style = load_voice_style(&[style_path], false)
            .map_err(|e| format!("Failed to load the voice style: {e}"))?;

        // Opening the output device costs ~200 ms on a cold Bluetooth device.
        // Do it while the first chunk is synthesized instead of serially.
        let selected_device = settings.selected_output_device.clone();
        let sink_handle = std::thread::spawn(move || {
            open_output_sink(selected_device).map_err(|e| e.to_string())
        });

        let synth_started = std::time::Instant::now();
        let first_samples = match synthesize_chunk(&engine, &first_chunk, &style, &language, speed)
        {
            Ok(samples) => samples,
            Err(error) => {
                let _ = sink_handle.join();
                return Err(error);
            }
        };
        log::info!(
            "TTS: first chunk synthesized in {:?} ({} chars)",
            synth_started.elapsed(),
            first_chunk.chars().count()
        );
        if generation != self.generation.load(Ordering::SeqCst) {
            let _ = sink_handle.join();
            return Ok(());
        }

        let sink = sink_handle
            .join()
            .map_err(|_| "The audio device thread panicked.".to_string())?
            .map_err(|e| format!("Failed to open the output device: {e}"))?;
        let player = Arc::new(rodio::Player::connect_new(sink.mixer()));
        player.set_volume(volume.clamp(0.0, 1.0));
        {
            let mut guard = self.playback.lock().unwrap_or_else(|e| e.into_inner());
            if generation != self.generation.load(Ordering::SeqCst) {
                player.stop();
                return Ok(());
            }
            *guard = Some(Playback {
                _sink: sink,
                player: player.clone(),
                generation,
            });
        }

        let mut played = 0usize;
        let mut pending_first = Some(first_samples);
        for chunk in std::iter::once(first_chunk).chain(chunks.into_iter().skip(1)) {
            let samples = match pending_first.take() {
                Some(samples) => samples,
                None => {
                    if generation != self.generation.load(Ordering::SeqCst) {
                        player.stop();
                        return Ok(());
                    }
                    synthesize_chunk(&engine, &chunk, &style, &language, speed)?
                }
            };
            if generation != self.generation.load(Ordering::SeqCst) {
                player.stop();
                return Ok(());
            }
            if samples.is_empty() {
                continue;
            }
            if played == 0 {
                info!(
                    "TTS: first audio after {:?} ({} samples)",
                    request_started.elapsed(),
                    samples.len()
                );
            }
            played += 1;
            player.append(rodio::buffer::SamplesBuffer::new(
                rodio::nz!(1),
                rodio::nz!(44100),
                samples,
            ));
        }

        if played == 0 {
            return Err("The speech engine produced no audio.".to_string());
        }
        player.sleep_until_end();
        self.finish_playback(generation);
        Ok(())
    }

    /// Retire playback and hide the overlay after an error in the streaming
    /// worker. A stop/newer request owns the state and gets no duplicate notice.
    fn fail_playback(&self, generation: u64, error: String) {
        if generation != self.generation.load(Ordering::SeqCst) {
            return;
        }
        if let Some(playback) = self
            .playback
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            playback.player.stop();
        }
        self.set_speaking(false);
        if self.overlay_active.swap(false, Ordering::SeqCst) {
            crate::overlay::hide_speaking_overlay(&self.app);
        }
        self.report_failure(error);
    }

    /// Load the ONNX sessions ahead of the first shortcut so the first read
    /// does not pay the ~4 s model-load cost.
    pub fn prewarm(self: &Arc<Self>) {
        if self.engine_loaded.load(Ordering::SeqCst) || !self.is_model_downloaded() {
            return;
        }
        let manager = self.clone();
        std::thread::spawn(move || {
            // Let the window, tray and transcription worker settle first.
            std::thread::sleep(Duration::from_millis(1500));
            if manager.engine_loaded.load(Ordering::SeqCst) {
                return;
            }
            if let Err(error) = tauri::async_runtime::block_on(manager.ensure_engine()) {
                debug!("TTS: prewarm skipped: {error}");
            }
        });
    }

    /// Retire the playback entry when its audio has drained. A newer request
    /// (or `stop`) has already replaced it, so a stale call is a no-op.
    pub fn finish_playback(&self, generation: u64) {
        let mut guard = self.playback.lock().unwrap_or_else(|e| e.into_inner());
        if guard.as_ref().map(|p| p.generation) == Some(generation) {
            guard.take();
            self.set_speaking(false);
            drop(guard);
            if self.overlay_active.swap(false, Ordering::SeqCst) {
                crate::overlay::hide_speaking_overlay(&self.app);
            }
        }
    }

    pub async fn download(self: &Arc<Self>) -> Result<(), String> {
        if self.downloading.swap(true, Ordering::SeqCst) {
            return Err("The Read Aloud model is already downloading.".to_string());
        }
        self.download_cancelled.store(false, Ordering::SeqCst);
        let result = self.download_inner().await;
        self.downloading.store(false, Ordering::SeqCst);
        if result.is_ok() {
            self.prewarm();
        }
        if let Err(error) = self.app.emit("tts-status", self.status()) {
            warn!("Failed to emit TTS status: {error}");
        }
        result
    }

    async fn download_inner(&self) -> Result<(), String> {
        if self.is_model_downloaded() {
            return Ok(());
        }
        info!(
            "TTS: downloading Supertonic 3 to {}",
            self.model_dir.display()
        );
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(60 * 30))
            .user_agent(concat!("Handy/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| format!("Failed to create download client: {e}"))?;

        for relative in MODEL_FILES.iter().chain(std::iter::once(&LICENSE_FILE)) {
            if self.download_cancelled.load(Ordering::SeqCst) {
                return Err("Download cancelled.".to_string());
            }
            let destination = self.model_dir.join(relative);
            if std::fs::metadata(&destination)
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
            {
                continue;
            }
            if let Some(parent) = destination.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
            }

            let part = self.model_dir.join(format!("{relative}.part"));
            let _ = tokio::fs::remove_file(&part).await;
            let url = format!(
                "https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{relative}?download=true"
            );
            debug!("TTS: GET {url}");
            let mut response = client
                .get(&url)
                .send()
                .await
                .map_err(|e| format!("Failed to download {relative}: {e}"))?
                .error_for_status()
                .map_err(|e| format!("Failed to download {relative}: {e}"))?;

            let mut file = tokio::fs::File::create(&part)
                .await
                .map_err(|e| format!("Failed to create {}: {e}", part.display()))?;
            use tokio::io::AsyncWriteExt;
            let mut since_emit = 0u64;
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|e| format!("Failed to read {relative}: {e}"))?
            {
                if self.download_cancelled.load(Ordering::SeqCst) {
                    drop(file);
                    let _ = tokio::fs::remove_file(&part).await;
                    return Err("Download cancelled.".to_string());
                }
                file.write_all(&chunk)
                    .await
                    .map_err(|e| format!("Failed to write {relative}: {e}"))?;
                since_emit += chunk.len() as u64;
                // A status event per 8 KB chunk would flood the webview; one
                // every 2 MB keeps the bar smooth without the noise.
                if since_emit >= 2 * 1024 * 1024 {
                    since_emit = 0;
                    if let Err(error) = self.app.emit("tts-download-progress", self.status()) {
                        debug!("Failed to emit TTS download progress: {error}");
                    }
                }
            }
            file.flush()
                .await
                .map_err(|e| format!("Failed to flush {relative}: {e}"))?;
            drop(file);
            tokio::fs::rename(&part, &destination)
                .await
                .map_err(|e| format!("Failed to finalize {relative}: {e}"))?;
            info!("TTS: downloaded {relative}");
        }

        if !self.is_model_downloaded() {
            return Err("The Read Aloud model download is incomplete.".to_string());
        }
        info!("TTS: model download complete");
        Ok(())
    }

    pub fn cancel_download(&self) {
        self.download_cancelled.store(true, Ordering::SeqCst);
    }

    /// Remove the model and unload the engine. Safe to call while speaking.
    pub async fn delete_model(&self) -> Result<(), String> {
        self.stop();
        *self.engine.lock().await = None;
        self.engine_loaded.store(false, Ordering::SeqCst);
        if let Ok(metadata) = tokio::fs::symlink_metadata(&self.model_dir).await {
            // A test/profile may point the model dir at a symlink; remove the
            // link rather than recursively deleting its target.
            if metadata.file_type().is_symlink() {
                tokio::fs::remove_file(&self.model_dir)
                    .await
                    .map_err(|e| format!("Failed to remove the Read Aloud model link: {e}"))?;
            } else {
                tokio::fs::remove_dir_all(&self.model_dir)
                    .await
                    .map_err(|e| format!("Failed to delete the Read Aloud model: {e}"))?;
            }
        }
        if let Err(error) = self.app.emit("tts-status", self.status()) {
            warn!("Failed to emit TTS status: {error}");
        }
        Ok(())
    }

    /// Speak a short sample directly (used by the voice picker). Uses the
    /// current language and speed settings.
    pub async fn preview(self: &Arc<Self>, voice: String) -> Result<(), String> {
        let settings = settings::get_settings(&self.app);
        self.speak(
            "Hello. This is what I sound like.".to_string(),
            voice,
            settings.tts_language,
            settings.tts_speed,
        )
        .await
    }

    /// Speak text requested from the UI or a shortcut, reading the current
    /// preferences. Kept here so the shortcut action and Tauri commands share
    /// exactly one path.
    pub async fn speak_with_settings(self: &Arc<Self>, text: String) -> Result<(), String> {
        let settings = settings::get_settings(&self.app);
        self.speak(
            text,
            settings.tts_voice,
            settings.tts_language,
            settings.tts_speed,
        )
        .await
    }

    /// Raise the standard read-aloud failure notice, then return the error so
    /// callers that have a dialog can also surface it.
    pub fn report_failure(&self, error: impl Into<String>) -> String {
        let error = error.into();
        warn!("Read aloud failed: {error}");
        notices::raise(
            &self.app,
            notices::Notice::new(
                notices::NoticeCode::ReadAloudFailed,
                notices::NoticeSource::ReadAloud,
            )
            .with_detail(error.clone()),
        );
        error
    }
}

fn synthesize_chunk(
    engine: &TtsEngine,
    text: &str,
    style: &Style,
    language: &str,
    speed: f32,
) -> Result<Vec<f32>, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("Failed to create TTS runtime: {e}"))?;
    runtime
        .block_on(engine.synthesize_with_style(
            text,
            style,
            speed.clamp(0.5, 2.0),
            1.0,
            Some(language),
            Some(VOICE_QUALITY),
        ))
        .map_err(|e| format!("Speech synthesis failed: {e}"))
}
