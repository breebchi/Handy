//! # supertonic3-tts
//!
//! Extremely fast, on-device, multilingual (31 languages) text to speech for
//! Rust built on the [Supertonic 3](https://huggingface.co/Supertone/supertonic-3)
//! ONNX model.
//!
//! ```no_run
//! use supertonic3_tts::TtsEngine;
//! use std::path::PathBuf;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let base = PathBuf::from("./supertonic3-model");
//! // CPU, or a GPU when built with a GPU feature (`cuda`, `tensorrt`, `nvrtx`,
//! // `webgpu`, `directml`, `coreml`, `rocm`, `migraphx`, `openvino`, `qnn`,
//! // `cann`, `nnapi`) and one is available; see `TtsEngine::on_device`
//! let engine = TtsEngine::new(base.join("onnx"), base, false).await?;
//! let wav = engine
//!   .synthesize_with_options("Hello world", Some("F1"), 1.0, 1.0, Some("en"), None)
//!   .await?;
//! // or with an explicit voice quality (5 fastest .. 12 best, None = default 8)
//! let wav_hq = engine
//!   .synthesize_with_options("Hello world", Some("F1"), 1.0, 1.0, Some("en"), Some(12))
//!   .await?;
//! engine.save_wav("output.wav", &wav).await?;
//!
//! // Voice cloning: reference recording (+ transcript) -> voice style JSON
//! let options = supertonic3_tts::CloneOptions {
//!   reference_text: Some("The words spoken in the recording.".into()),
//!   ..Default::default()
//! };
//! let cloned = engine.clone_voice_from_wav("reference.wav", &options).await?;
//! cloned.save("cloned_voice.json")?;
//! let wav = engine
//!   .synthesize_with_style("Hello", &cloned.style, 1.0, 1.0, Some("en"), None)
//!   .await?;
//! # Ok(())
//! # }
//! ```

pub mod clone;
pub mod device;
pub mod features;
pub mod helper;
pub mod speaker;

pub use clone::{
  probe_bank_for, CloneOptions, CloneProgress, CloneStage, ClonedVoice, LossWeights, VoiceCloner,
  PROBE_BANK,
};
pub use device::{compiled_gpu_providers, gpu_support_compiled, Backend, Device};
pub use features::Audio;
pub use helper::{
  is_valid_lang, is_valid_voice, language_name, Style, AVAILABLE_LANGS, LANG_AGNOSTIC,
  SUPPORTED_LANGS, VOICE_STYLES,
};
pub use speaker::SpeakerEncoder;

use anyhow::{anyhow, Result};
use helper::{load_text_to_speech_on, load_voice_style, set_verbose, write_wav_file, TextToSpeech};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Default voice quality (number of flow-matching denoising steps).
/// 8 is the model default; 5 trades a little clarity for noticeably less
/// vector-estimator time, 12 is the best quality.
pub const DEFAULT_VOICE_QUALITY: usize = 8;

/// Minimum allowed voice quality (fastest, lowest quality).
pub const MIN_VOICE_QUALITY: usize = 5;

/// Maximum allowed voice quality (slowest, best quality).
pub const MAX_VOICE_QUALITY: usize = 12;

/// Validate a voice quality value against the allowed range
/// [`MIN_VOICE_QUALITY`]..=[`MAX_VOICE_QUALITY`].
pub fn validate_voice_quality(voice_quality: usize) -> Result<usize> {
  if (MIN_VOICE_QUALITY..=MAX_VOICE_QUALITY).contains(&voice_quality) {
    Ok(voice_quality)
  } else {
    Err(anyhow!(
      "Invalid voice_quality value {}: must be between {} (fastest) and {} (best quality)",
      voice_quality,
      MIN_VOICE_QUALITY,
      MAX_VOICE_QUALITY
    ))
  }
}

/// Silence inserted between text chunks, in seconds.
pub const DEFAULT_SILENCE_DURATION: f32 = 0.3;

/// Voice style used when none is given.
pub const DEFAULT_VOICE: &str = "M1";

/// Language used when none is given.
pub const DEFAULT_LANGUAGE: &str = "en";

pub struct TtsEngine {
  inner: Arc<Mutex<TextToSpeech>>,
  base_path: PathBuf,
  verbose: bool,
  voice_quality: usize,
  backend: Backend,
}

impl TtsEngine {
  /// Create a new TTS engine. Loads the ONNX model files from the provided
  /// `onnx_dir`. `base_path` is the model root that contains the
  /// `voice_styles` directory. Runs on the GPU when the crate was built with
  /// a GPU feature and one is available, otherwise on the CPU
  /// ([`Device::Auto`]; see [`TtsEngine::on_device`] to choose).
  pub async fn new(onnx_dir: PathBuf, base_path: PathBuf, verbose: bool) -> Result<Self> {
    Self::on_device(onnx_dir, base_path, verbose, Device::Auto).await
  }

  /// Same as [`TtsEngine::new`] with an explicit [`Device`]. `Device::Gpu`
  /// fails when no GPU provider is compiled in (cargo features `cuda`,
  /// `tensorrt`, `nvrtx`, `webgpu`, `directml`, `coreml`, `rocm`, `migraphx`,
  /// `openvino`, `qnn`, `cann`, `nnapi`) or it cannot be initialised.
  pub async fn on_device(
    onnx_dir: PathBuf,
    base_path: PathBuf,
    verbose: bool,
    device: Device,
  ) -> Result<Self> {
    if !onnx_dir.join("tts.json").is_file() {
      return Err(anyhow!(
        "Supertonic 3 model not found at {} (expected tts.json, unicode_indexer.json and \
         the *.onnx files there). Download it from \
         https://huggingface.co/Supertone/supertonic-3 (see README / scripts/download-model.sh)",
        onnx_dir.display()
      ));
    }
    set_verbose(verbose);
    let onnx_dir_str = onnx_dir
      .to_str()
      .ok_or_else(|| anyhow!("onnx dir path is not valid UTF-8: {}", onnx_dir.display()))?;
    let tts = load_text_to_speech_on(onnx_dir_str, device)?;
    let backend = tts.backend().clone();
    Ok(Self {
      inner: Arc::new(Mutex::new(tts)),
      base_path,
      verbose,
      voice_quality: DEFAULT_VOICE_QUALITY,
      backend,
    })
  }

  /// Execution provider the model runs on (`CPU`, `CUDAExecutionProvider`,
  /// ...).
  pub fn backend(&self) -> &Backend {
    &self.backend
  }

  /// Change the default voice quality (denoising steps, quality vs speed)
  /// used by `synthesize_with_options` and `synthesize_with_style`
  /// when called with `voice_quality = None`.
  /// Allowed range is 5 (fastest) to 12 (best quality), default 8.
  pub fn with_voice_quality(mut self, voice_quality: usize) -> Result<Self> {
    self.voice_quality = validate_voice_quality(voice_quality)?;
    Ok(self)
  }

  /// Default voice quality used when none is passed.
  pub fn voice_quality(&self) -> usize {
    self.voice_quality
  }

  /// Resolve a voice argument to a voice style JSON path. Accepts a voice
  /// name (`M1`, `F3`, ...), a file inside the model root, or a path to a
  /// custom voice style JSON (e.g. one created with Voice Builder).
  fn voice_style_path(&self, voice: Option<&str>) -> PathBuf {
    let base_path =
      std::fs::canonicalize(&self.base_path).unwrap_or_else(|_| self.base_path.clone());
    let v = voice.unwrap_or(DEFAULT_VOICE);
    let direct = PathBuf::from(v);
    if direct.is_file() {
      return direct;
    }
    if v.contains(std::path::MAIN_SEPARATOR) {
      return base_path.join(v);
    }
    let candidate = base_path.join(v);
    if candidate.is_file() {
      candidate
    } else {
      base_path.join(format!("voice_styles/{}.json", v))
    }
  }

  /// Synthesize audio from text with optional voice style, speed, gain,
  /// language and voice quality.
  /// Returns mono f32 samples at [`TtsEngine::sample_rate`].
  ///
  /// * `voice`: one of `M1..M5`, `F1..F5`, the name (without `.json`) of
  ///   any voice style JSON in the model's `voice_styles` folder (e.g. a
  ///   cloned voice saved there), or a path to a voice style JSON (defaults
  ///   to `M1`).
  /// * `speed`: 1.0 = normal, >1.0 faster, <1.0 slower (useful range 0.7 - 2.0).
  /// * `gain`: linear multiplier applied to the samples.
  /// * `language`: any of [`SUPPORTED_LANGS`], or `"na"` for language
  ///   agnostic synthesis (defaults to `en`).
  /// * `voice_quality`: number of denoising steps, from [`MIN_VOICE_QUALITY`]
  ///   (5, fastest) to [`MAX_VOICE_QUALITY`] (12, best). `None` uses the
  ///   engine default (8 unless changed with [`TtsEngine::with_voice_quality`]).
  ///   Out of range values return an error.
  pub async fn synthesize_with_options(
    &self,
    text: &str,
    voice: Option<&str>,
    speed: f32,
    gain: f32,
    language: Option<&str>,
    voice_quality: Option<usize>,
  ) -> Result<Vec<f32>> {
    let language = language.unwrap_or(DEFAULT_LANGUAGE);
    let voice_quality = match voice_quality {
      Some(q) => validate_voice_quality(q)?,
      None => self.voice_quality,
    };
    if !is_valid_lang(language) {
      return Err(anyhow!(
        "Invalid language: {}. Available: {}",
        language,
        AVAILABLE_LANGS.join(", ")
      ));
    }
    if text.trim().is_empty() {
      return Ok(Vec::new());
    }

    let path = self.voice_style_path(voice);
    let path_str = path
      .to_str()
      .ok_or_else(|| anyhow!("voice style path is not valid UTF-8: {}", path.display()))?;
    let voice_style = load_voice_style(&[path_str.to_string()], self.verbose)?;
    self
      .synthesize_with_style(
        text,
        &voice_style,
        speed,
        gain,
        Some(language),
        Some(voice_quality),
      )
      .await
  }

  /// Same as [`TtsEngine::synthesize_with_options`] but with an in-memory
  /// [`Style`] (for example the `style` of a [`ClonedVoice`]) instead of a
  /// voice name / path.
  pub async fn synthesize_with_style(
    &self,
    text: &str,
    voice_style: &Style,
    speed: f32,
    gain: f32,
    language: Option<&str>,
    voice_quality: Option<usize>,
  ) -> Result<Vec<f32>> {
    let language = language.unwrap_or(DEFAULT_LANGUAGE);
    let voice_quality = match voice_quality {
      Some(q) => validate_voice_quality(q)?,
      None => self.voice_quality,
    };
    if !is_valid_lang(language) {
      return Err(anyhow!(
        "Invalid language: {}. Available: {}",
        language,
        AVAILABLE_LANGS.join(", ")
      ));
    }
    if text.trim().is_empty() {
      return Ok(Vec::new());
    }

    let mut tts = self.inner.lock().await;
    let (wav, _duration) = tts.call(
      text,
      language,
      voice_style,
      voice_quality,
      speed,
      DEFAULT_SILENCE_DURATION,
    )?;
    let wav: Vec<f32> = if (gain - 1.0).abs() < f32::EPSILON {
      wav
    } else {
      wav.iter().map(|x| (x * gain).clamp(-1.0, 1.0)).collect()
    };
    Ok(wav)
  }

  /// Save the given audio buffer to a 16-bit mono WAV file. The operation is
  /// performed in a blocking task.
  pub async fn save_wav(&self, path: &str, audio: &[f32]) -> Result<()> {
    let sample_rate = self.sample_rate().await;
    let audio_vec = audio.to_vec();
    let path_owned = path.to_owned();
    tokio::task::spawn_blocking(move || write_wav_file(&path_owned, &audio_vec, sample_rate))
      .await?
  }

  /// Return the sample rate of the underlying TextToSpeech instance (44100 Hz
  /// for Supertonic 3).
  pub async fn sample_rate(&self) -> i32 {
    let tts = self.inner.lock().await;
    tts.sample_rate
  }

  /// Directory holding the preset voice styles (`<base_path>/voice_styles`).
  pub fn voice_styles_dir(&self) -> PathBuf {
    self.base_path.join("voice_styles")
  }

  /// Clone a voice from a reference recording (any sample rate, mono or
  /// stereo): search for style tensors that make the model sound like the
  /// reference. Give the transcript in `options.reference_text` for the best
  /// result. Runs on a blocking thread and takes minutes on a CPU (see
  /// [`CloneOptions`] for the budget). The returned [`ClonedVoice`] can be
  /// saved as a voice style JSON ([`ClonedVoice::save`]) and used with
  /// `--voice <path>` / `synthesize_with_options(.., Some(path), ..)`, or
  /// used directly with [`TtsEngine::synthesize_with_style`].
  pub async fn clone_voice(
    &self,
    reference: &Audio,
    options: &CloneOptions,
  ) -> Result<ClonedVoice> {
    self
      .clone_voice_with_progress(reference, options, |_| {})
      .await
  }

  /// [`TtsEngine::clone_voice`] with the reference loaded from a WAV file.
  pub async fn clone_voice_from_wav<P: AsRef<Path>>(
    &self,
    reference_wav: P,
    options: &CloneOptions,
  ) -> Result<ClonedVoice> {
    let reference = features::load_wav(reference_wav)?;
    self.clone_voice(&reference, options).await
  }

  /// [`TtsEngine::clone_voice`] with a progress callback invoked after every
  /// iteration (loss, stage, best style so far for checkpointing).
  pub async fn clone_voice_with_progress<F>(
    &self,
    reference: &Audio,
    options: &CloneOptions,
    mut progress: F,
  ) -> Result<ClonedVoice>
  where
    F: FnMut(&CloneProgress<'_>) + Send + 'static,
  {
    let inner = Arc::clone(&self.inner);
    let reference = reference.clone();
    let options = options.clone();
    let styles_dir = self.voice_styles_dir();
    tokio::task::spawn_blocking(move || -> Result<ClonedVoice> {
      let mut tts = inner.blocking_lock();
      let cloner = VoiceCloner::new(&mut tts, &reference, &styles_dir, options)?;
      cloner.run(&mut progress)
    })
    .await?
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn voice_quality_range_is_validated() {
    assert!(validate_voice_quality(MIN_VOICE_QUALITY - 1).is_err());
    assert!(validate_voice_quality(MAX_VOICE_QUALITY + 1).is_err());
    assert!(validate_voice_quality(0).is_err());
    for q in MIN_VOICE_QUALITY..=MAX_VOICE_QUALITY {
      assert_eq!(validate_voice_quality(q).unwrap(), q);
    }
    assert_eq!(DEFAULT_VOICE_QUALITY, 8);
  }
}
