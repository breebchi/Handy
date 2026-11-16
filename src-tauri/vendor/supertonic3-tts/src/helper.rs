// ============================================================================
// TTS Helper Module - Supertonic 3 (multilingual) inference
// ============================================================================
//
// Includes MIT-licensed inference code Copyright (c) 2025 Supertone Inc.
// (see LICENSE).

use std::sync::atomic::{AtomicBool, Ordering};
static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Set global verbose flag used by helper print statements
pub fn set_verbose(v: bool) {
  VERBOSE.store(v, Ordering::Relaxed);
}

fn verbose() -> bool {
  VERBOSE.load(Ordering::Relaxed)
}

/// Current value of the global verbose flag.
pub fn is_verbose() -> bool {
  verbose()
}

use crate::device::{load_session, resolve_backend, Backend, Device};
use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::{Array, Array3};
use ort::session::Session;
use ort::value::Value;
use rand_distr::{Distribution, Normal};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

// ============================================================================
// Languages / voices
// ============================================================================

/// Voice styles shipped with the Supertonic 3 model (5 male, 5 female).
/// Every voice can speak every supported language.
pub const VOICE_STYLES: [&str; 10] = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"];

/// Languages accepted by the Supertonic 3 model. "na" means language
/// agnostic: the model guesses the language from the text itself.
pub const AVAILABLE_LANGS: &[&str] = &[
  "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
  "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
];

/// Concrete languages (everything in `AVAILABLE_LANGS` except "na").
pub const SUPPORTED_LANGS: &[&str] = &[
  "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
  "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
];

/// Language code for language-agnostic synthesis.
pub const LANG_AGNOSTIC: &str = "na";

pub fn is_valid_lang(lang: &str) -> bool {
  AVAILABLE_LANGS.contains(&lang)
}

pub fn is_valid_voice(voice: &str) -> bool {
  VOICE_STYLES.contains(&voice)
}

/// Human readable language names, keyed by language code.
pub fn language_name(code: &str) -> Option<&'static str> {
  static NAMES: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
  NAMES
    .get_or_init(|| {
      HashMap::from([
        ("ar", "Arabic"),
        ("bg", "Bulgarian"),
        ("cs", "Czech"),
        ("da", "Danish"),
        ("de", "German"),
        ("el", "Greek"),
        ("en", "English"),
        ("es", "Spanish"),
        ("et", "Estonian"),
        ("fi", "Finnish"),
        ("fr", "French"),
        ("hi", "Hindi"),
        ("hr", "Croatian"),
        ("hu", "Hungarian"),
        ("id", "Indonesian"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("lt", "Lithuanian"),
        ("lv", "Latvian"),
        ("nl", "Dutch"),
        ("pl", "Polish"),
        ("pt", "Portuguese"),
        ("ro", "Romanian"),
        ("ru", "Russian"),
        ("sk", "Slovak"),
        ("sl", "Slovenian"),
        ("sv", "Swedish"),
        ("tr", "Turkish"),
        ("uk", "Ukrainian"),
        ("vi", "Vietnamese"),
        ("na", "Language agnostic"),
      ])
    })
    .get(code)
    .copied()
}

// ============================================================================
// Configuration Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
  pub ae: AEConfig,
  pub ttl: TTLConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AEConfig {
  pub sample_rate: i32,
  pub base_chunk_size: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTLConfig {
  pub chunk_compress_factor: i32,
  pub latent_dim: i32,
}

/// Load configuration from JSON file
pub fn load_cfgs<P: AsRef<Path>>(onnx_dir: P) -> Result<Config> {
  let cfg_path = onnx_dir.as_ref().join("tts.json");
  let file = File::open(&cfg_path).with_context(|| format!("open {}", cfg_path.display()))?;
  let reader = BufReader::new(file);
  let cfgs: Config = serde_json::from_reader(reader)?;
  Ok(cfgs)
}

// ============================================================================
// Voice Style Data Structure
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceStyleData {
  pub style_ttl: StyleComponent,
  pub style_dp: StyleComponent,
  /// Free-form metadata (the presets carry their source file, cloned voices
  /// the cloning settings). Ignored by the model.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleComponent {
  pub data: Vec<Vec<Vec<f32>>>,
  pub dims: Vec<usize>,
  #[serde(rename = "type")]
  pub dtype: String,
}

// ============================================================================
// Unicode Text Processor
// ============================================================================

pub struct UnicodeProcessor {
  indexer: Vec<i64>,
}

impl UnicodeProcessor {
  pub fn new<P: AsRef<Path>>(unicode_indexer_json_path: P) -> Result<Self> {
    let file = File::open(&unicode_indexer_json_path)
      .with_context(|| format!("open {}", unicode_indexer_json_path.as_ref().display()))?;
    let reader = BufReader::new(file);
    let indexer: Vec<i64> = serde_json::from_reader(reader)?;
    Ok(UnicodeProcessor { indexer })
  }

  pub fn call(
    &self,
    text_list: &[String],
    lang_list: &[String],
  ) -> Result<(Vec<Vec<i64>>, Array3<f32>)> {
    let mut processed_texts: Vec<String> = Vec::new();
    for (text, lang) in text_list.iter().zip(lang_list.iter()) {
      processed_texts.push(preprocess_text(text, lang)?);
    }

    let text_ids_lengths: Vec<usize> = processed_texts.iter().map(|t| t.chars().count()).collect();

    let max_len = *text_ids_lengths.iter().max().unwrap_or(&0);

    let mut text_ids = Vec::new();
    for text in &processed_texts {
      let mut row = vec![0i64; max_len];
      let unicode_vals = text_to_unicode_values(text);
      for (j, &val) in unicode_vals.iter().enumerate() {
        if val < self.indexer.len() {
          row[j] = self.indexer[val];
        } else {
          row[j] = -1;
        }
      }
      text_ids.push(row);
    }

    let text_mask = get_text_mask(&text_ids_lengths);

    Ok((text_ids, text_mask))
  }
}

fn emoji_regex() -> &'static Regex {
  static RE: OnceLock<Regex> = OnceLock::new();
  RE.get_or_init(|| {
    Regex::new(
      r"[\x{1F600}-\x{1F64F}\x{1F300}-\x{1F5FF}\x{1F680}-\x{1F6FF}\x{1F700}-\x{1F77F}\x{1F780}-\x{1F7FF}\x{1F800}-\x{1F8FF}\x{1F900}-\x{1F9FF}\x{1FA00}-\x{1FA6F}\x{1FA70}-\x{1FAFF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}\x{1F1E6}-\x{1F1FF}]+",
    )
    .unwrap()
  })
}

fn whitespace_regex() -> &'static Regex {
  static RE: OnceLock<Regex> = OnceLock::new();
  RE.get_or_init(|| Regex::new(r"\s+").unwrap())
}

fn ends_with_punct_regex() -> &'static Regex {
  static RE: OnceLock<Regex> = OnceLock::new();
  RE.get_or_init(|| {
    Regex::new(r#"[.!?;:,'"\u{201C}\u{201D}\u{2018}\u{2019})\]}…。」』】〉》›»]$"#).unwrap()
  })
}

/// Text normalisation expected by the Supertonic 3 model: NFKD, strip
/// emojis and odd symbols, normalise quotes/dashes, tidy punctuation spacing
/// and wrap the result in `<lang>...</lang>` tags which the multilingual
/// model was trained with.
pub fn preprocess_text(text: &str, lang: &str) -> Result<String> {
  if !is_valid_lang(lang) {
    bail!(
      "Invalid language: {}. Available: {:?}",
      lang,
      AVAILABLE_LANGS
    );
  }

  let mut text: String = text.nfkd().collect();

  // Remove emojis (wide Unicode range)
  text = emoji_regex().replace_all(&text, "").to_string();

  // Replace various dashes and symbols
  let replacements = [
    ("–", "-"),         // en dash
    ("‑", "-"),         // non-breaking hyphen
    ("—", "-"),         // em dash
    ("_", " "),         // underscore
    ("\u{201C}", "\""), // left double quote
    ("\u{201D}", "\""), // right double quote
    ("\u{2018}", "'"),  // left single quote
    ("\u{2019}", "'"),  // right single quote
    ("´", "'"),         // acute accent
    ("`", "'"),         // grave accent
    ("[", " "),         // left bracket
    ("]", " "),         // right bracket
    ("|", " "),         // vertical bar
    ("/", " "),         // slash
    ("#", " "),         // hash
    ("→", " "),         // right arrow
    ("←", " "),         // left arrow
  ];
  for (from, to) in &replacements {
    text = text.replace(from, to);
  }

  // Remove special symbols
  for symbol in ["♥", "☆", "♡", "©", "\\"] {
    text = text.replace(symbol, "");
  }

  // Replace known expressions
  let expr_replacements = [
    ("@", " at "),
    ("e.g.,", "for example, "),
    ("i.e.,", "that is, "),
  ];
  for (from, to) in &expr_replacements {
    text = text.replace(from, to);
  }

  // Fix spacing around punctuation
  for (from, to) in [
    (" ,", ","),
    (" .", "."),
    (" !", "!"),
    (" ?", "?"),
    (" ;", ";"),
    (" :", ":"),
    (" '", "'"),
  ] {
    text = text.replace(from, to);
  }

  // Remove duplicate quotes
  while text.contains("\"\"") {
    text = text.replace("\"\"", "\"");
  }
  while text.contains("''") {
    text = text.replace("''", "'");
  }
  while text.contains("``") {
    text = text.replace("``", "`");
  }

  // Remove extra spaces
  text = whitespace_regex().replace_all(&text, " ").to_string();
  text = text.trim().to_string();

  // If text doesn't end with punctuation, quotes, or closing brackets, add a period
  if !text.is_empty() && !ends_with_punct_regex().is_match(&text) {
    text.push('.');
  }

  // Wrap text with language tags
  Ok(format!("<{}>{}</{}>", lang, text, lang))
}

pub fn text_to_unicode_values(text: &str) -> Vec<usize> {
  text.chars().map(|c| c as usize).collect()
}

pub fn length_to_mask(lengths: &[usize], max_len: Option<usize>) -> Array3<f32> {
  let bsz = lengths.len();
  let max_len = max_len.unwrap_or_else(|| *lengths.iter().max().unwrap_or(&0));

  let mut mask = Array3::<f32>::zeros((bsz, 1, max_len));
  for (i, &len) in lengths.iter().enumerate() {
    for j in 0..len.min(max_len) {
      mask[[i, 0, j]] = 1.0;
    }
  }
  mask
}

pub fn get_text_mask(text_ids_lengths: &[usize]) -> Array3<f32> {
  let max_len = *text_ids_lengths.iter().max().unwrap_or(&0);
  length_to_mask(text_ids_lengths, Some(max_len))
}

/// Sample noisy latent from normal distribution and apply mask
pub fn sample_noisy_latent(
  duration: &[f32],
  sample_rate: i32,
  base_chunk_size: i32,
  chunk_compress: i32,
  latent_dim: i32,
) -> (Array3<f32>, Array3<f32>) {
  let bsz = duration.len();
  let max_dur = duration.iter().fold(0.0f32, |a, &b| a.max(b));

  let wav_len_max = (max_dur * sample_rate as f32) as usize;
  let wav_lengths: Vec<usize> = duration
    .iter()
    .map(|&d| (d * sample_rate as f32) as usize)
    .collect();

  let chunk_size = (base_chunk_size * chunk_compress) as usize;
  let latent_len = wav_len_max.div_ceil(chunk_size);
  let latent_dim_val = (latent_dim * chunk_compress) as usize;

  let mut noisy_latent = Array3::<f32>::zeros((bsz, latent_dim_val, latent_len));

  let normal = Normal::new(0.0, 1.0).unwrap();
  let mut rng = rand::thread_rng();

  for b in 0..bsz {
    for d in 0..latent_dim_val {
      for t in 0..latent_len {
        noisy_latent[[b, d, t]] = normal.sample(&mut rng);
      }
    }
  }

  let latent_lengths: Vec<usize> = wav_lengths
    .iter()
    .map(|&len| len.div_ceil(chunk_size))
    .collect();

  let latent_mask = length_to_mask(&latent_lengths, Some(latent_len));

  // Apply mask
  for b in 0..bsz {
    for d in 0..latent_dim_val {
      for t in 0..latent_len {
        noisy_latent[[b, d, t]] *= latent_mask[[b, 0, t]];
      }
    }
  }

  (noisy_latent, latent_mask)
}

// ============================================================================
// WAV File I/O
// ============================================================================

pub fn write_wav_file<P: AsRef<Path>>(
  filename: P,
  audio_data: &[f32],
  sample_rate: i32,
) -> Result<()> {
  let spec = WavSpec {
    channels: 1,
    sample_rate: sample_rate as u32,
    bits_per_sample: 16,
    sample_format: SampleFormat::Int,
  };

  let mut writer = WavWriter::create(filename, spec)?;

  for &sample in audio_data {
    let clamped = if sample.is_finite() {
      sample.clamp(-1.0, 1.0)
    } else {
      0.0
    };
    let val = (clamped * 32767.0) as i16;
    writer.write_sample(val)?;
  }

  writer.finalize()?;
  Ok(())
}

// ============================================================================
// Text Chunking
// ============================================================================

const MAX_CHUNK_LENGTH: usize = 300;

const ABBREVIATIONS: &[&str] = &[
  "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "St.", "Ave.", "Rd.", "Blvd.", "Dept.",
  "Inc.", "Ltd.", "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D.",
];

/// Maximum chunk length for a language. Shorter chunks for languages whose
/// characters carry more information per code point.
pub fn max_chunk_length(lang: &str) -> usize {
  if lang == "ko" || lang == "ja" {
    120
  } else {
    MAX_CHUNK_LENGTH
  }
}

pub fn chunk_text(text: &str, max_len: Option<usize>) -> Vec<String> {
  let max_len = max_len.unwrap_or(MAX_CHUNK_LENGTH);
  let text = text.trim();

  if text.is_empty() {
    return vec![String::new()];
  }

  // Split by paragraphs
  let para_re = Regex::new(r"\n\s*\n").unwrap();
  let paragraphs: Vec<&str> = para_re.split(text).collect();
  let mut chunks: Vec<String> = Vec::new();

  for para_str in paragraphs {
    let para = para_str.trim();
    if para.is_empty() {
      continue;
    }

    if para.len() <= max_len {
      chunks.push(para.to_string());
      continue;
    }

    // Split by sentences
    let sentences = split_sentences(para);
    let mut current = String::new();
    let mut current_len = 0;

    for sentence in sentences {
      let sentence = sentence.trim();
      if sentence.is_empty() {
        continue;
      }

      let sentence_len = sentence.len();
      if sentence_len > max_len {
        // If sentence is longer than max_len, split by comma or space
        if !current.is_empty() {
          chunks.push(current.trim().to_string());
          current.clear();
          current_len = 0;
        }

        // Try splitting by comma
        let parts: Vec<&str> = sentence.split(',').collect();
        for part in parts {
          let part = part.trim();
          if part.is_empty() {
            continue;
          }

          let part_len = part.len();
          if part_len > max_len {
            // Split by space as last resort
            let words: Vec<&str> = part.split_whitespace().collect();
            let mut word_chunk = String::new();
            let mut word_chunk_len = 0;

            for word in words {
              let word_len = word.len();
              if word_chunk_len + word_len + 1 > max_len && !word_chunk.is_empty() {
                chunks.push(word_chunk.trim().to_string());
                word_chunk.clear();
                word_chunk_len = 0;
              }

              if !word_chunk.is_empty() {
                word_chunk.push(' ');
                word_chunk_len += 1;
              }
              word_chunk.push_str(word);
              word_chunk_len += word_len;
            }

            if !word_chunk.is_empty() {
              chunks.push(word_chunk.trim().to_string());
            }
          } else {
            if current_len + part_len + 1 > max_len && !current.is_empty() {
              chunks.push(current.trim().to_string());
              current.clear();
              current_len = 0;
            }

            if !current.is_empty() {
              current.push_str(", ");
              current_len += 2;
            }
            current.push_str(part);
            current_len += part_len;
          }
        }
        continue;
      }

      if current_len + sentence_len + 1 > max_len && !current.is_empty() {
        chunks.push(current.trim().to_string());
        current.clear();
        current_len = 0;
      }

      if !current.is_empty() {
        current.push(' ');
        current_len += 1;
      }
      current.push_str(sentence);
      current_len += sentence_len;
    }

    if !current.is_empty() {
      chunks.push(current.trim().to_string());
    }
  }

  if chunks.is_empty() {
    vec![String::new()]
  } else {
    chunks
  }
}

fn split_sentences(text: &str) -> Vec<String> {
  // Rust's regex doesn't support lookbehind, so we use a simpler approach:
  // split on sentence boundaries and then check if they're abbreviations
  let re = Regex::new(r"([.!?])\s+").unwrap();

  let matches: Vec<_> = re.find_iter(text).collect();
  if matches.is_empty() {
    return vec![text.to_string()];
  }

  let mut sentences: Vec<String> = Vec::new();
  let mut last_end: usize = 0;

  for m in matches {
    // Get the text before the punctuation
    let before_punc = &text[last_end..m.start()];

    // Check if this ends with an abbreviation
    let mut is_abbrev = false;
    for abbrev in ABBREVIATIONS {
      let combined = format!("{}{}", before_punc.trim(), &text[m.start()..m.start() + 1]);
      if combined.ends_with(abbrev) {
        is_abbrev = true;
        break;
      }
    }

    if !is_abbrev {
      // This is a real sentence boundary
      sentences.push(text[last_end..m.end()].to_string());
      last_end = m.end();
    }
  }

  // Add the remaining text
  if last_end < text.len() {
    sentences.push(text[last_end..].to_string());
  }

  if sentences.is_empty() {
    vec![text.to_string()]
  } else {
    sentences
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

pub fn timer<F, T>(name: &str, f: F) -> Result<T>
where
  F: FnOnce() -> Result<T>,
{
  let start = std::time::Instant::now();
  if verbose() {
    println!("{}...", name);
  }
  let result = f()?;
  let elapsed = start.elapsed().as_secs_f64();
  if verbose() {
    println!("  -> {} completed in {:.2} sec", name, elapsed);
  }
  Ok(result)
}

pub fn sanitize_filename(text: &str, max_len: usize) -> String {
  // Take first max_len characters (Unicode code points, not bytes)
  text
    .chars()
    .take(max_len)
    .map(|c| if c.is_alphanumeric() { c } else { '_' })
    .collect()
}

// ============================================================================
// ONNX Runtime Integration
// ============================================================================

/// Voice style tensors fed to the model: `ttl` (`[batch, 50, 256]`, text
/// encoder + vector estimator) and `dp` (`[batch, 8, 16]`, duration
/// predictor). A voice is nothing more than these two tensors.
#[derive(Debug, Clone)]
pub struct Style {
  pub ttl: Array3<f32>,
  pub dp: Array3<f32>,
}

impl Style {
  /// Build a style from a parsed voice style JSON (`VoiceStyleData`).
  pub fn from_voice_style_data(data: &VoiceStyleData) -> Result<Style> {
    let to_array = |c: &StyleComponent| -> Result<Array3<f32>> {
      let flat: Vec<f32> = c.data.iter().flatten().flatten().copied().collect();
      let dims = &c.dims;
      if dims.len() != 3 {
        bail!("voice style tensor must have 3 dims, got {:?}", dims);
      }
      Ok(Array3::from_shape_vec((dims[0], dims[1], dims[2]), flat)?)
    };
    Ok(Style {
      ttl: to_array(&data.style_ttl)?,
      dp: to_array(&data.style_dp)?,
    })
  }

  /// Convert back to the JSON serialisable voice style structure.
  pub fn to_voice_style_data(&self) -> VoiceStyleData {
    let to_component = |a: &Array3<f32>| -> StyleComponent {
      let dims = a.shape().to_vec();
      let data: Vec<Vec<Vec<f32>>> = (0..dims[0])
        .map(|b| {
          (0..dims[1])
            .map(|r| (0..dims[2]).map(|c| a[[b, r, c]]).collect())
            .collect()
        })
        .collect();
      StyleComponent {
        data,
        dims,
        dtype: "float32".to_string(),
      }
    };
    VoiceStyleData {
      style_ttl: to_component(&self.ttl),
      style_dp: to_component(&self.dp),
      metadata: None,
    }
  }
}

/// Save a style as a voice style JSON, in the same format as the bundled
/// presets (loadable with `load_voice_style` or as `--voice <path>`).
/// `metadata` is stored under the `metadata` key when given.
pub fn save_voice_style<P: AsRef<Path>>(
  path: P,
  style: &Style,
  metadata: Option<serde_json::Value>,
) -> Result<()> {
  let mut data = style.to_voice_style_data();
  data.metadata = metadata;
  if let Some(parent) = path.as_ref().parent() {
    if !parent.as_os_str().is_empty() {
      std::fs::create_dir_all(parent)
        .with_context(|| format!("create directory {}", parent.display()))?;
    }
  }
  let file = File::create(&path)
    .with_context(|| format!("create voice style file {}", path.as_ref().display()))?;
  serde_json::to_writer(std::io::BufWriter::new(file), &data)?;
  Ok(())
}

pub struct TextToSpeech {
  cfgs: Config,
  text_processor: UnicodeProcessor,
  dp_ort: Session,
  text_enc_ort: Session,
  vector_est_ort: Session,
  vocoder_ort: Session,
  pub sample_rate: i32,
  backend: Backend,
}

impl TextToSpeech {
  pub fn new(
    cfgs: Config,
    text_processor: UnicodeProcessor,
    dp_ort: Session,
    text_enc_ort: Session,
    vector_est_ort: Session,
    vocoder_ort: Session,
  ) -> Self {
    let sample_rate = cfgs.ae.sample_rate;
    TextToSpeech {
      cfgs,
      text_processor,
      dp_ort,
      text_enc_ort,
      vector_est_ort,
      vocoder_ort,
      sample_rate,
      backend: Backend::cpu(),
    }
  }

  /// Execution provider the sessions run on.
  pub fn backend(&self) -> &Backend {
    &self.backend
  }

  /// Run the full pipeline for a batch of texts. Returns the raw vocoder
  /// output (padded to whole latent chunks) and the predicted duration in
  /// seconds per batch item.
  fn _infer(
    &mut self,
    text_list: &[String],
    lang_list: &[String],
    style: &Style,
    total_step: usize,
    speed: f32,
  ) -> Result<(Vec<f32>, Vec<f32>)> {
    self._infer_with_noise(text_list, lang_list, style, total_step, speed, None)
  }

  /// Latent dimension of the vector estimator input (`latent_dim *
  /// chunk_compress_factor`).
  pub fn latent_dim(&self) -> usize {
    (self.cfgs.ttl.latent_dim * self.cfgs.ttl.chunk_compress_factor) as usize
  }

  /// Number of audio samples covered by one latent frame.
  pub fn latent_chunk_size(&self) -> usize {
    (self.cfgs.ae.base_chunk_size * self.cfgs.ttl.chunk_compress_factor) as usize
  }

  /// Run the duration predictor only. Returns the predicted duration in
  /// seconds (at speed 1.0) for a single text.
  pub fn predict_duration(&mut self, text: &str, lang: &str, style: &Style) -> Result<f32> {
    let (text_ids, text_mask) = self
      .text_processor
      .call(&[text.to_string()], &[lang.to_string()])?;
    let text_ids_array = Array::from_shape_vec((1, text_ids[0].len()), text_ids[0].clone())?;
    let dp_outputs = self.dp_ort.run(ort::inputs! {
      "text_ids" => &Value::from_array(text_ids_array)?,
      "style_dp" => &Value::from_array(style.dp.clone())?,
      "text_mask" => &Value::from_array(text_mask)?
    })?;
    let (_, duration_data) = dp_outputs["duration"].try_extract_tensor::<f32>()?;
    Ok(duration_data[0])
  }

  /// Single-shot inference of one text (no chunking), returning the samples
  /// cut to the predicted duration and that duration in seconds. When
  /// `noise` is given (`[1, latent_dim, frames]`, see
  /// [`TextToSpeech::latent_dim`]) its leading frames are used as the
  /// starting latent instead of a fresh random draw, which makes the output
  /// deterministic for a given style. Used by voice cloning.
  pub fn infer_single(
    &mut self,
    text: &str,
    lang: &str,
    style: &Style,
    total_step: usize,
    speed: f32,
    noise: Option<&Array3<f32>>,
  ) -> Result<(Vec<f32>, f32)> {
    let (wav, duration) = self._infer_with_noise(
      &[text.to_string()],
      &[lang.to_string()],
      style,
      total_step,
      speed,
      noise,
    )?;
    let dur = duration[0];
    let wav_len = (self.sample_rate as f32 * dur) as usize;
    let mut wav = wav;
    wav.truncate(wav_len.min(wav.len()));
    Ok((wav, dur))
  }

  fn _infer_with_noise(
    &mut self,
    text_list: &[String],
    lang_list: &[String],
    style: &Style,
    total_step: usize,
    speed: f32,
    noise: Option<&Array3<f32>>,
  ) -> Result<(Vec<f32>, Vec<f32>)> {
    let bsz = text_list.len();

    // Process text
    let (text_ids, text_mask) = self.text_processor.call(text_list, lang_list)?;

    let text_ids_array = {
      let text_ids_shape = (bsz, text_ids[0].len());
      let mut flat = Vec::new();
      for row in &text_ids {
        flat.extend_from_slice(row);
      }
      Array::from_shape_vec(text_ids_shape, flat)?
    };

    let text_ids_value = Value::from_array(text_ids_array)?;
    let text_mask_value = Value::from_array(text_mask.clone())?;
    let style_dp_value = Value::from_array(style.dp.clone())?;

    // Predict duration
    let dp_outputs = self.dp_ort.run(ort::inputs! {
      "text_ids" => &text_ids_value,
      "style_dp" => &style_dp_value,
      "text_mask" => &text_mask_value
    })?;

    let (_, duration_data) = dp_outputs["duration"].try_extract_tensor::<f32>()?;
    let mut duration: Vec<f32> = duration_data.to_vec();

    // Apply speed factor to duration
    for dur in duration.iter_mut() {
      *dur /= speed;
    }

    // Encode text
    let style_ttl_value = Value::from_array(style.ttl.clone())?;
    let text_enc_outputs = self.text_enc_ort.run(ort::inputs! {
      "text_ids" => &text_ids_value,
      "style_ttl" => &style_ttl_value,
      "text_mask" => &text_mask_value
    })?;

    let (text_emb_shape, text_emb_data) =
      text_enc_outputs["text_emb"].try_extract_tensor::<f32>()?;
    let text_emb = Array3::from_shape_vec(
      (
        text_emb_shape[0] as usize,
        text_emb_shape[1] as usize,
        text_emb_shape[2] as usize,
      ),
      text_emb_data.to_vec(),
    )?;

    // Sample noisy latent (or reuse the caller supplied fixed noise)
    let (mut xt, latent_mask) = sample_noisy_latent(
      &duration,
      self.sample_rate,
      self.cfgs.ae.base_chunk_size,
      self.cfgs.ttl.chunk_compress_factor,
      self.cfgs.ttl.latent_dim,
    );
    if let Some(fixed) = noise {
      let (b, d, t) = xt.dim();
      let (fb, fd, ft) = fixed.dim();
      if fb < b || fd != d {
        bail!(
          "fixed noise has shape [{}, {}, {}], expected [{}, {}, >= {}]",
          fb,
          fd,
          ft,
          b,
          d,
          t
        );
      }
      // Frames beyond the provided noise keep their fresh random values.
      for bi in 0..b {
        for di in 0..d {
          for ti in 0..t.min(ft) {
            xt[[bi, di, ti]] = fixed[[bi, di, ti]] * latent_mask[[bi, 0, ti]];
          }
        }
      }
    }

    // Prepare constant arrays
    let total_step_array = Array::from_elem(bsz, total_step as f32);

    // Denoising loop
    for step in 0..total_step {
      let current_step_array = Array::from_elem(bsz, step as f32);

      let xt_value = Value::from_array(xt.clone())?;
      let text_emb_value = Value::from_array(text_emb.clone())?;
      let latent_mask_value = Value::from_array(latent_mask.clone())?;
      let text_mask_value2 = Value::from_array(text_mask.clone())?;
      let current_step_value = Value::from_array(current_step_array)?;
      let total_step_value = Value::from_array(total_step_array.clone())?;

      let vector_est_outputs = self.vector_est_ort.run(ort::inputs! {
        "noisy_latent" => &xt_value,
        "text_emb" => &text_emb_value,
        "style_ttl" => &style_ttl_value,
        "latent_mask" => &latent_mask_value,
        "text_mask" => &text_mask_value2,
        "current_step" => &current_step_value,
        "total_step" => &total_step_value,
      })?;

      let (denoised_shape, denoised_data) =
        vector_est_outputs["denoised_latent"].try_extract_tensor::<f32>()?;
      xt = Array3::from_shape_vec(
        (
          denoised_shape[0] as usize,
          denoised_shape[1] as usize,
          denoised_shape[2] as usize,
        ),
        denoised_data.to_vec(),
      )?;
    }

    // Generate waveform
    let final_latent_value = Value::from_array(xt)?;
    let vocoder_outputs = self.vocoder_ort.run(ort::inputs! {
      "latent" => &final_latent_value
    })?;

    let (_, wav_data) = vocoder_outputs["wav_tts"].try_extract_tensor::<f32>()?;
    let wav: Vec<f32> = wav_data.to_vec();

    Ok((wav, duration))
  }

  /// Synthesize a (possibly long) text. The text is split into
  /// sentence-aware chunks, each chunk is synthesized separately and the
  /// results are concatenated with `silence_duration` seconds of silence in
  /// between. Returns the samples and the total duration in seconds.
  pub fn call(
    &mut self,
    text: &str,
    lang: &str,
    style: &Style,
    total_step: usize,
    speed: f32,
    silence_duration: f32,
  ) -> Result<(Vec<f32>, f32)> {
    let chunks = chunk_text(text, Some(max_chunk_length(lang)));

    let mut wav_cat: Vec<f32> = Vec::new();
    let mut dur_cat: f32 = 0.0;
    let mut first = true;

    for chunk in chunks.iter() {
      if chunk.trim().is_empty() {
        continue;
      }
      let (wav, duration) = timer(
        &format!("synthesize chunk ({} chars)", chunk.chars().count()),
        || {
          self._infer(
            std::slice::from_ref(chunk),
            &[lang.to_string()],
            style,
            total_step,
            speed,
          )
        },
      )?;

      // The vocoder emits whole latent chunks; cut the tail down to the
      // predicted duration.
      let dur = duration[0];
      let wav_len = (self.sample_rate as f32 * dur) as usize;
      let wav_chunk = &wav[..wav_len.min(wav.len())];

      if first {
        wav_cat.extend_from_slice(wav_chunk);
        dur_cat = dur;
        first = false;
      } else {
        let silence_len = (silence_duration * self.sample_rate as f32) as usize;
        wav_cat.extend(std::iter::repeat_n(0.0f32, silence_len));
        wav_cat.extend_from_slice(wav_chunk);
        dur_cat += silence_duration + dur;
      }
    }

    // Sanitize: no NaN/Inf, keep within [-1, 1]
    for s in wav_cat.iter_mut() {
      *s = if s.is_finite() {
        s.clamp(-1.0, 1.0)
      } else {
        0.0
      };
    }

    Ok((wav_cat, dur_cat))
  }

  /// Batched single-shot inference (no chunking). Returns the raw vocoder
  /// output and the per item durations in seconds.
  pub fn batch(
    &mut self,
    text_list: &[String],
    lang_list: &[String],
    style: &Style,
    total_step: usize,
    speed: f32,
  ) -> Result<(Vec<f32>, Vec<f32>)> {
    self._infer(text_list, lang_list, style, total_step, speed)
  }
}

// ============================================================================
// Component Loading Functions
// ============================================================================

/// Load voice style from JSON files
pub fn load_voice_style(voice_style_paths: &[String], verbose_flag: bool) -> Result<Style> {
  let bsz = voice_style_paths.len();
  if bsz == 0 {
    bail!("No voice style path given");
  }

  // Read first file to get dimensions
  let first_file = File::open(&voice_style_paths[0])
    .with_context(|| format!("Failed to open voice style file {}", voice_style_paths[0]))?;
  let first_reader = BufReader::new(first_file);
  let first_data: VoiceStyleData = serde_json::from_reader(first_reader)?;

  let ttl_dims = &first_data.style_ttl.dims;
  let dp_dims = &first_data.style_dp.dims;

  let ttl_dim1 = ttl_dims[1];
  let ttl_dim2 = ttl_dims[2];
  let dp_dim1 = dp_dims[1];
  let dp_dim2 = dp_dims[2];

  // Pre-allocate arrays with full batch size
  let ttl_size = bsz * ttl_dim1 * ttl_dim2;
  let dp_size = bsz * dp_dim1 * dp_dim2;
  let mut ttl_flat = vec![0.0f32; ttl_size];
  let mut dp_flat = vec![0.0f32; dp_size];

  // Fill in the data
  for (i, path) in voice_style_paths.iter().enumerate() {
    let file =
      File::open(path).with_context(|| format!("Failed to open voice style file {}", path))?;
    let reader = BufReader::new(file);
    let data: VoiceStyleData = serde_json::from_reader(reader)?;

    // Flatten TTL data
    let ttl_offset = i * ttl_dim1 * ttl_dim2;
    let mut idx = 0;
    for batch in &data.style_ttl.data {
      for row in batch {
        for &val in row {
          ttl_flat[ttl_offset + idx] = val;
          idx += 1;
        }
      }
    }

    // Flatten DP data
    let dp_offset = i * dp_dim1 * dp_dim2;
    idx = 0;
    for batch in &data.style_dp.data {
      for row in batch {
        for &val in row {
          dp_flat[dp_offset + idx] = val;
          idx += 1;
        }
      }
    }
  }

  let ttl_style = Array3::from_shape_vec((bsz, ttl_dim1, ttl_dim2), ttl_flat)?;
  let dp_style = Array3::from_shape_vec((bsz, dp_dim1, dp_dim2), dp_flat)?;

  if verbose_flag && verbose() {
    println!("Loaded {} voice styles\n", bsz);
  }

  Ok(Style {
    ttl: ttl_style,
    dp: dp_style,
  })
}

/// Load TTS components from `<onnx_dir>/{*.onnx,tts.json,unicode_indexer.json}`.
/// `use_gpu = false` loads on the CPU, `true` requires a GPU (see
/// [`load_text_to_speech_on`] and [`crate::Device`] for automatic selection).
pub fn load_text_to_speech(onnx_dir: &str, use_gpu: bool) -> Result<TextToSpeech> {
  let device = if use_gpu { Device::Gpu } else { Device::Cpu };
  load_text_to_speech_on(onnx_dir, device)
}

/// Load TTS components on the given [`Device`] (`Device::Auto` = GPU when
/// compiled in and available, otherwise CPU).
pub fn load_text_to_speech_on(onnx_dir: &str, device: Device) -> Result<TextToSpeech> {
  let backend = resolve_backend(device)?;
  let onnx_dir = Path::new(onnx_dir);
  let cfgs = load_cfgs(onnx_dir)?;

  let dp_path = onnx_dir.join("duration_predictor.onnx");
  let text_enc_path = onnx_dir.join("text_encoder.onnx");
  let vector_est_path = onnx_dir.join("vector_estimator.onnx");
  let vocoder_path = onnx_dir.join("vocoder.onnx");

  let dp_ort = load_session(&dp_path, &backend)?;
  let text_enc_ort = load_session(&text_enc_path, &backend)?;
  let vector_est_ort = load_session(&vector_est_path, &backend)?;
  let vocoder_ort = load_session(&vocoder_path, &backend)?;

  let unicode_indexer_path = onnx_dir.join("unicode_indexer.json");
  let text_processor = UnicodeProcessor::new(&unicode_indexer_path)?;

  if verbose() {
    println!(
      "Supertonic 3 model loaded from {} on {}",
      onnx_dir.display(),
      backend
    );
  }

  let mut tts = TextToSpeech::new(
    cfgs,
    text_processor,
    dp_ort,
    text_enc_ort,
    vector_est_ort,
    vocoder_ort,
  );
  tts.backend = backend;
  Ok(tts)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn preprocess_wraps_with_language_tags() {
    assert_eq!(preprocess_text("hola", "es").unwrap(), "<es>hola.</es>");
    assert_eq!(
      preprocess_text("Hi 😀 there", "en").unwrap(),
      "<en>Hi there.</en>"
    );
    assert!(preprocess_text("x", "xx").is_err());
    assert!(preprocess_text("x", "na").is_ok());
  }

  #[test]
  fn supported_langs_exclude_language_agnostic() {
    assert!(!SUPPORTED_LANGS.contains(&"na"));
    assert_eq!(SUPPORTED_LANGS.len(), 31);
    assert_eq!(AVAILABLE_LANGS.len(), 32);
    for l in SUPPORTED_LANGS {
      assert!(AVAILABLE_LANGS.contains(l));
      assert!(language_name(l).is_some(), "missing name for {}", l);
    }
  }

  #[test]
  fn chunking_respects_max_len() {
    let long = "This is a sentence. ".repeat(40);
    let chunks = chunk_text(&long, Some(100));
    assert!(chunks.len() > 1);
    for c in &chunks {
      assert!(c.len() <= 100, "chunk too long: {}", c.len());
    }
    assert_eq!(
      chunk_text("Dr. Smith went home. He slept.", Some(300)).len(),
      1
    );
    assert_eq!(max_chunk_length("ko"), 120);
    assert_eq!(max_chunk_length("en"), 300);
  }
}
