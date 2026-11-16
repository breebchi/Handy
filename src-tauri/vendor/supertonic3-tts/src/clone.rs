// ============================================================================
// Voice cloning: invert a reference recording into Supertonic 3 style
// tensors (`style_ttl` + `style_dp`), fully offline and on the CPU.
//
// A Supertonic 3 voice is just two tensors that condition the text encoder /
// vector estimator (`style_ttl`, 50 x 256) and the duration predictor
// (`style_dp`, 8 x 16). Cloning therefore means searching for tensors that
// make the model speak like the reference. ONNX Runtime is inference only
// (no gradients), so the search is gradient free: the synthesized audio of a
// few probe sentences is compared with the reference (log-mel soft-DTW on
// the transcript, spectral envelope and pitch statistics, tempo) and the
// style tensors are refined with a natural evolution strategy in three
// stages of decreasing constraint (global preset mixing, per-row preset
// mixing, free per-element refinement). Every preset row has unit norm, and
// the search keeps the tensors on that manifold.
// ============================================================================

use crate::features::{
  denoise, l1_mean, load_wav, mel_dtw_distance, mel_statistics, pitch_stats, rms, speech_bounds,
  Audio, MelAnalyzer, PitchStats,
};
use crate::helper::{
  is_valid_lang, load_voice_style, save_voice_style, Style, TextToSpeech, VOICE_STYLES,
};
use crate::speaker::{cosine, SpeakerEncoder};
use anyhow::{anyhow, bail, Result};
use ndarray::Array3;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::path::{Path, PathBuf};

/// Denoising steps used while cloning by default.
pub const DEFAULT_CLONE_VOICE_QUALITY: usize = 8;

/// Speech RMS under which the silence barrier kicks in.
const SILENCE_RMS: f32 = 0.02;

/// Soft-DTW smoothing.
const DTW_GAMMA: f32 = 0.1;

/// Log-mel frames are averaged over this many frames before alignment.
const MEL_POOL: usize = 4;

/// Energy threshold (relative to the loudest 10 ms) used to find the speech
/// region of the reference.
const SPEECH_THRESHOLD: f32 = 0.02;

/// Longest trailing pause (seconds) kept in the tempo target.
const MAX_TRAILING_PAUSE: f32 = 0.5;

/// Transcript windows longer than this trigger a warning: the transcript is
/// synthesized as one utterance, so it should be a sentence or two.
pub const MAX_TRANSCRIPT_SECONDS: f32 = 15.0;

/// Built-in probe sentences for `lang` (`en`, `es`, `fr`, `de`, `it`, `pt`),
/// empty for other languages. Drawn at random (next to the reference
/// transcript) to keep the cloned voice consistent on unseen text.
pub fn probe_bank_for(lang: &str) -> &'static [&'static str] {
  match lang {
    "en" => PROBE_BANK,
    "es" => PROBE_BANK_ES,
    "fr" => PROBE_BANK_FR,
    "de" => PROBE_BANK_DE,
    "it" => PROBE_BANK_IT,
    "pt" => PROBE_BANK_PT,
    _ => &[],
  }
}

/// Spanish probe sentences.
pub const PROBE_BANK_ES: &[&str] = &[
  "Por favor, deja tu mensaje después de la señal y te llamaré más tarde.",
  "La reunión se ha movido al jueves por la tarde a las tres.",
  "Nunca pensé que el jardín pudiera verse tan bonito en primavera.",
  "¿Me puedes decir cómo llegar a la estación de tren desde aquí?",
  "Cada mañana camina junto al río antes de que la ciudad despierte.",
  "La receta lleva dos huevos, una taza de harina y una pizca de sal.",
  "Deberíamos mirar el pronóstico del tiempo antes de planear la excursión.",
  "Mi hermano colecciona radios antiguas y las repara los fines de semana.",
  "El museo cierra temprano los domingos, así que mejor vamos mañana.",
  "Gracias a todos por venir esta noche, significa mucho para nosotros.",
  "Gira a la izquierda en el segundo semáforo y la biblioteca está a la derecha.",
  "Llovió toda la noche, pero por la mañana el cielo estaba completamente despejado.",
];

/// French probe sentences.
pub const PROBE_BANK_FR: &[&str] = &[
  "Merci de laisser votre message après le signal, je vous rappellerai.",
  "La réunion a été déplacée à jeudi après-midi, à trois heures.",
  "Je n'aurais jamais cru que le jardin serait aussi beau au printemps.",
  "Pouvez-vous m'indiquer le chemin de la gare à partir d'ici ?",
  "Chaque matin, elle marche le long de la rivière avant que la ville ne s'éveille.",
  "La recette demande deux œufs, une tasse de farine et une pincée de sel.",
  "Nous devrions consulter la météo avant de préparer la randonnée.",
  "Mon frère collectionne de vieilles radios et les répare le week-end.",
  "Le musée ferme tôt le dimanche, allons-y demain plutôt.",
  "Il a plu toute la nuit, mais au matin le ciel était parfaitement dégagé.",
];

/// German probe sentences.
pub const PROBE_BANK_DE: &[&str] = &[
  "Bitte hinterlassen Sie nach dem Signalton eine Nachricht, ich rufe zurück.",
  "Die Besprechung wurde auf Donnerstagnachmittag um drei Uhr verschoben.",
  "Ich hätte nie gedacht, dass der Garten im Frühling so schön aussieht.",
  "Können Sie mir sagen, wie ich von hier zum Bahnhof komme?",
  "Jeden Morgen geht sie am Fluss entlang, bevor die Stadt erwacht.",
  "Das Rezept braucht zwei Eier, eine Tasse Mehl und eine Prise Salz.",
  "Wir sollten die Wettervorhersage prüfen, bevor wir die Wanderung planen.",
  "Mein Bruder sammelt alte Radios und repariert sie am Wochenende.",
  "Das Museum schließt sonntags früh, also gehen wir lieber morgen.",
  "Es hat die ganze Nacht geregnet, aber am Morgen war der Himmel klar.",
];

/// Italian probe sentences.
pub const PROBE_BANK_IT: &[&str] = &[
  "Per favore lascia un messaggio dopo il segnale e ti richiamerò.",
  "La riunione è stata spostata a giovedì pomeriggio alle tre.",
  "Non avrei mai pensato che il giardino fosse così bello in primavera.",
  "Mi può dire come arrivare alla stazione da qui?",
  "Ogni mattina cammina lungo il fiume prima che la città si svegli.",
  "La ricetta richiede due uova, una tazza di farina e un pizzico di sale.",
  "Dovremmo controllare le previsioni prima di organizzare l'escursione.",
  "Mio fratello colleziona vecchie radio e le ripara nel fine settimana.",
  "Il museo chiude presto la domenica, quindi andiamo domani.",
  "Ha piovuto tutta la notte, ma al mattino il cielo era perfettamente sereno.",
];

/// Portuguese probe sentences.
pub const PROBE_BANK_PT: &[&str] = &[
  "Por favor, deixe a sua mensagem depois do sinal e eu ligo de volta.",
  "A reunião foi adiada para quinta-feira à tarde, às três horas.",
  "Nunca imaginei que o jardim pudesse ficar tão bonito na primavera.",
  "Pode dizer-me como chegar à estação de comboios a partir daqui?",
  "Todas as manhãs ela caminha ao longo do rio antes de a cidade acordar.",
  "A receita leva dois ovos, uma chávena de farinha e uma pitada de sal.",
  "Devíamos ver a previsão do tempo antes de planear a caminhada.",
  "O meu irmão coleciona rádios antigos e repara-os ao fim de semana.",
  "O museu fecha cedo aos domingos, por isso vamos amanhã.",
  "Choveu a noite toda, mas de manhã o céu estava completamente limpo.",
];

/// Built-in English probe sentences.
pub const PROBE_BANK: &[&str] = &[
  "Please leave your message after the tone and I will call you back.",
  "The meeting has been moved to Thursday afternoon at three o'clock.",
  "I never expected the garden to look this beautiful in early spring.",
  "Could you tell me how to get to the train station from here?",
  "Every morning she walks along the river before the city wakes up.",
  "The recipe calls for two eggs, a cup of flour and a pinch of salt.",
  "We should probably check the forecast before planning the hike.",
  "My brother collects old radios and repairs them on the weekend.",
  "The museum closes early on Sundays, so let's go tomorrow instead.",
  "Thank you all for coming tonight, it really means a lot to us.",
  "A small cafe opened last week just around the corner from my office.",
  "Turn left at the second light and the library is on your right.",
  "He read the letter twice and then put it carefully back in the envelope.",
  "The children built a sandcastle while their parents watched from the shade.",
  "I think the blue one suits you better than the grey one, honestly.",
  "Remember to water the plants and feed the cat while we are away.",
  "The orchestra played the final movement with remarkable energy.",
  "It rained all night, but by morning the sky was perfectly clear.",
  "Our flight was delayed, so we spent the evening exploring the old town.",
  "She has been learning to play the piano for almost three years now.",
  "The documentary explains how bees communicate through dance.",
  "Let me know if you need anything else before the end of the day.",
  "The bridge was built more than a century ago and still carries traffic.",
  "Nobody remembered to bring a map, so we followed the river instead.",
];

/// Weights of the loss terms combined while cloning (0 disables a term).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossWeights {
  /// Speaker identity: `1 - cosine` between the speaker embeddings of the
  /// synthesized probes and of the reference (needs a speaker embedding
  /// model, see `CloneOptions::speaker_model`). The term that makes the
  /// clone sound like the same person.
  pub speaker: f32,
  /// Log-mel soft-DTW between the synthesized transcript and the reference
  /// (needs `reference_text`). Dense timbre + prosody signal.
  pub mel: f32,
  /// Spectral envelope statistics (mean / std of the log-mel bands over
  /// speech frames), text independent.
  pub spectrum: f32,
  /// Pitch statistics (mean / spread of log F0), text independent.
  pub pitch: f32,
  /// Tempo: predicted duration of the transcript vs the reference length
  /// (needs `reference_text`). The only term steering `style_dp`.
  pub duration: f32,
  /// Barrier that keeps the synthesized speech from collapsing to silence.
  pub silence: f32,
  /// L2 pull of the free refinement back toward its starting point.
  pub prior: f32,
}

impl Default for LossWeights {
  fn default() -> Self {
    LossWeights {
      speaker: 4.0,
      mel: 1.0,
      spectrum: 1.0,
      pitch: 2.0,
      duration: 0.8,
      silence: 1.0,
      prior: 0.0,
    }
  }
}

/// Voice cloning settings. `Default` gives a reasonable CPU budget; raise the
/// iteration counts for a closer match.
#[derive(Debug, Clone)]
pub struct CloneOptions {
  /// Language of the reference / probe sentences (`na` for language
  /// agnostic). Default `en`.
  pub language: String,
  /// Transcript of the reference recording. Strongly recommended: enables
  /// the log-mel alignment and tempo terms. Without it only the text
  /// independent terms (spectrum, pitch, silence) are used.
  pub reference_text: Option<String>,
  /// `(start, end)` seconds of the reference that match `reference_text`.
  /// Default: the whole recording (leading / trailing silence trimmed).
  /// For a long recording, select the sentence the transcript covers here;
  /// the timbre and pitch statistics still use the whole recording.
  pub reference_window: Option<(f32, f32)>,
  /// Preset name (`M1`..`F5`) or voice style JSON path to start from.
  /// `None` (default) selects the base automatically: all ten male and
  /// female presets are scored against the reference and the closest one
  /// is used.
  pub init_voice: Option<String>,
  /// Iterations of stage 1: global mixing of the presets (20 parameters).
  pub mix_iterations: usize,
  /// Iterations of stage 2: per-row mixing of the presets (580 parameters).
  pub row_iterations: usize,
  /// Iterations of stage 3: free per-element refinement (12928 parameters).
  pub iterations: usize,
  /// Step size of the optimizer.
  pub learning_rate: f32,
  /// Relative size of the random perturbations used to estimate gradients.
  pub perturbation: f32,
  /// Sentences synthesized per loss evaluation: the transcript (when given)
  /// plus random probe sentences.
  pub probe_count: usize,
  /// Custom probe sentences (in `language`). Empty = built-in English bank
  /// for `en`, no extra probes otherwise.
  pub probe_bank: Vec<String>,
  /// Redraw the random probe sentences (and the transcript's diffusion
  /// noise) every this many iterations.
  pub resample_every: usize,
  /// Denoising steps used while cloning (5 fastest .. 12 best).
  pub voice_quality: usize,
  /// Random seed.
  pub seed: u64,
  /// Loss term weights.
  pub weights: LossWeights,
  /// Path of a speaker embedding ONNX model (3D-Speaker / WeSpeaker export,
  /// see `scripts/download-model.sh`). `None` disables the speaker identity
  /// term, which makes clones a lot less faithful.
  pub speaker_model: Option<PathBuf>,
  /// Strength (0..=1) of the spectral gating denoiser applied to the
  /// reference before analysis (0 = off).
  pub denoise: f32,
  /// Device for the speaker embedding model (the TTS model uses the
  /// engine's device). `Device::Auto` = GPU when compiled in and available.
  pub device: crate::Device,
}

impl Default for CloneOptions {
  fn default() -> Self {
    CloneOptions {
      language: "en".to_string(),
      reference_text: None,
      reference_window: None,
      init_voice: None,
      mix_iterations: 40,
      row_iterations: 60,
      iterations: 100,
      learning_rate: 0.05,
      perturbation: 0.05,
      probe_count: 2,
      probe_bank: Vec::new(),
      resample_every: 20,
      voice_quality: DEFAULT_CLONE_VOICE_QUALITY,
      seed: 42,
      weights: LossWeights::default(),
      speaker_model: None,
      denoise: 0.5,
      device: crate::Device::Auto,
    }
  }
}

/// Which optimisation stage a progress report refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloneStage {
  /// Evaluating the presets to pick the warm start.
  SelectPreset,
  /// Global mixing of the presets.
  MixPresets,
  /// Per-row mixing of the presets.
  MixRows,
  /// Free per-element refinement.
  Refine,
}

impl std::fmt::Display for CloneStage {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let s = match self {
      CloneStage::SelectPreset => "select-preset",
      CloneStage::MixPresets => "mix-presets",
      CloneStage::MixRows => "mix-rows",
      CloneStage::Refine => "refine",
    };
    f.write_str(s)
  }
}

/// Progress report passed to the callback of
/// [`crate::TtsEngine::clone_voice_with_progress`] after every iteration.
#[derive(Debug, Clone)]
pub struct CloneProgress<'a> {
  pub stage: CloneStage,
  /// Iteration within the stage (1-based).
  pub iteration: usize,
  /// Total iterations of the stage.
  pub total: usize,
  /// Loss of the current parameters.
  pub loss: f32,
  /// Best loss seen so far (all stages).
  pub best_loss: f32,
  /// Individual (unweighted) loss terms of the current evaluation.
  pub terms: Vec<(&'static str, f32)>,
  /// Best style so far (save it to checkpoint).
  pub best_style: &'a Style,
  /// Seconds elapsed since cloning started.
  pub elapsed: f32,
}

/// Result of voice cloning.
#[derive(Debug, Clone)]
pub struct ClonedVoice {
  /// The cloned style tensors, ready for synthesis.
  pub style: Style,
  /// Execution provider the speaker embedding model ran on (`None` without
  /// a speaker model). The TTS networks run on the engine's backend.
  pub speaker_backend: Option<crate::Backend>,
  /// Final (best) loss.
  pub loss: f32,
  /// Loss of the warm start preset, for comparison.
  pub initial_loss: f32,
  /// Speaker embedding cosine similarity between the cloned voice and the
  /// reference (`None` without a speaker model). Rough guide with the
  /// bundled model: different voices score below ~0.3, the same voice
  /// above ~0.7.
  pub speaker_similarity: Option<f32>,
  /// Name / path of the warm start voice.
  pub init_voice: String,
  /// Loss after every iteration, all stages concatenated.
  pub history: Vec<f32>,
  /// Options used.
  pub options: CloneOptions,
  /// Wall-clock seconds spent.
  pub elapsed: f32,
}

impl ClonedVoice {
  /// Save as a voice style JSON usable as `--voice <path>` /
  /// `synthesize_with_options(.., Some(path), ..)`.
  pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
    let metadata = serde_json::json!({
      "source": "supertonic3-tts voice cloning",
      "init_voice": self.init_voice,
      "loss": self.loss,
      "initial_loss": self.initial_loss,
      "speaker_similarity": self.speaker_similarity,
      "language": self.options.language,
      "reference_text": self.options.reference_text,
      "iterations": [self.options.mix_iterations, self.options.row_iterations, self.options.iterations],
      "voice_quality": self.options.voice_quality,
      "seed": self.options.seed,
      "elapsed_seconds": self.elapsed,
    });
    save_voice_style(path, &self.style, Some(metadata))
  }
}

// ============================================================================
// Reference analysis
// ============================================================================

struct ReferenceTargets {
  rms: f32,
  /// Speech region of the reference (denoised, model sample rate), kept for
  /// the speaker embedding.
  clip: Vec<f32>,
  mel_frames: Option<Vec<Vec<f32>>>, // pooled log-mel of the transcript window
  mel_mean: Vec<f32>,
  mel_std: Vec<f32>,
  pitch: Option<PitchStats>,
  duration_target: Option<f32>,
}

fn analyse_reference(
  reference: &Audio,
  options: &CloneOptions,
  mel: &MelAnalyzer,
) -> Result<ReferenceTargets> {
  let sr = mel.sample_rate();
  let audio = reference.resampled(sr);
  let samples: Vec<f32> = if options.denoise > 0.0 {
    denoise(&audio.samples, options.denoise)
  } else {
    audio.samples.clone()
  };
  let samples = &samples;
  let n = samples.len();
  if n < sr as usize / 4 {
    bail!(
      "reference recording is too short ({:.2} s)",
      audio.duration()
    );
  }

  // Speech region of the whole recording: the text independent statistics
  // (spectral envelope, pitch) use all of it, so a longer recording helps.
  let (clip_start, clip_end) = speech_bounds(samples, sr, SPEECH_THRESHOLD)
    .ok_or_else(|| anyhow!("reference recording is silent"))?;
  let clip = &samples[clip_start..clip_end];
  if clip.len() < sr as usize / 4 {
    bail!(
      "reference speech region is too short ({:.2} s)",
      clip.len() as f32 / sr as f32
    );
  }

  // Transcript window: the part of the recording that `reference_text`
  // transcribes (whole speech region unless `reference_window` is given).
  let (speech_start, speech_end) = match options.reference_window {
    Some((a, b)) => {
      let s = ((a.max(0.0)) * sr as f32) as usize;
      let e = ((b * sr as f32) as usize).min(n);
      if e <= s {
        bail!("invalid reference window {}..{}", a, b);
      }
      (s, e)
    }
    None => (clip_start, clip_end),
  };
  let window = &samples[speech_start..speech_end];
  if window.len() < sr as usize / 4 {
    bail!(
      "reference window is too short ({:.2} s)",
      window.len() as f32 / sr as f32
    );
  }

  let ref_rms = rms(window).max(1e-4);
  let frames = mel.log_mel(clip);
  let (mel_mean, mel_std) =
    mel_statistics(&frames).ok_or_else(|| anyhow!("reference has no speech frames"))?;
  let pitch = pitch_stats(clip, sr);

  let (mel_frames, duration_target) = if options.reference_text.is_some() {
    let window_seconds = window.len() as f32 / sr as f32;
    if window_seconds > MAX_TRANSCRIPT_SECONDS {
      eprintln!(
        "Warning: the transcript window is {:.1} s long; the transcript is synthesized as one \
         utterance, which works best for a sentence or two (up to ~{:.0} s). Pass \
         reference_window / --ref-window to select the sentence matching the transcript (the \
         rest of the recording is still used for the timbre and pitch statistics).",
        window_seconds, MAX_TRANSCRIPT_SECONDS
      );
    }
    let trailing = ((n - speech_end) as f32 / sr as f32).min(MAX_TRAILING_PAUSE);
    let dur_target = window_seconds + trailing;
    (Some(mel.log_mel_pooled(window, MEL_POOL)), Some(dur_target))
  } else {
    (None, None)
  };

  Ok(ReferenceTargets {
    rms: ref_rms,
    clip: clip.to_vec(),
    mel_frames,
    mel_mean,
    mel_std,
    pitch,
    duration_target,
  })
}

// ============================================================================
// Style parametrisations (all keep every row on the unit sphere)
// ============================================================================

fn normalize_rows(a: &mut Array3<f32>) {
  let (b, r, c) = a.dim();
  for bi in 0..b {
    for ri in 0..r {
      let mut norm = 0.0f32;
      for ci in 0..c {
        norm += a[[bi, ri, ci]] * a[[bi, ri, ci]];
      }
      let norm = norm.sqrt();
      if norm > 1e-8 {
        for ci in 0..c {
          a[[bi, ri, ci]] /= norm;
        }
      }
    }
  }
}

/// Mean cosine similarity between the rows of two styles' `ttl` tensors.
pub fn style_similarity(a: &Style, b: &Style) -> f32 {
  let (_, r, c) = a.ttl.dim();
  if a.ttl.dim() != b.ttl.dim() || r == 0 {
    return 0.0;
  }
  let mut total = 0.0f32;
  for ri in 0..r {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for ci in 0..c {
      let x = a.ttl[[0, ri, ci]];
      let y = b.ttl[[0, ri, ci]];
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    total += dot / (na.sqrt() * nb.sqrt()).max(1e-8);
  }
  total / r as f32
}

/// Style = normalised weighted sum of basis styles, one weight vector per
/// tensor shared by all rows.
fn mix_global(basis: &[Style], w_ttl: &[f32], w_dp: &[f32]) -> Style {
  let mut ttl = Array3::<f32>::zeros(basis[0].ttl.dim());
  let mut dp = Array3::<f32>::zeros(basis[0].dp.dim());
  for (k, b) in basis.iter().enumerate() {
    ttl.scaled_add(w_ttl[k], &b.ttl);
    dp.scaled_add(w_dp[k], &b.dp);
  }
  normalize_rows(&mut ttl);
  normalize_rows(&mut dp);
  Style { ttl, dp }
}

/// Style = normalised weighted sum of basis styles with separate weights
/// for every row: `w_ttl[row * k + basis]`, `w_dp[row * k + basis]`.
fn mix_rows(basis: &[Style], w_ttl: &[f32], w_dp: &[f32]) -> Style {
  let k = basis.len();
  let (_, rt, ct) = basis[0].ttl.dim();
  let (_, rd, cd) = basis[0].dp.dim();
  let mut ttl = Array3::<f32>::zeros(basis[0].ttl.dim());
  let mut dp = Array3::<f32>::zeros(basis[0].dp.dim());
  for ri in 0..rt {
    for (bi, b) in basis.iter().enumerate() {
      let w = w_ttl[ri * k + bi];
      if w != 0.0 {
        for ci in 0..ct {
          ttl[[0, ri, ci]] += w * b.ttl[[0, ri, ci]];
        }
      }
    }
  }
  for ri in 0..rd {
    for (bi, b) in basis.iter().enumerate() {
      let w = w_dp[ri * k + bi];
      if w != 0.0 {
        for ci in 0..cd {
          dp[[0, ri, ci]] += w * b.dp[[0, ri, ci]];
        }
      }
    }
  }
  normalize_rows(&mut ttl);
  normalize_rows(&mut dp);
  Style { ttl, dp }
}

/// Style = normalised (init + delta), `delta` flattened as ttl then dp.
fn apply_delta(init: &Style, delta: &[f32]) -> Style {
  let mut ttl = init.ttl.clone();
  let mut dp = init.dp.clone();
  let n_ttl = ttl.len();
  for (v, d) in ttl.iter_mut().zip(&delta[..n_ttl]) {
    *v += d;
  }
  for (v, d) in dp.iter_mut().zip(&delta[n_ttl..]) {
    *v += d;
  }
  normalize_rows(&mut ttl);
  normalize_rows(&mut dp);
  Style { ttl, dp }
}

// ============================================================================
// The cloner
// ============================================================================

struct Probe {
  text: String,
  noise: Array3<f32>,
}

/// Loss evaluation result.
#[derive(Debug, Clone)]
struct Evaluation {
  total: f32,
  terms: Vec<(&'static str, f32)>,
  speaker_cos: Option<f32>,
}

/// Gradient-free optimiser state shared by the three stages: a natural
/// evolution strategy with antithetic samples and Adam-style step
/// normalisation.
struct Optimizer {
  m: Vec<f32>,
  v: Vec<f32>,
  t: usize,
}

impl Optimizer {
  fn new(n: usize) -> Self {
    Optimizer {
      m: vec![0.0; n],
      v: vec![0.0; n],
      t: 0,
    }
  }

  fn step(&mut self, params: &mut [f32], grad: &[f32], lr: f32) {
    self.t += 1;
    let (b1, b2, eps) = (0.8f32, 0.99f32, 1e-8f32);
    let c1 = 1.0 - b1.powi(self.t as i32);
    let c2 = 1.0 - b2.powi(self.t as i32);
    for i in 0..params.len() {
      self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
      self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];
      let mhat = self.m[i] / c1;
      let vhat = self.v[i] / c2;
      params[i] -= lr * mhat / (vhat.sqrt() + eps);
    }
  }
}

/// Drives the search. Create with [`VoiceCloner::new`], run with
/// [`VoiceCloner::run`]. `crate::TtsEngine::clone_voice` wraps this.
pub struct VoiceCloner<'a> {
  tts: &'a mut TextToSpeech,
  options: CloneOptions,
  targets: ReferenceTargets,
  mel: MelAnalyzer,
  presets: Vec<(String, Style)>,
  probe_bank: Vec<String>,
  probes: Vec<Probe>,
  speaker: Option<SpeakerEncoder>,
  reference_embedding: Option<Vec<f32>>,
  best_similarity: Option<f32>,
  rng: StdRng,
  start: std::time::Instant,
  history: Vec<f32>,
  best_loss: f32,
  best_style: Option<Style>,
}

impl<'a> VoiceCloner<'a> {
  /// Prepare cloning of `reference` (any sample rate, mono or not) with the
  /// presets found in `voice_styles_dir`.
  pub fn new(
    tts: &'a mut TextToSpeech,
    reference: &Audio,
    voice_styles_dir: &Path,
    options: CloneOptions,
  ) -> Result<Self> {
    if !is_valid_lang(&options.language) {
      bail!("Invalid language for cloning: {}", options.language);
    }
    if !(crate::MIN_VOICE_QUALITY..=crate::MAX_VOICE_QUALITY).contains(&options.voice_quality) {
      bail!(
        "Invalid voice_quality {}: must be between {} and {}",
        options.voice_quality,
        crate::MIN_VOICE_QUALITY,
        crate::MAX_VOICE_QUALITY
      );
    }
    if let Some(t) = &options.reference_text {
      if t.trim().is_empty() {
        bail!("reference_text is empty");
      }
    }
    let sr = tts.sample_rate as u32;
    let mel = MelAnalyzer::new(sr, 2048, 512, 80);
    let targets = analyse_reference(reference, &options, &mel)?;

    // Basis presets (+ the custom init voice when it is a file).
    let mut presets: Vec<(String, Style)> = Vec::new();
    for name in VOICE_STYLES {
      let p = voice_styles_dir.join(format!("{}.json", name));
      if p.is_file() {
        let style = load_voice_style(&[p.to_string_lossy().to_string()], false)?;
        presets.push((name.to_string(), style));
      }
    }
    if let Some(init) = &options.init_voice {
      if !VOICE_STYLES.contains(&init.as_str()) {
        let p = PathBuf::from(init);
        if !p.is_file() {
          bail!(
            "init voice {} is neither a preset ({}) nor a file",
            init,
            VOICE_STYLES.join(", ")
          );
        }
        let style = load_voice_style(&[p.to_string_lossy().to_string()], false)?;
        presets.push((init.clone(), style));
      } else if !presets.iter().any(|(n, _)| n == init) {
        bail!(
          "preset {} not found in {}",
          init,
          voice_styles_dir.display()
        );
      }
    }
    if presets.is_empty() {
      bail!(
        "no voice style presets found in {}",
        voice_styles_dir.display()
      );
    }

    let mut probe_bank: Vec<String> = options
      .probe_bank
      .iter()
      .filter(|s| !s.trim().is_empty())
      .cloned()
      .collect();
    if probe_bank.is_empty() {
      probe_bank = probe_bank_for(&options.language)
        .iter()
        .map(|s| s.to_string())
        .collect();
    }

    // Speaker embedding model (identity term).
    let (speaker, reference_embedding) = match (&options.speaker_model, options.weights.speaker) {
      (Some(path), w) if w > 0.0 => {
        let mut enc = SpeakerEncoder::on_device(path, options.device)?;
        if crate::helper::is_verbose() {
          println!(
            "Speaker embedding model {} loaded on {}",
            path.display(),
            enc.backend()
          );
        }
        let emb = enc.embed(&targets.clip, sr)?;
        (Some(enc), Some(emb))
      }
      _ => (None, None),
    };
    if options.reference_text.is_none() && probe_bank.is_empty() {
      bail!(
        "nothing to synthesize: give the transcript of the reference (reference_text) or \
         probe sentences in language '{}'",
        options.language
      );
    }
    if options.probe_count == 0 {
      bail!("probe_count must be at least 1");
    }

    Ok(VoiceCloner {
      tts,
      rng: StdRng::seed_from_u64(options.seed),
      options,
      targets,
      mel,
      presets,
      probe_bank,
      probes: Vec::new(),
      speaker,
      reference_embedding,
      best_similarity: None,
      start: std::time::Instant::now(),
      history: Vec::new(),
      best_loss: f32::INFINITY,
      best_style: None,
    })
  }

  fn draw_noise(&mut self, frames: usize) -> Array3<f32> {
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let dim = self.tts.latent_dim();
    let mut noise = Array3::<f32>::zeros((1, dim, frames));
    for v in noise.iter_mut() {
      *v = normal.sample(&mut self.rng);
    }
    noise
  }

  /// Pick the probe sentences (transcript first) and draw their diffusion
  /// noise. `style` is used to size the noise buffers.
  fn sample_probes(&mut self, style: &Style) -> Result<()> {
    let mut texts: Vec<String> = Vec::new();
    if let Some(t) = &self.options.reference_text {
      texts.push(t.clone());
    }
    let extra = self.options.probe_count.saturating_sub(texts.len());
    let mut bank = self.probe_bank.clone();
    bank.shuffle(&mut self.rng);
    texts.extend(bank.into_iter().take(extra));

    let chunk = self.tts.latent_chunk_size();
    let sr = self.tts.sample_rate as f32;
    let mut probes = Vec::with_capacity(texts.len());
    for text in texts {
      let dur = self
        .tts
        .predict_duration(&text, &self.options.language, style)?;
      let frames = ((dur * 2.0 * sr) / chunk as f32).ceil() as usize + 16;
      let noise = self.draw_noise(frames);
      probes.push(Probe { text, noise });
    }
    self.probes = probes;
    Ok(())
  }

  /// Synthesize every probe with `style` and compute the loss terms.
  fn evaluate(&mut self, style: &Style, prior: f32) -> Result<Evaluation> {
    let w = self.options.weights;
    let sr = self.tts.sample_rate as u32;
    let mut mel_term = 0.0f32;
    let mut dur_term = 0.0f32;
    let mut spectrum_term = 0.0f32;
    let mut pitch_term = 0.0f32;
    let mut silence_term = 0.0f32;
    let mut speaker_term = 0.0f32;
    let mut speaker_cos = 0.0f32;
    let n_probes = self.probes.len() as f32;
    let has_ref_text = self.options.reference_text.is_some();
    let use_speaker = self.speaker.is_some() && w.speaker > 0.0;

    for i in 0..self.probes.len() {
      let (wav, dur) = {
        let probe = &self.probes[i];
        self.tts.infer_single(
          &probe.text,
          &self.options.language,
          style,
          self.options.voice_quality,
          1.0,
          Some(&probe.noise),
        )?
      };
      if wav.is_empty() {
        silence_term += 1.0;
        spectrum_term += 2.0;
        pitch_term += 1.0;
        speaker_term += 1.0;
        continue;
      }
      if use_speaker {
        let target = self.reference_embedding.as_ref().unwrap();
        let cos = match self.speaker.as_mut().unwrap().embed(&wav, sr) {
          Ok(emb) => cosine(&emb, target),
          Err(_) => 0.0,
        };
        speaker_cos += cos;
        speaker_term += 1.0 - cos;
      }
      // Silence barrier on the raw output level.
      let level = rms(&wav);
      silence_term += (1.0 - level / SILENCE_RMS).max(0.0);

      // Level-match to the reference before any spectral comparison.
      let gain = self.targets.rms / level.max(1e-5);
      let scaled: Vec<f32> = wav.iter().map(|v| v * gain).collect();

      if w.spectrum > 0.0 {
        let frames = self.mel.log_mel(&scaled);
        match mel_statistics(&frames) {
          Some((mean, std)) => {
            spectrum_term +=
              l1_mean(&mean, &self.targets.mel_mean) + 0.5 * l1_mean(&std, &self.targets.mel_std);
          }
          None => spectrum_term += 2.0,
        }
      }
      if w.pitch > 0.0 {
        if let Some(target) = self.targets.pitch {
          match pitch_stats(&scaled, sr) {
            Some(p) => {
              let dm = p.mean_log2_f0 - target.mean_log2_f0;
              let ds = p.std_log2_f0 - target.std_log2_f0;
              pitch_term += dm * dm + ds * ds;
            }
            None => pitch_term += 1.0,
          }
        }
      }
      if i == 0 && has_ref_text {
        if w.mel > 0.0 {
          if let Some(ref_frames) = &self.targets.mel_frames {
            let frames = self.mel.log_mel_pooled(&scaled, MEL_POOL);
            mel_term = if frames.is_empty() {
              5.0
            } else {
              mel_dtw_distance(&frames, ref_frames, DTW_GAMMA)?
            };
          }
        }
        if w.duration > 0.0 {
          if let Some(target) = self.targets.duration_target {
            let r = dur / target - 1.0;
            dur_term = r * r;
          }
        }
      }
    }
    spectrum_term /= n_probes;
    pitch_term /= n_probes;
    silence_term /= n_probes;
    speaker_term /= n_probes;
    speaker_cos /= n_probes;

    let mut terms: Vec<(&'static str, f32)> = Vec::new();
    let mut total = 0.0f32;
    let mut push = |name: &'static str, weight: f32, value: f32| {
      if weight > 0.0 {
        terms.push((name, value));
        total += weight * value;
      }
    };
    if use_speaker {
      push("speaker", w.speaker, speaker_term);
    }
    if has_ref_text {
      push("mel", w.mel, mel_term);
      push("dur", w.duration, dur_term);
    }
    push("spectrum", w.spectrum, spectrum_term);
    push("pitch", w.pitch, pitch_term);
    push("silence", w.silence, silence_term);
    push("prior", w.prior, prior);
    if use_speaker {
      // Informational: the raw cosine similarity (not part of the total).
      terms.push(("cos", speaker_cos));
    }
    Ok(Evaluation {
      total,
      terms,
      speaker_cos: use_speaker.then_some(speaker_cos),
    })
  }

  fn track_best(&mut self, eval: &Evaluation, style: &Style) {
    if eval.total < self.best_loss {
      self.best_loss = eval.total;
      self.best_style = Some(style.clone());
      self.best_similarity = eval.speaker_cos;
    }
  }

  /// Re-score the best style on the current probe set (called after the
  /// probes are redrawn so that "best" always refers to the same terms).
  fn rescore_best(&mut self, prior: f32) -> Result<()> {
    if let Some(best) = self.best_style.clone() {
      let e = self.evaluate(&best, prior)?;
      self.best_loss = e.total;
      self.best_similarity = e.speaker_cos;
    }
    Ok(())
  }

  fn report(
    &self,
    callback: &mut dyn FnMut(&CloneProgress<'_>),
    stage: CloneStage,
    iteration: usize,
    total: usize,
    eval: &Evaluation,
  ) {
    if let Some(best) = &self.best_style {
      callback(&CloneProgress {
        stage,
        iteration,
        total,
        loss: eval.total,
        best_loss: self.best_loss,
        terms: eval.terms.clone(),
        best_style: best,
        elapsed: self.start.elapsed().as_secs_f32(),
      });
    }
  }

  /// One NES stage over a parameter vector. `build` maps parameters to a
  /// style, `prior` gives the prior term for a parameter vector.
  #[allow(clippy::too_many_arguments)]
  fn optimise_stage(
    &mut self,
    stage: CloneStage,
    params: &mut Vec<f32>,
    iterations: usize,
    sigma: f32,
    lr: f32,
    build: &dyn Fn(&[f32]) -> Style,
    prior: &dyn Fn(&[f32]) -> f32,
    callback: &mut dyn FnMut(&CloneProgress<'_>),
  ) -> Result<Style> {
    let n = params.len();
    let mut opt = Optimizer::new(n);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let mut current_style = build(params);
    // The stage starts from the best style so far; its loss on the current
    // probes is the value to beat.
    let mut stage_best_loss = self.best_loss;
    let mut stage_best_params = params.clone();
    let resample_every = self.options.resample_every.max(1);

    for it in 1..=iterations {
      if it > 1 && it % resample_every == 0 {
        // New probe sentences + noise: the loss landscape changes, so the
        // running best must be re-measured on the same terms.
        self.sample_probes(&current_style)?;
        self.rescore_best(prior(&stage_best_params))?;
        stage_best_loss = self.best_loss;
      }
      let eps: Vec<f32> = (0..n).map(|_| normal.sample(&mut self.rng)).collect();
      let plus: Vec<f32> = params
        .iter()
        .zip(&eps)
        .map(|(p, e)| p + sigma * e)
        .collect();
      let minus: Vec<f32> = params
        .iter()
        .zip(&eps)
        .map(|(p, e)| p - sigma * e)
        .collect();
      let style_plus = build(&plus);
      let style_minus = build(&minus);
      let eval_plus = self.evaluate(&style_plus, prior(&plus))?;
      let eval_minus = self.evaluate(&style_minus, prior(&minus))?;
      let (better_eval, better_params, better_style) = if eval_plus.total <= eval_minus.total {
        (&eval_plus, &plus, &style_plus)
      } else {
        (&eval_minus, &minus, &style_minus)
      };
      if better_eval.total < stage_best_loss {
        stage_best_loss = better_eval.total;
        stage_best_params = better_params.clone();
        self.track_best(better_eval, better_style);
      }

      // NES gradient estimate from the antithetic pair.
      let scale = (eval_plus.total - eval_minus.total) / (2.0 * sigma);
      let grad: Vec<f32> = eps.iter().map(|e| e * scale).collect();
      opt.step(params, &grad, lr);
      current_style = build(params);

      self.history.push(better_eval.total);
      self.report(callback, stage, it, iterations, better_eval);
    }
    // Return the best point of the stage (the final parameters are checked
    // too, on the current probes).
    if iterations > 0 {
      let e = self.evaluate(&current_style, prior(params))?;
      if e.total < stage_best_loss {
        stage_best_params = params.clone();
        self.track_best(&e, &current_style);
      }
    }
    *params = stage_best_params;
    Ok(build(params))
  }

  /// Run all stages and return the cloned voice.
  pub fn run(mut self, callback: &mut dyn FnMut(&CloneProgress<'_>)) -> Result<ClonedVoice> {
    let k = self.presets.len();
    let basis: Vec<Style> = self.presets.iter().map(|(_, s)| s.clone()).collect();

    // Stage 0: pick the warm start.
    let init_index = match &self.options.init_voice {
      Some(name) => self
        .presets
        .iter()
        .position(|(n, _)| n == name)
        .ok_or_else(|| anyhow!("init voice {} not loaded", name))?,
      None => 0,
    };
    self.sample_probes(&basis[init_index])?;
    let init_index = if self.options.init_voice.is_some() {
      let e = self.evaluate(&basis[init_index], 0.0)?;
      self.track_best(&e, &basis[init_index]);
      self.report(callback, CloneStage::SelectPreset, 1, 1, &e);
      init_index
    } else {
      let mut best = (f32::INFINITY, 0usize);
      for (i, candidate) in basis.iter().enumerate() {
        let e = self.evaluate(candidate, 0.0)?;
        self.track_best(&e, candidate);
        self.report(callback, CloneStage::SelectPreset, i + 1, k, &e);
        if e.total < best.0 {
          best = (e.total, i);
        }
      }
      best.1
    };
    let init_name = self.presets[init_index].0.clone();
    let initial_loss = self.best_loss;

    // Stage 1: global preset mixing.
    let mut w = vec![0.0f32; 2 * k];
    w[init_index] = 1.0;
    w[k + init_index] = 1.0;
    let lr = self.options.learning_rate;
    let sigma = self.options.perturbation;
    let mixed = {
      let basis_ref = &basis;
      let build = move |p: &[f32]| mix_global(basis_ref, &p[..k], &p[k..]);
      self.optimise_stage(
        CloneStage::MixPresets,
        &mut w,
        self.options.mix_iterations,
        sigma * 2.0,
        lr,
        &build,
        &|_| 0.0,
        callback,
      )?
    };

    // Stage 2: per-row preset mixing, starting from the global mix.
    let (_, rt, _) = basis[0].ttl.dim();
    let (_, rd, _) = basis[0].dp.dim();
    let mut wr = vec![0.0f32; rt * k + rd * k];
    for ri in 0..rt {
      for bi in 0..k {
        wr[ri * k + bi] = w[bi];
      }
    }
    for ri in 0..rd {
      for bi in 0..k {
        wr[rt * k + ri * k + bi] = w[k + bi];
      }
    }
    let mixed = if self.options.row_iterations > 0 {
      let basis_ref = &basis;
      let build = move |p: &[f32]| mix_rows(basis_ref, &p[..rt * k], &p[rt * k..]);
      self.optimise_stage(
        CloneStage::MixRows,
        &mut wr,
        self.options.row_iterations,
        sigma * 2.0,
        lr,
        &build,
        &|_| 0.0,
        callback,
      )?
    } else {
      mixed
    };

    // Stage 3: free refinement of every element.
    let final_style = if self.options.iterations > 0 {
      let init = mixed.clone();
      let n = init.ttl.len() + init.dp.len();
      let mut delta = vec![0.0f32; n];
      let build = move |p: &[f32]| apply_delta(&init, p);
      let prior = |p: &[f32]| p.iter().map(|d| d * d).sum::<f32>() / p.len().max(1) as f32;
      // Per-element perturbation so that a whole row moves by about `sigma`.
      let (_, _, ct) = basis[0].ttl.dim();
      let sigma_elem = sigma / (ct as f32).sqrt();
      self.optimise_stage(
        CloneStage::Refine,
        &mut delta,
        self.options.iterations,
        sigma_elem,
        lr * sigma_elem,
        &build,
        &prior,
        callback,
      )?
    } else {
      mixed
    };

    let style = self.best_style.clone().unwrap_or(final_style);
    let speaker_backend = self.speaker.as_ref().map(|e| e.backend().clone());
    Ok(ClonedVoice {
      style,
      speaker_backend,
      loss: self.best_loss,
      initial_loss,
      speaker_similarity: self.best_similarity,
      init_voice: init_name,
      history: self.history,
      options: self.options,
      elapsed: self.start.elapsed().as_secs_f32(),
    })
  }
}

/// Load a reference recording from a WAV file (helper re-exported for CLI /
/// library users).
pub fn load_reference<P: AsRef<Path>>(path: P) -> Result<Audio> {
  load_wav(path)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn style(rows: &[[f32; 4]]) -> Style {
    let mut ttl = Array3::<f32>::zeros((1, rows.len(), 4));
    for (r, row) in rows.iter().enumerate() {
      for c in 0..4 {
        ttl[[0, r, c]] = row[c];
      }
    }
    let mut dp = Array3::<f32>::zeros((1, 1, 2));
    dp[[0, 0, 0]] = 1.0;
    Style { ttl, dp }
  }

  #[test]
  fn mixing_keeps_unit_rows() {
    let a = style(&[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]);
    let b = style(&[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]);
    let m = mix_global(&[a.clone(), b.clone()], &[1.0, 1.0], &[1.0, 0.0]);
    for r in 0..2 {
      let n: f32 = (0..4).map(|c| m.ttl[[0, r, c]].powi(2)).sum();
      assert!((n - 1.0).abs() < 1e-5);
    }
    assert!((m.ttl[[0, 0, 0]] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
    let rows = mix_rows(&[a.clone(), b.clone()], &[1.0, 0.0, 0.0, 1.0], &[1.0, 0.0]);
    assert!((rows.ttl[[0, 0, 0]] - 1.0).abs() < 1e-6);
    assert!((rows.ttl[[0, 1, 3]] - 1.0).abs() < 1e-6);
    let d = apply_delta(&a, &[0.0; 8 + 2]);
    assert!((style_similarity(&a, &d) - 1.0).abs() < 1e-6);
    assert!(style_similarity(&a, &b).abs() < 1e-6);
  }

  #[test]
  fn optimizer_moves_against_gradient() {
    let mut o = Optimizer::new(2);
    let mut p = vec![1.0f32, -1.0];
    o.step(&mut p, &[1.0, -1.0], 0.1);
    assert!(p[0] < 1.0 && p[1] > -1.0);
  }

  #[test]
  fn defaults_are_sane() {
    let o = CloneOptions::default();
    assert_eq!(o.language, "en");
    assert!(o.probe_count >= 1);
    assert_eq!(o.weights, LossWeights::default());
    assert!(PROBE_BANK.len() >= 16);
  }
}
