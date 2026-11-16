// ============================================================================
// Speaker identity: Kaldi-compatible log mel filterbank front end and an
// ONNX speaker embedding model (WeSpeaker / 3D-Speaker export format, e.g.
// the models shipped with sherpa-onnx). Used by voice cloning as the
// "does it sound like the same person" term.
// ============================================================================

use crate::device::{load_session, resolve_backend, Backend, Device};
use crate::features::resample;
use anyhow::{anyhow, bail, Result};
use ndarray::Array3;
use ort::session::Session;
use ort::value::Value;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::path::Path;
use std::sync::Arc;

/// Options of the Kaldi fbank computation (defaults match sherpa-onnx).
#[derive(Debug, Clone)]
pub struct FbankOptions {
  pub sample_rate: u32,
  pub num_bins: usize,
  pub frame_length_ms: f32,
  pub frame_shift_ms: f32,
  pub low_freq: f32,
  /// Upper mel edge; `<= 0` means Nyquist + `high_freq` (Kaldi convention).
  pub high_freq: f32,
  pub preemph_coeff: f32,
  pub remove_dc_offset: bool,
  /// `false`: frames are centred and the edges reflected (Kaldi
  /// `snip_edges=false`).
  pub snip_edges: bool,
  /// Multiply the samples by 32768 (models trained on int16-scaled audio).
  pub scale_to_int16: bool,
  /// `"povey"` (Kaldi default, hann^0.85) or `"hamming"`.
  pub window_type: String,
}

impl Default for FbankOptions {
  fn default() -> Self {
    FbankOptions {
      sample_rate: 16000,
      num_bins: 80,
      frame_length_ms: 25.0,
      frame_shift_ms: 10.0,
      low_freq: 20.0,
      high_freq: -400.0,
      preemph_coeff: 0.97,
      remove_dc_offset: true,
      snip_edges: false,
      scale_to_int16: true,
      window_type: "povey".to_string(),
    }
  }
}

fn mel_scale(freq: f32) -> f32 {
  1127.0 * (1.0 + freq / 700.0).ln()
}

/// Kaldi style log mel filterbank extractor.
pub struct Fbank {
  opts: FbankOptions,
  frame_len: usize,
  frame_shift: usize,
  n_fft: usize,
  window: Vec<f32>,
  /// Per bin: (first fft bin, weights).
  bins: Vec<(usize, Vec<f32>)>,
  fft: Arc<dyn Fft<f32>>,
}

impl Fbank {
  pub fn new(opts: FbankOptions) -> Result<Self> {
    let sr = opts.sample_rate as f32;
    let frame_len = (sr * opts.frame_length_ms / 1000.0).round() as usize;
    let frame_shift = (sr * opts.frame_shift_ms / 1000.0).round() as usize;
    if frame_len == 0 || frame_shift == 0 {
      bail!("invalid fbank frame options");
    }
    let n_fft = frame_len.next_power_of_two();
    let window: Vec<f32> = (0..frame_len)
      .map(|i| {
        let a = 2.0 * std::f32::consts::PI * i as f32 / (frame_len as f32 - 1.0);
        match opts.window_type.as_str() {
          "hamming" => 0.54 - 0.46 * a.cos(),
          "hann" | "hanning" => 0.5 - 0.5 * a.cos(),
          // Povey window: hann ^ 0.85 (Kaldi default)
          _ => (0.5 - 0.5 * a.cos()).powf(0.85),
        }
      })
      .collect();

    let nyquist = sr / 2.0;
    let high = if opts.high_freq > 0.0 {
      opts.high_freq
    } else {
      nyquist + opts.high_freq
    };
    if !(opts.low_freq >= 0.0 && opts.low_freq < high && high <= nyquist) {
      bail!("invalid mel frequency range {}..{}", opts.low_freq, high);
    }
    let mel_low = mel_scale(opts.low_freq);
    let mel_high = mel_scale(high);
    let mel_delta = (mel_high - mel_low) / (opts.num_bins as f32 + 1.0);
    let n_bins = n_fft / 2 + 1;
    let fft_bin_width = sr / n_fft as f32;
    let mut bins = Vec::with_capacity(opts.num_bins);
    for b in 0..opts.num_bins {
      let left = mel_low + b as f32 * mel_delta;
      let center = left + mel_delta;
      let right = center + mel_delta;
      let mut first = None;
      let mut weights = Vec::new();
      for k in 0..n_bins {
        let mel = mel_scale(k as f32 * fft_bin_width);
        let w = if mel > left && mel < right {
          if mel <= center {
            (mel - left) / (center - left)
          } else {
            (right - mel) / (right - center)
          }
        } else {
          0.0
        };
        if w > 0.0 {
          if first.is_none() {
            first = Some(k);
          }
          weights.push(w);
        } else if first.is_some() {
          break;
        }
      }
      bins.push((first.unwrap_or(0), weights));
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    Ok(Fbank {
      opts,
      frame_len,
      frame_shift,
      n_fft,
      window,
      bins,
      fft,
    })
  }

  pub fn num_bins(&self) -> usize {
    self.opts.num_bins
  }

  pub fn sample_rate(&self) -> u32 {
    self.opts.sample_rate
  }

  /// Number of frames for `n` samples.
  fn num_frames(&self, n: usize) -> usize {
    if self.opts.snip_edges {
      if n < self.frame_len {
        0
      } else {
        1 + (n - self.frame_len) / self.frame_shift
      }
    } else {
      (n + self.frame_shift / 2) / self.frame_shift
    }
  }

  /// Compute log mel filterbank features (`frames x num_bins`) of samples
  /// already at [`Fbank::sample_rate`].
  pub fn compute(&self, samples: &[f32]) -> Vec<Vec<f32>> {
    let n = samples.len();
    let n_frames = self.num_frames(n);
    let scale = if self.opts.scale_to_int16 {
      32768.0
    } else {
      1.0
    };
    let mut out = Vec::with_capacity(n_frames);
    let mut frame = vec![0.0f32; self.frame_len];
    let mut buf = vec![Complex::new(0.0f32, 0.0f32); self.n_fft];
    for f in 0..n_frames {
      let start: isize = if self.opts.snip_edges {
        (f * self.frame_shift) as isize
      } else {
        (f * self.frame_shift) as isize - ((self.frame_len - self.frame_shift) / 2) as isize
      };
      for (j, v) in frame.iter_mut().enumerate() {
        let mut s = start + j as isize;
        if s < 0 {
          s = -s - 1;
        }
        if s >= n as isize {
          s = 2 * n as isize - 1 - s;
        }
        let s = s.clamp(0, n as isize - 1) as usize;
        *v = samples[s] * scale;
      }
      if self.opts.remove_dc_offset {
        let mean = frame.iter().sum::<f32>() / self.frame_len as f32;
        frame.iter_mut().for_each(|v| *v -= mean);
      }
      if self.opts.preemph_coeff != 0.0 {
        for j in (1..self.frame_len).rev() {
          frame[j] -= self.opts.preemph_coeff * frame[j - 1];
        }
        frame[0] -= self.opts.preemph_coeff * frame[0];
      }
      for (j, v) in buf.iter_mut().enumerate() {
        *v = if j < self.frame_len {
          Complex::new(frame[j] * self.window[j], 0.0)
        } else {
          Complex::new(0.0, 0.0)
        };
      }
      self.fft.process(&mut buf);
      let mut feats = vec![0.0f32; self.opts.num_bins];
      for (b, (first, weights)) in self.bins.iter().enumerate() {
        let mut e = 0.0f32;
        for (i, w) in weights.iter().enumerate() {
          e += buf[first + i].norm_sqr() * w;
        }
        feats[b] = e.max(f32::EPSILON).ln();
      }
      out.push(feats);
    }
    out
  }
}

/// Speaker embedding extractor: fbank front end + ONNX model. Accepts the
/// WeSpeaker / 3D-Speaker ONNX exports distributed with sherpa-onnx (input
/// `[1, frames, feat_dim]`, output `[1, dim]`), reading the front-end
/// settings from the model metadata.
pub struct SpeakerEncoder {
  session: Session,
  backend: Backend,
  fbank: Fbank,
  subtract_mean: bool,
  input_name: String,
  output_name: String,
  dim: usize,
}

impl SpeakerEncoder {
  /// Load a speaker embedding ONNX model (`Device::Auto`: GPU when compiled
  /// in and available, otherwise CPU).
  pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
    Self::on_device(path, Device::Auto)
  }

  /// Load a speaker embedding ONNX model on the given device.
  pub fn on_device<P: AsRef<Path>>(path: P, device: Device) -> Result<Self> {
    let path = path.as_ref();
    let backend = resolve_backend(device)?;
    let session = load_session(path, &backend)?;
    let meta: std::collections::HashMap<String, String> = {
      let m = session.metadata()?;
      let keys = m.custom_keys().unwrap_or_default();
      keys
        .into_iter()
        .filter_map(|k| m.custom(&k).map(|v| (k, v)))
        .collect()
    };
    let get = |k: &str| -> Option<String> { meta.get(k).cloned() };
    let sample_rate: u32 = get("sample_rate")
      .and_then(|v| v.trim().parse().ok())
      .unwrap_or(16000);
    let normalize_samples = get("normalize_samples")
      .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
      .unwrap_or(true);
    let subtract_mean = get("feature_normalize_type")
      .map(|v| v.trim() == "global-mean")
      .unwrap_or(false);
    let dim: usize = get("output_dim")
      .and_then(|v| v.trim().parse().ok())
      .unwrap_or(0);
    if let Some(fw) = get("framework") {
      if fw != "wespeaker" && fw != "3d-speaker" {
        bail!(
          "unsupported speaker model framework '{}' (expected wespeaker or 3d-speaker)",
          fw
        );
      }
    }
    let (input_name, feat_dim) = {
      let input = session
        .inputs()
        .first()
        .ok_or_else(|| anyhow!("speaker model has no input"))?;
      let dim = input
        .dtype()
        .tensor_shape()
        .and_then(|s| s.last().copied())
        .filter(|&d| d > 0)
        .map(|d| d as usize)
        .unwrap_or(80);
      (input.name().to_string(), dim)
    };
    let output_name = session
      .outputs()
      .first()
      .map(|o| o.name().to_string())
      .ok_or_else(|| anyhow!("speaker model has no output"))?;
    let fbank = Fbank::new(FbankOptions {
      sample_rate,
      num_bins: feat_dim,
      scale_to_int16: !normalize_samples,
      ..FbankOptions::default()
    })?;
    Ok(SpeakerEncoder {
      session,
      backend,
      fbank,
      subtract_mean,
      input_name,
      output_name,
      dim,
    })
  }

  /// Force (or disable) per-utterance mean subtraction of the features,
  /// overriding what the model metadata says.
  pub fn set_subtract_mean(&mut self, on: bool) {
    self.subtract_mean = on;
  }

  /// Whether the features are mean-normalised per utterance.
  pub fn subtract_mean(&self) -> bool {
    self.subtract_mean
  }

  /// Replace the front end options (window type, mel range, ...).
  pub fn set_fbank_options(&mut self, opts: FbankOptions) -> Result<()> {
    self.fbank = Fbank::new(opts)?;
    Ok(())
  }

  /// Current front end options.
  pub fn fbank_options(&self) -> &FbankOptions {
    &self.fbank.opts
  }

  /// Execution provider the model runs on.
  pub fn backend(&self) -> &Backend {
    &self.backend
  }

  /// Embedding dimension (0 when the model does not declare it).
  pub fn dim(&self) -> usize {
    self.dim
  }

  /// Sample rate the model expects.
  pub fn sample_rate(&self) -> u32 {
    self.fbank.sample_rate()
  }

  /// L2-normalised embedding of mono `samples` at `sample_rate` (resampled
  /// when needed).
  pub fn embed(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
    let target = self.fbank.sample_rate();
    let audio: Vec<f32> = if sample_rate != target {
      resample(samples, sample_rate, target)
    } else {
      samples.to_vec()
    };
    let mut feats = self.fbank.compute(&audio);
    if feats.len() < 10 {
      bail!(
        "audio too short for a speaker embedding ({} frames)",
        feats.len()
      );
    }
    if self.subtract_mean {
      let n = feats.len() as f32;
      let d = feats[0].len();
      let mut mean = vec![0.0f32; d];
      for f in &feats {
        for (m, v) in mean.iter_mut().zip(f) {
          *m += v;
        }
      }
      mean.iter_mut().for_each(|m| *m /= n);
      for f in feats.iter_mut() {
        for (v, m) in f.iter_mut().zip(&mean) {
          *v -= m;
        }
      }
    }
    let frames = feats.len();
    let d = feats[0].len();
    let flat: Vec<f32> = feats.into_iter().flatten().collect();
    let x = Array3::from_shape_vec((1, frames, d), flat)?;
    let outputs = self.session.run(ort::inputs! {
      self.input_name.as_str() => &Value::from_array(x)?
    })?;
    let (_, data) = outputs[self.output_name.as_str()].try_extract_tensor::<f32>()?;
    let mut emb: Vec<f32> = data.to_vec();
    let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    emb.iter_mut().for_each(|v| *v /= norm);
    Ok(emb)
  }
}

/// Cosine similarity of two vectors (0 when either is empty).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
  if a.is_empty() || a.len() != b.len() {
    return 0.0;
  }
  let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
  let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
  let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
  dot / (na * nb).max(1e-8)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn fbank_shapes_follow_kaldi() {
    let fb = Fbank::new(FbankOptions::default()).unwrap();
    // 1 s at 16 kHz, snip_edges=false: (16000 + 80) / 160 = 100 frames
    let x: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
    let f = fb.compute(&x);
    assert_eq!(f.len(), 100);
    assert_eq!(f[0].len(), 80);
    assert!(f.iter().all(|fr| fr.iter().all(|v| v.is_finite())));
    let fb2 = Fbank::new(FbankOptions {
      snip_edges: true,
      ..FbankOptions::default()
    })
    .unwrap();
    assert_eq!(fb2.compute(&x).len(), 98);
  }

  #[test]
  fn cosine_is_bounded() {
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    assert_eq!(cosine(&[], &[]), 0.0);
  }
}
