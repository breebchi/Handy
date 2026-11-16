// ============================================================================
// Audio analysis used by voice cloning: WAV loading, resampling, log-mel
// spectrograms, pitch statistics and soft-DTW.
// ============================================================================

use anyhow::{anyhow, bail, Context, Result};
use hound::{SampleFormat, WavReader};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::path::Path;
use std::sync::Arc;

/// Mono audio buffer with its sample rate.
#[derive(Debug, Clone)]
pub struct Audio {
  pub samples: Vec<f32>,
  pub sample_rate: u32,
}

impl Audio {
  /// Duration in seconds.
  pub fn duration(&self) -> f32 {
    self.samples.len() as f32 / self.sample_rate as f32
  }

  /// Root mean square of the samples.
  pub fn rms(&self) -> f32 {
    rms(&self.samples)
  }

  /// Return a copy resampled to `sample_rate` (no-op when already there).
  pub fn resampled(&self, sample_rate: u32) -> Audio {
    if self.sample_rate == sample_rate {
      return self.clone();
    }
    Audio {
      samples: resample(&self.samples, self.sample_rate, sample_rate),
      sample_rate,
    }
  }
}

/// Root mean square of a buffer (0 for an empty buffer).
pub fn rms(x: &[f32]) -> f32 {
  if x.is_empty() {
    return 0.0;
  }
  (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// Load a WAV file (any channel count, 8/16/24/32-bit PCM or 32-bit float)
/// as mono f32 samples in [-1, 1], with DC offset removed.
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Audio> {
  let path = path.as_ref();
  let mut reader =
    WavReader::open(path).with_context(|| format!("open reference wav {}", path.display()))?;
  let spec = reader.spec();
  let channels = spec.channels.max(1) as usize;

  let interleaved: Vec<f32> = match spec.sample_format {
    SampleFormat::Float => reader
      .samples::<f32>()
      .collect::<std::result::Result<Vec<_>, _>>()?,
    SampleFormat::Int => {
      let scale = 1.0 / ((1i64 << (spec.bits_per_sample - 1)) as f32);
      reader
        .samples::<i32>()
        .map(|s| s.map(|v| v as f32 * scale))
        .collect::<std::result::Result<Vec<_>, _>>()?
    }
  };

  let samples = mix_to_mono(&interleaved, channels);
  if samples.is_empty() {
    bail!("reference wav {} contains no samples", path.display());
  }
  Ok(Audio {
    samples: remove_dc(&samples),
    sample_rate: spec.sample_rate,
  })
}

/// Average interleaved channels down to mono.
pub fn mix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
  if channels <= 1 {
    return interleaved.to_vec();
  }
  interleaved
    .chunks(channels)
    .map(|frame| frame.iter().sum::<f32>() / channels as f32)
    .collect()
}

fn remove_dc(x: &[f32]) -> Vec<f32> {
  let mean = x.iter().sum::<f32>() / x.len().max(1) as f32;
  x.iter().map(|v| v - mean).collect()
}

/// Windowed-sinc resampler (Kaiser-free Hann window, 32 taps per side). Good
/// enough for feature extraction of an offline reference clip.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
  if from == to || x.is_empty() {
    return x.to_vec();
  }
  let ratio = to as f64 / from as f64;
  let out_len = (x.len() as f64 * ratio).round() as usize;
  // Cut-off relative to the input rate: when downsampling limit to the new
  // Nyquist frequency.
  let cutoff = ratio.min(1.0) * 0.95;
  let taps: isize = 32;
  let mut out = Vec::with_capacity(out_len);
  for n in 0..out_len {
    let center = n as f64 / ratio;
    let i0 = center.floor() as isize;
    let mut acc = 0.0f64;
    let mut wsum = 0.0f64;
    for i in (i0 - taps + 1)..=(i0 + taps) {
      if i < 0 || i >= x.len() as isize {
        continue;
      }
      let d = center - i as f64;
      let sinc = if d.abs() < 1e-9 {
        1.0
      } else {
        let a = std::f64::consts::PI * d * cutoff;
        a.sin() / a
      };
      let w = 0.5 + 0.5 * (std::f64::consts::PI * d / taps as f64).cos();
      let k = sinc * cutoff * w;
      acc += k * x[i as usize] as f64;
      wsum += k;
    }
    // The kernel has unit DC gain; only the truncated edge windows need a
    // renormalisation.
    let v = if wsum > 0.5 && wsum < 0.98 {
      acc / wsum
    } else {
      acc
    };
    out.push(v as f32);
  }
  out
}

/// Find the first and last sample whose short-time energy exceeds `threshold`
/// times the peak short-time energy (`(start, end)` sample indices). Returns
/// `None` when the clip is silent.
pub fn speech_bounds(x: &[f32], sample_rate: u32, threshold: f32) -> Option<(usize, usize)> {
  let win = (sample_rate as usize / 100).max(1); // 10 ms
  let energies: Vec<f32> = x.chunks(win).map(rms).collect();
  let peak = energies.iter().cloned().fold(0.0f32, f32::max);
  if peak <= 0.0 {
    return None;
  }
  let thr = peak * threshold;
  let first = energies.iter().position(|&e| e > thr)?;
  let last = energies.iter().rposition(|&e| e > thr)?;
  Some((first * win, ((last + 1) * win).min(x.len())))
}

/// Find the first utterance of a recording: from the first speech frame to
/// the first pause of at least `min_pause` seconds that follows at least
/// `min_length` seconds of speech. Returns `(start, end)` in seconds, or
/// `None` when the clip is silent. Falls back to the whole speech region
/// when no such pause exists.
pub fn first_utterance(
  x: &[f32],
  sample_rate: u32,
  min_pause: f32,
  min_length: f32,
) -> Option<(f32, f32)> {
  let win = (sample_rate as usize / 100).max(1); // 10 ms
  let energies: Vec<f32> = x.chunks(win).map(rms).collect();
  let peak = energies.iter().cloned().fold(0.0f32, f32::max);
  if peak <= 0.0 {
    return None;
  }
  let thr = peak * 0.02;
  let speech: Vec<bool> = energies.iter().map(|&e| e > thr).collect();
  let first = speech.iter().position(|&s| s)?;
  let last = speech.iter().rposition(|&s| s)?;
  let frame_s = win as f32 / sample_rate as f32;
  let pause_frames = (min_pause / frame_s).ceil() as usize;
  let min_frames = (min_length / frame_s).ceil() as usize;
  let mut i = first;
  let mut last_speech = first;
  while i <= last {
    if speech[i] {
      last_speech = i;
      i += 1;
      continue;
    }
    // Length of this silent run.
    let mut j = i;
    while j <= last && !speech[j] {
      j += 1;
    }
    if j - i >= pause_frames && last_speech + 1 - first >= min_frames {
      return Some((first as f32 * frame_s, (last_speech + 1) as f32 * frame_s));
    }
    i = j;
  }
  Some((first as f32 * frame_s, (last + 1) as f32 * frame_s))
}

// ============================================================================
// Spectral gating denoiser
// ============================================================================

/// Reduce stationary background noise (microphone hiss, room tone) with
/// spectral gating: the noise spectrum is estimated from the quietest frames
/// and every time-frequency bin close to that floor is attenuated.
/// `strength` in `0..=1` scales the attenuation (0 = untouched, 1 = full
/// gating). Used to clean a voice cloning reference before analysis.
pub fn denoise(x: &[f32], strength: f32) -> Vec<f32> {
  let strength = strength.clamp(0.0, 1.0);
  let n_fft = 1024usize;
  let hop = n_fft / 4;
  if strength <= 0.0 || x.len() < 2 * n_fft {
    return x.to_vec();
  }
  let mut planner = FftPlanner::new();
  let fft = planner.plan_fft_forward(n_fft);
  let ifft = planner.plan_fft_inverse(n_fft);
  let window: Vec<f32> = (0..n_fft)
    .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos())
    .collect();
  let pad = n_fft / 2;
  let padded = reflect_pad(x, pad);
  let n_frames = 1 + (padded.len() - n_fft) / hop;
  let n_bins = n_fft / 2 + 1;

  // STFT
  let mut spectra: Vec<Vec<Complex<f32>>> = Vec::with_capacity(n_frames);
  let mut buf = vec![Complex::new(0.0f32, 0.0f32); n_fft];
  for f in 0..n_frames {
    let start = f * hop;
    for i in 0..n_fft {
      buf[i] = Complex::new(padded[start + i] * window[i], 0.0);
    }
    fft.process(&mut buf);
    spectra.push(buf.clone());
  }

  // Noise profile: mean magnitude per bin over the quietest 10 % of frames.
  let mags: Vec<Vec<f32>> = spectra
    .iter()
    .map(|s| s[..n_bins].iter().map(|c| c.norm()).collect())
    .collect();
  let mut energies: Vec<(f32, usize)> = mags
    .iter()
    .enumerate()
    .map(|(i, m)| (m.iter().map(|v| v * v).sum::<f32>(), i))
    .collect();
  energies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
  let n_quiet = (n_frames / 10).max(5).min(n_frames);
  let mut noise = vec![0.0f32; n_bins];
  for &(_, i) in energies.iter().take(n_quiet) {
    for (k, v) in noise.iter_mut().enumerate() {
      *v += mags[i][k];
    }
  }
  noise.iter_mut().for_each(|v| *v /= n_quiet as f32);

  // Gain mask: attenuate bins near the noise floor, smoothed over time and
  // frequency to avoid musical noise.
  let threshold = 1.5f32;
  let mut mask: Vec<Vec<f32>> = mags
    .iter()
    .map(|m| {
      m.iter()
        .zip(&noise)
        .map(|(&mag, &nz)| {
          if mag <= 1e-9 {
            0.0
          } else {
            ((mag - threshold * nz) / mag).clamp(0.0, 1.0)
          }
        })
        .collect()
    })
    .collect();
  let smoothed: Vec<Vec<f32>> = (0..n_frames)
    .map(|f| {
      (0..n_bins)
        .map(|k| {
          let mut acc = 0.0f32;
          let mut cnt = 0.0f32;
          for df in -1i32..=1 {
            let ff = f as i32 + df;
            if ff < 0 || ff >= n_frames as i32 {
              continue;
            }
            for dk in -1i32..=1 {
              let kk = k as i32 + dk;
              if kk < 0 || kk >= n_bins as i32 {
                continue;
              }
              acc += mask[ff as usize][kk as usize];
              cnt += 1.0;
            }
          }
          acc / cnt.max(1.0)
        })
        .collect()
    })
    .collect();
  mask = smoothed;

  // Apply and inverse STFT (overlap-add with window-square normalisation).
  let mut out = vec![0.0f32; padded.len()];
  let mut norm = vec![0.0f32; padded.len()];
  for f in 0..n_frames {
    for k in 0..n_fft {
      let bin = if k < n_bins { k } else { n_fft - k };
      let g = 1.0 - strength * (1.0 - mask[f][bin]);
      buf[k] = spectra[f][k] * g;
    }
    ifft.process(&mut buf);
    let start = f * hop;
    for i in 0..n_fft {
      out[start + i] += buf[i].re / n_fft as f32 * window[i];
      norm[start + i] += window[i] * window[i];
    }
  }
  (0..x.len())
    .map(|i| {
      let j = i + pad;
      if norm[j] > 1e-6 {
        out[j] / norm[j]
      } else {
        out[j]
      }
    })
    .collect()
}

// ============================================================================
// Log-mel spectrogram (torchaudio compatible: HTK mel scale, power 2)
// ============================================================================

fn hz_to_mel(hz: f32) -> f32 {
  2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
  700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Log-mel analyser with a fixed FFT plan.
pub struct MelAnalyzer {
  sample_rate: u32,
  n_fft: usize,
  hop: usize,
  n_mels: usize,
  window: Vec<f32>,
  filters: Vec<Vec<(usize, f32)>>, // sparse (bin, weight) per mel band
  fft: Arc<dyn Fft<f32>>,
}

impl MelAnalyzer {
  /// `n_fft` = window length, `hop` = hop size, `n_mels` mel bands from 0 Hz
  /// to Nyquist.
  pub fn new(sample_rate: u32, n_fft: usize, hop: usize, n_mels: usize) -> Self {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let window: Vec<f32> = (0..n_fft)
      .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos())
      .collect();
    let n_bins = n_fft / 2 + 1;
    let f_max = sample_rate as f32 / 2.0;
    let m_min = hz_to_mel(0.0);
    let m_max = hz_to_mel(f_max);
    let m_pts: Vec<f32> = (0..n_mels + 2)
      .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32))
      .collect();
    let bin_freqs: Vec<f32> = (0..n_bins)
      .map(|b| b as f32 * sample_rate as f32 / n_fft as f32)
      .collect();
    let mut filters = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
      let (lo, mid, hi) = (m_pts[m], m_pts[m + 1], m_pts[m + 2]);
      let mut bank = Vec::new();
      for (b, &f) in bin_freqs.iter().enumerate() {
        let w = if f >= lo && f <= mid {
          (f - lo) / (mid - lo).max(1e-6)
        } else if f > mid && f <= hi {
          (hi - f) / (hi - mid).max(1e-6)
        } else {
          0.0
        };
        if w > 0.0 {
          bank.push((b, w));
        }
      }
      filters.push(bank);
    }
    MelAnalyzer {
      sample_rate,
      n_fft,
      hop,
      n_mels,
      window,
      filters,
      fft,
    }
  }

  pub fn sample_rate(&self) -> u32 {
    self.sample_rate
  }

  pub fn n_mels(&self) -> usize {
    self.n_mels
  }

  /// Log-mel frames (`frames x n_mels`), centred frames with reflect padding
  /// like torchaudio. `log(mel + 1e-5)`.
  pub fn log_mel(&self, x: &[f32]) -> Vec<Vec<f32>> {
    if x.is_empty() {
      return Vec::new();
    }
    let pad = self.n_fft / 2;
    let padded = reflect_pad(x, pad);
    let n_frames = 1 + (padded.len().saturating_sub(self.n_fft)) / self.hop;
    let mut buf = vec![Complex::new(0.0f32, 0.0f32); self.n_fft];
    let mut frames = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
      let start = f * self.hop;
      for i in 0..self.n_fft {
        buf[i] = Complex::new(padded[start + i] * self.window[i], 0.0);
      }
      self.fft.process(&mut buf);
      let mut mel = vec![0.0f32; self.n_mels];
      for (m, bank) in self.filters.iter().enumerate() {
        let mut acc = 0.0f32;
        for &(b, w) in bank {
          acc += buf[b].norm_sqr() * w;
        }
        mel[m] = (acc + 1e-5).ln();
      }
      frames.push(mel);
    }
    frames
  }

  /// Log-mel frames averaged over groups of `pool` consecutive frames.
  pub fn log_mel_pooled(&self, x: &[f32], pool: usize) -> Vec<Vec<f32>> {
    let frames = self.log_mel(x);
    let pool = pool.max(1);
    frames
      .chunks_exact(pool)
      .map(|group| {
        let mut acc = vec![0.0f32; self.n_mels];
        for fr in group {
          for (a, v) in acc.iter_mut().zip(fr) {
            *a += v;
          }
        }
        for a in acc.iter_mut() {
          *a /= pool as f32;
        }
        acc
      })
      .collect()
  }
}

fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
  let n = x.len();
  let mut out = Vec::with_capacity(n + 2 * pad);
  for i in 0..pad {
    let idx = (pad - i) % n.max(1);
    out.push(x[idx.min(n - 1)]);
  }
  out.extend_from_slice(x);
  for i in 0..pad {
    let idx = n.saturating_sub(2 + i % n.max(1));
    out.push(x[idx.min(n - 1)]);
  }
  out
}

/// Per-band mean and standard deviation over time of log-mel frames whose
/// energy (mean log-mel) is above the median minus `margin`. Text independent
/// spectral envelope statistics, used as a crude timbre descriptor.
pub fn mel_statistics(frames: &[Vec<f32>]) -> Option<(Vec<f32>, Vec<f32>)> {
  if frames.is_empty() {
    return None;
  }
  let n_mels = frames[0].len();
  let mut energies: Vec<f32> = frames
    .iter()
    .map(|f| f.iter().sum::<f32>() / n_mels as f32)
    .collect();
  let mut sorted = energies.clone();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let peak = sorted[sorted.len() - 1];
  let thr = peak - 4.0; // roughly 35 dB below the loudest frame
  energies
    .iter_mut()
    .for_each(|e| *e = if *e >= thr { 1.0 } else { 0.0 });
  let count = energies.iter().sum::<f32>();
  if count < 1.0 {
    return None;
  }
  let mut mean = vec![0.0f32; n_mels];
  for (f, &keep) in frames.iter().zip(&energies) {
    if keep > 0.0 {
      for (m, v) in mean.iter_mut().zip(f) {
        *m += v;
      }
    }
  }
  mean.iter_mut().for_each(|m| *m /= count);
  let mut var = vec![0.0f32; n_mels];
  for (f, &keep) in frames.iter().zip(&energies) {
    if keep > 0.0 {
      for ((s, v), m) in var.iter_mut().zip(f).zip(&mean) {
        *s += (v - m) * (v - m);
      }
    }
  }
  let std: Vec<f32> = var.iter().map(|v| (v / count).sqrt()).collect();
  Some((mean, std))
}

// ============================================================================
// Pitch statistics
// ============================================================================

/// Pitch summary of a clip: mean and standard deviation of log2(F0) over
/// voiced frames, plus the voiced ratio.
#[derive(Debug, Clone, Copy)]
pub struct PitchStats {
  pub mean_log2_f0: f32,
  pub std_log2_f0: f32,
  pub voiced_ratio: f32,
}

/// Estimate F0 statistics with a normalised autocorrelation pitch tracker on
/// a decimated (~11 kHz) copy of the signal. Search range 60 - 500 Hz.
/// Returns `None` when no voiced frame is found.
pub fn pitch_stats(x: &[f32], sample_rate: u32) -> Option<PitchStats> {
  let target: u32 = 11025;
  let y = if sample_rate > target * 5 / 4 {
    resample(x, sample_rate, target)
  } else {
    x.to_vec()
  };
  let sr = if sample_rate > target * 5 / 4 {
    target
  } else {
    sample_rate
  } as f32;
  let frame = (sr * 0.04) as usize; // 40 ms
  let hop = frame / 2;
  let lag_min = (sr / 500.0) as usize;
  let lag_max = (sr / 60.0) as usize;
  if y.len() < frame + lag_max {
    return None;
  }
  let peak_rms = y
    .chunks(hop.max(1))
    .map(rms)
    .fold(0.0f32, f32::max)
    .max(1e-6);
  let mut log_f0s = Vec::new();
  let mut n_frames = 0usize;
  let mut start = 0;
  while start + frame + lag_max <= y.len() {
    let seg = &y[start..start + frame];
    n_frames += 1;
    if rms(seg) > 0.1 * peak_rms {
      let e0: f32 = seg.iter().map(|v| v * v).sum::<f32>().max(1e-9);
      let mut best_lag = 0usize;
      let mut best_corr = 0.0f32;
      for lag in lag_min..=lag_max {
        let seg2 = &y[start + lag..start + lag + frame];
        let e1: f32 = seg2.iter().map(|v| v * v).sum::<f32>().max(1e-9);
        let dot: f32 = seg.iter().zip(seg2).map(|(a, b)| a * b).sum();
        let corr = dot / (e0 * e1).sqrt();
        if corr > best_corr {
          best_corr = corr;
          best_lag = lag;
        }
      }
      if best_corr > 0.7 && best_lag > 0 {
        // Prefer the lowest lag (highest F0) that is nearly as good as the
        // best, to avoid octave-down errors.
        let mut lag = best_lag;
        for cand in lag_min..best_lag {
          if cand * 2 == best_lag || cand * 3 == best_lag {
            let seg2 = &y[start + cand..start + cand + frame];
            let e1: f32 = seg2.iter().map(|v| v * v).sum::<f32>().max(1e-9);
            let dot: f32 = seg.iter().zip(seg2).map(|(a, b)| a * b).sum();
            let corr = dot / (e0 * e1).sqrt();
            if corr > 0.85 * best_corr {
              lag = cand;
              break;
            }
          }
        }
        log_f0s.push((sr / lag as f32).log2());
      }
    }
    start += hop;
  }
  if log_f0s.is_empty() || n_frames == 0 {
    return None;
  }
  let mean = log_f0s.iter().sum::<f32>() / log_f0s.len() as f32;
  let var = log_f0s.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / log_f0s.len() as f32;
  Some(PitchStats {
    mean_log2_f0: mean,
    std_log2_f0: var.sqrt(),
    voiced_ratio: log_f0s.len() as f32 / n_frames as f32,
  })
}

// ============================================================================
// Soft-DTW
// ============================================================================

/// Soft-DTW (Cuturi & Blondel) over an `n x m` cost matrix given as rows,
/// with smoothing `gamma`. Returns the soft minimum alignment cost.
pub fn soft_dtw(cost: &[Vec<f32>], gamma: f32) -> f32 {
  let n = cost.len();
  if n == 0 {
    return 0.0;
  }
  let m = cost[0].len();
  if m == 0 {
    return 0.0;
  }
  let inf = f32::INFINITY;
  let mut prev = vec![inf; m + 1];
  let mut cur = vec![inf; m + 1];
  prev[0] = 0.0;
  let soft_min = |a: f32, b: f32, c: f32| -> f32 {
    let mx = (-a / gamma).max(-b / gamma).max(-c / gamma);
    if mx == -inf {
      return inf;
    }
    let s = (-a / gamma - mx).exp() + (-b / gamma - mx).exp() + (-c / gamma - mx).exp();
    -gamma * (mx + s.ln())
  };
  for i in 1..=n {
    cur[0] = inf;
    for j in 1..=m {
      let r = soft_min(prev[j - 1], prev[j], cur[j - 1]);
      cur[j] = cost[i - 1][j - 1] + r;
    }
    std::mem::swap(&mut prev, &mut cur);
  }
  prev[m]
}

/// Mean absolute difference between two vectors of the same length.
pub fn l1_mean(a: &[f32], b: &[f32]) -> f32 {
  if a.is_empty() {
    return 0.0;
  }
  a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

/// Log-mel soft-DTW distance between two frame sequences, normalised by the
/// number of frames (`(n + m)`), L1 frame cost divided by the band count.
pub fn mel_dtw_distance(a: &[Vec<f32>], b: &[Vec<f32>], gamma: f32) -> Result<f32> {
  if a.is_empty() || b.is_empty() {
    return Err(anyhow!("cannot align empty log-mel sequences"));
  }
  let cost: Vec<Vec<f32>> = a
    .iter()
    .map(|fa| b.iter().map(|fb| l1_mean(fa, fb)).collect())
    .collect();
  Ok(soft_dtw(&cost, gamma) / (a.len() + b.len()) as f32)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn resample_keeps_length_ratio_and_level() {
    let sr = 16000u32;
    let x: Vec<f32> = (0..sr)
      .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
      .collect();
    let y = resample(&x, sr, 44100);
    assert!((y.len() as i64 - 44100).abs() <= 1);
    let r = rms(&y[2000..40000]) / rms(&x[1000..15000]);
    assert!((r - 1.0).abs() < 0.05, "level ratio {}", r);
    let z = resample(&y, 44100, sr);
    assert!((z.len() as i64 - sr as i64).abs() <= 2);
    let r2 = rms(&z[1000..15000]) / rms(&x[1000..15000]);
    assert!((r2 - 1.0).abs() < 0.05, "down level ratio {}", r2);
  }

  #[test]
  fn pitch_of_a_sine_is_found() {
    let sr = 44100u32;
    let x: Vec<f32> = (0..sr)
      .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin())
      .collect();
    let p = pitch_stats(&x, sr).unwrap();
    let f0 = 2f32.powf(p.mean_log2_f0);
    assert!((f0 - 220.0).abs() < 8.0, "f0 {}", f0);
    assert!(p.std_log2_f0 < 0.05);
  }

  #[test]
  fn soft_dtw_of_identical_sequences_is_small() {
    let mel = MelAnalyzer::new(44100, 2048, 512, 80);
    let sr = 44100u32;
    let x: Vec<f32> = (0..sr)
      .map(|i| (2.0 * std::f32::consts::PI * 330.0 * i as f32 / sr as f32).sin())
      .collect();
    let a = mel.log_mel_pooled(&x, 4);
    let b = mel.log_mel_pooled(&x, 4);
    assert_eq!(a[0].len(), 80);
    let d = mel_dtw_distance(&a, &b, 0.1).unwrap();
    assert!(d.abs() < 0.05, "self distance {}", d);
    let noise: Vec<f32> = (0..sr)
      .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
      .collect();
    let c = mel.log_mel_pooled(&noise, 4);
    assert!(mel_dtw_distance(&a, &c, 0.1).unwrap() > d + 0.5);
  }

  #[test]
  fn denoise_removes_hiss_and_keeps_tone() {
    let sr = 16000u32;
    let n = sr as usize * 2;
    let mut rng = 12345u32;
    let noise: Vec<f32> = (0..n)
      .map(|_| {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.02
      })
      .collect();
    let tone: Vec<f32> = (0..n)
      .map(|i| {
        if i > n / 4 && i < 3 * n / 4 {
          (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin() * 0.3
        } else {
          0.0
        }
      })
      .collect();
    let mixed: Vec<f32> = tone.iter().zip(&noise).map(|(a, b)| a + b).collect();
    let clean = denoise(&mixed, 1.0);
    assert_eq!(clean.len(), mixed.len());
    let quiet_before = rms(&mixed[..n / 8]);
    let quiet_after = rms(&clean[..n / 8]);
    assert!(
      quiet_after < quiet_before * 0.5,
      "{} vs {}",
      quiet_after,
      quiet_before
    );
    let tone_before = rms(&mixed[n / 3..2 * n / 3]);
    let tone_after = rms(&clean[n / 3..2 * n / 3]);
    assert!(
      (tone_after / tone_before - 1.0).abs() < 0.1,
      "{} vs {}",
      tone_after,
      tone_before
    );
    assert_eq!(denoise(&mixed, 0.0), mixed);
  }

  #[test]
  fn first_utterance_stops_at_the_first_pause() {
    let sr = 16000u32;
    let n = sr as usize * 6;
    let x: Vec<f32> = (0..n)
      .map(|i| {
        let t = i as f32 / sr as f32;
        // speech 0.5..2.5 s, pause, speech 3.5..5.5 s
        if (0.5..2.5).contains(&t) || (3.5..5.5).contains(&t) {
          (i as f32 * 0.07).sin() * 0.5
        } else {
          0.0
        }
      })
      .collect();
    let (a, b) = first_utterance(&x, sr, 0.6, 1.0).unwrap();
    assert!((a - 0.5).abs() < 0.03, "start {}", a);
    assert!((b - 2.5).abs() < 0.03, "end {}", b);
    // A shorter pause than min_pause: the whole speech region is returned.
    let (_, b2) = first_utterance(&x, sr, 1.5, 1.0).unwrap();
    assert!((b2 - 5.5).abs() < 0.03, "end {}", b2);
  }

  #[test]
  fn speech_bounds_trims_silence() {
    let sr = 44100u32;
    let mut x = vec![0.0f32; sr as usize];
    for (i, v) in x.iter_mut().enumerate().take(30000).skip(10000) {
      *v = (i as f32 * 0.05).sin();
    }
    let (s, e) = speech_bounds(&x, sr, 0.05).unwrap();
    assert!(s <= 10000 && s > 8000, "start {}", s);
    assert!((30000..32000).contains(&e), "end {}", e);
  }
}
