// ============================================================================
// Compute device selection: CPU or a GPU execution provider of ONNX Runtime.
//
// GPU support is optional and compiled in with cargo features (`cuda`,
// `tensorrt`, `nvrtx`, `webgpu`, `directml`, `coreml`, `rocm`, `migraphx`,
// `openvino`, `qnn`, `cann`, `nnapi`). Without any of them every device
// choice resolves to the CPU.
// ============================================================================

use anyhow::{anyhow, Result};
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use std::path::Path;

/// Where the ONNX models run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
  /// Use a GPU when the crate was built with a GPU feature and the provider
  /// can be initialised, otherwise the CPU. Never fails because of the GPU.
  #[default]
  Auto,
  /// CPU only.
  Cpu,
  /// Require a GPU (first device): fails when no GPU provider is compiled
  /// in or it cannot be initialised.
  Gpu,
  /// Require a specific GPU by index (providers that address devices by
  /// number: CUDA, TensorRT, ROCm, MIGraphX, DirectML, WebGPU, ...).
  GpuIndex(i32),
}

impl Device {
  /// Parse `auto`, `cpu`, `gpu` or `gpu:<index>`.
  pub fn parse(s: &str) -> Result<Device> {
    let s = s.trim().to_ascii_lowercase();
    match s.as_str() {
      "auto" => Ok(Device::Auto),
      "cpu" => Ok(Device::Cpu),
      "gpu" | "cuda" => Ok(Device::Gpu),
      other => {
        if let Some(idx) = other
          .strip_prefix("gpu:")
          .or_else(|| other.strip_prefix("cuda:"))
        {
          let n: i32 = idx
            .parse()
            .map_err(|_| anyhow!("invalid device index in '{}'", s))?;
          Ok(Device::GpuIndex(n))
        } else {
          Err(anyhow!(
            "unknown device '{}': use auto, cpu, gpu or gpu:<index>",
            s
          ))
        }
      }
    }
  }

  fn wants_gpu(self) -> bool {
    !matches!(self, Device::Cpu)
  }

  fn requires_gpu(self) -> bool {
    matches!(self, Device::Gpu | Device::GpuIndex(_))
  }

  fn index(self) -> i32 {
    match self {
      Device::GpuIndex(i) => i,
      _ => 0,
    }
  }
}

impl std::fmt::Display for Device {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Device::Auto => f.write_str("auto"),
      Device::Cpu => f.write_str("cpu"),
      Device::Gpu => f.write_str("gpu"),
      Device::GpuIndex(i) => write!(f, "gpu:{}", i),
    }
  }
}

/// Name of the CPU execution provider as reported by [`Backend`].
pub const CPU_PROVIDER: &str = "CPU";

/// The execution provider actually used by a set of sessions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Backend {
  /// `"CPU"` or the ONNX Runtime provider name, e.g.
  /// `"CUDAExecutionProvider"`.
  pub provider: String,
  /// Device index for GPU providers (0 on the CPU).
  pub device_index: i32,
}

impl Backend {
  pub fn cpu() -> Backend {
    Backend {
      provider: CPU_PROVIDER.to_string(),
      device_index: 0,
    }
  }

  pub fn is_gpu(&self) -> bool {
    self.provider != CPU_PROVIDER
  }
}

impl std::fmt::Display for Backend {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.is_gpu() {
      write!(f, "{} (device {})", self.provider, self.device_index)
    } else {
      f.write_str(CPU_PROVIDER)
    }
  }
}

/// Names of the GPU / accelerator execution providers compiled into this
/// build, in the order [`Device::Auto`] tries them: vendor specific
/// providers first, the generic WebGPU one last. Empty without any GPU
/// cargo feature.
pub fn compiled_gpu_providers() -> Vec<&'static str> {
  #[allow(unused_mut)]
  let mut v: Vec<&'static str> = Vec::new();
  #[cfg(feature = "tensorrt")]
  v.push("TensorrtExecutionProvider");
  #[cfg(feature = "nvrtx")]
  v.push("NvTensorRTRTXExecutionProvider");
  #[cfg(feature = "cuda")]
  v.push("CUDAExecutionProvider");
  #[cfg(feature = "migraphx")]
  v.push("MIGraphXExecutionProvider");
  #[cfg(feature = "rocm")]
  v.push("ROCMExecutionProvider");
  #[cfg(feature = "openvino")]
  v.push("OpenVINOExecutionProvider");
  #[cfg(feature = "directml")]
  v.push("DmlExecutionProvider");
  #[cfg(feature = "coreml")]
  v.push("CoreMLExecutionProvider");
  #[cfg(feature = "qnn")]
  v.push("QNNExecutionProvider");
  #[cfg(feature = "cann")]
  v.push("CANNExecutionProvider");
  #[cfg(feature = "nnapi")]
  v.push("NnapiExecutionProvider");
  #[cfg(feature = "webgpu")]
  v.push("WebGpuExecutionProvider");
  v
}

/// Whether this build can use a GPU at all.
pub fn gpu_support_compiled() -> bool {
  !compiled_gpu_providers().is_empty()
}

/// Register `ep` on a fresh session builder.
#[allow(dead_code)]
fn with_ep(provider: &str, ep: ort::ep::ExecutionProviderDispatch) -> Result<SessionBuilder> {
  Session::builder()?
    .with_execution_providers([ep.error_on_failure()])
    .map_err(|e| anyhow!("register {}: {}", provider, e))
}

/// Create a session builder registered with the given provider (by the
/// names of [`compiled_gpu_providers`]). Errors when the provider cannot be
/// initialised (missing driver / runtime libraries, no device, ...).
#[allow(unused_variables)]
fn builder_for(provider: &str, device_index: i32) -> Result<SessionBuilder> {
  match provider {
    #[cfg(feature = "tensorrt")]
    "TensorrtExecutionProvider" => with_ep(
      provider,
      ort::ep::TensorRT::default()
        .with_device_id(device_index)
        .build(),
    ),
    #[cfg(feature = "nvrtx")]
    "NvTensorRTRTXExecutionProvider" => with_ep(
      provider,
      ort::ep::NVRTX::default()
        .with_device_id(device_index.max(0) as u32)
        .build(),
    ),
    #[cfg(feature = "cuda")]
    "CUDAExecutionProvider" => with_ep(
      provider,
      ort::ep::CUDA::default()
        .with_device_id(device_index)
        .build(),
    ),
    #[cfg(feature = "migraphx")]
    "MIGraphXExecutionProvider" => with_ep(
      provider,
      ort::ep::MIGraphX::default()
        .with_device_id(device_index)
        .build(),
    ),
    #[cfg(feature = "rocm")]
    "ROCMExecutionProvider" => with_ep(
      provider,
      ort::ep::ROCm::default()
        .with_device_id(device_index)
        .build(),
    ),
    #[cfg(feature = "openvino")]
    "OpenVINOExecutionProvider" => with_ep(
      provider,
      ort::ep::OpenVINO::default()
        .with_device_type(if device_index > 0 {
          format!("GPU.{}", device_index)
        } else {
          "GPU".to_string()
        })
        .build(),
    ),
    #[cfg(feature = "directml")]
    "DmlExecutionProvider" => with_ep(
      provider,
      ort::ep::DirectML::default()
        .with_device_id(device_index)
        .build(),
    ),
    #[cfg(feature = "coreml")]
    "CoreMLExecutionProvider" => with_ep(provider, ort::ep::CoreML::default().build()),
    #[cfg(feature = "qnn")]
    "QNNExecutionProvider" => with_ep(
      provider,
      ort::ep::QNN::default().with_device_id(device_index).build(),
    ),
    #[cfg(feature = "cann")]
    "CANNExecutionProvider" => with_ep(
      provider,
      ort::ep::CANN::default()
        .with_device_id(device_index)
        .build(),
    ),
    #[cfg(feature = "nnapi")]
    "NnapiExecutionProvider" => with_ep(provider, ort::ep::NNAPI::default().build()),
    #[cfg(feature = "webgpu")]
    "WebGpuExecutionProvider" => with_ep(
      provider,
      ort::ep::WebGPU::default()
        .with_device_id(device_index)
        .build(),
    ),
    other => Err(anyhow!("execution provider {} is not compiled in", other)),
  }
}

/// Resolve `device` to a concrete backend: the first compiled GPU provider
/// that initialises, or the CPU. `Device::Gpu` / `Device::GpuIndex` return
/// an error instead of falling back.
pub fn resolve_backend(device: Device) -> Result<Backend> {
  if !device.wants_gpu() {
    return Ok(Backend::cpu());
  }
  let providers = compiled_gpu_providers();
  if providers.is_empty() {
    if device.requires_gpu() {
      return Err(anyhow!(
        "device {} requested but this build has no GPU support: rebuild with a GPU cargo \
         feature (e.g. `--features cuda`, see README)",
        device
      ));
    }
    return Ok(Backend::cpu());
  }
  let mut errors = Vec::new();
  for p in providers {
    match builder_for(p, device.index()) {
      Ok(_) => {
        return Ok(Backend {
          provider: p.to_string(),
          device_index: device.index(),
        })
      }
      Err(e) => errors.push(format!("{}: {}", p, e)),
    }
  }
  if device.requires_gpu() {
    Err(anyhow!(
      "device {} requested but no GPU execution provider could be initialised:\n  {}",
      device,
      errors.join("\n  ")
    ))
  } else {
    Ok(Backend::cpu())
  }
}

/// Session builder for a resolved backend.
pub fn session_builder(backend: &Backend) -> Result<SessionBuilder> {
  if backend.is_gpu() {
    builder_for(&backend.provider, backend.device_index)
  } else {
    Ok(Session::builder()?)
  }
}

/// Load an ONNX model on `backend`.
pub fn load_session<P: AsRef<Path>>(path: P, backend: &Backend) -> Result<Session> {
  let path = path.as_ref();
  session_builder(backend)?
    .commit_from_file(path)
    .map_err(|e| anyhow!("load onnx model {} on {}: {}", path.display(), backend, e))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn device_parsing() {
    assert_eq!(Device::parse("auto").unwrap(), Device::Auto);
    assert_eq!(Device::parse("CPU").unwrap(), Device::Cpu);
    assert_eq!(Device::parse("gpu").unwrap(), Device::Gpu);
    assert_eq!(Device::parse("gpu:1").unwrap(), Device::GpuIndex(1));
    assert!(Device::parse("tpu").is_err());
    assert!(Device::parse("gpu:x").is_err());
    assert_eq!(Device::default(), Device::Auto);
  }

  #[test]
  fn cpu_and_auto_resolve_without_gpu_features() {
    assert_eq!(resolve_backend(Device::Cpu).unwrap(), Backend::cpu());
    let auto = resolve_backend(Device::Auto).unwrap();
    if !gpu_support_compiled() {
      assert_eq!(auto, Backend::cpu());
      assert!(resolve_backend(Device::Gpu).is_err());
    }
    assert_eq!(Backend::cpu().to_string(), "CPU");
  }
}
