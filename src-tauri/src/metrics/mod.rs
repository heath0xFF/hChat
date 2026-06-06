//! Live inference-metrics subsystem. A single background poller resolves the
//! currently-active backend (`MetricsTarget`), collects from its configured
//! sources every ~1.5s, and emits a `metrics` event to the frontend.
//!
//! Sources:
//! - GPU/system: macmon (local Apple Silicon) or a remote `hchat-agent`.
//! - Server: Prometheus `/metrics` scrape (vLLM / llama.cpp).
//! - Per-request decode/TTFT/prefill is measured client-side during streaming
//!   (see `commands::stream_once`) and reported separately.

pub mod gpu;
pub mod prometheus;

use crate::config::{GpuKind, Runtime};
use prometheus::ServerStats;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

/// Shared slot the macmon thread writes the latest local-GPU reading into.
pub type LocalGpu = Arc<Mutex<Option<GpuStats>>>;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct GpuDevice {
    pub name: String,
    pub util_pct: f64,
    pub mem_used_gb: f64,
    pub mem_total_gb: f64,
    pub temp_c: f64,
    pub power_w: f64,
    pub power_limit_w: f64,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct GpuStats {
    #[serde(default)]
    pub source: String,
    pub vram_used_gb: Option<f64>,
    pub vram_total_gb: Option<f64>,
    pub power_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub temp_c: Option<f64>,
    pub util_pct: Option<f64>,
    #[serde(default)]
    pub devices: Vec<GpuDevice>,
}

#[derive(Serialize, Clone)]
pub struct MetricsSnapshot {
    pub endpoint: String,
    pub runtime: String,
    pub gpu: Option<GpuStats>,
    pub server: Option<ServerStats>,
}

/// Managed Tauri state: the shared slot the poller reads and commands write.
pub struct MetricsHandle(pub Arc<Mutex<Option<MetricsTarget>>>);

/// The currently-active backend the dashboard is watching. Set from the
/// frontend via `set_metrics_target` whenever the chat endpoint changes.
#[derive(Clone)]
pub struct MetricsTarget {
    pub endpoint: String,
    pub runtime: Runtime,
    pub prometheus_url: Option<String>,
    pub gpu: GpuKind,
    pub agent_url: Option<String>,
}

fn runtime_label(r: Runtime) -> &'static str {
    match r {
        Runtime::Vllm => "VLLM",
        Runtime::Omlx => "OMLX",
        Runtime::LlamaCpp => "LLAMA.CPP",
        Runtime::LlamaSwap => "LLAMA-SWAP",
        Runtime::Openai => "OPENAI",
    }
}

/// Spawn the macmon sampler thread + the async poller. Returns the shared
/// `MetricsTarget` slot the command layer updates.
pub fn start(app: AppHandle) -> Arc<Mutex<Option<MetricsTarget>>> {
    let target: Arc<Mutex<Option<MetricsTarget>>> = Arc::new(Mutex::new(None));
    let local_gpu: LocalGpu = Arc::new(Mutex::new(None));
    gpu::spawn_macmon(local_gpu.clone());

    let target_for_task = target.clone();
    tauri::async_runtime::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap_or_default();
        // (endpoint, last prometheus sample) for rate derivation.
        let mut prev: Option<(String, prometheus::Sample)> = None;

        loop {
            tokio::time::sleep(Duration::from_millis(1500)).await;
            let Some(tgt) = target_for_task.lock().unwrap().clone() else {
                continue;
            };

            let gpu = match tgt.gpu {
                GpuKind::Macmon => local_gpu.lock().unwrap().clone(),
                GpuKind::Agent => match &tgt.agent_url {
                    Some(url) => gpu::fetch_agent(&client, url).await,
                    None => None,
                },
                GpuKind::None => None,
            };

            let server = if let Some(purl) = &tgt.prometheus_url {
                match client.get(purl).send().await {
                    Ok(r) if r.status().is_success() => {
                        let body = r.text().await.unwrap_or_default();
                        let (all, bare) = prometheus::parse(&body);
                        let cur = prometheus::Sample {
                            at: Instant::now(),
                            all,
                            bare,
                        };
                        let prev_sample = prev
                            .as_ref()
                            .filter(|(ep, _)| *ep == tgt.endpoint)
                            .map(|(_, s)| s);
                        let stats = prometheus::derive(tgt.runtime, prev_sample, &cur);
                        prev = Some((tgt.endpoint.clone(), cur));
                        Some(stats)
                    }
                    _ => None,
                }
            } else {
                None
            };

            let snapshot = MetricsSnapshot {
                endpoint: tgt.endpoint.clone(),
                runtime: runtime_label(tgt.runtime).to_string(),
                gpu,
                server,
            };
            let _ = app.emit("metrics", snapshot);
        }
    });

    target
}
