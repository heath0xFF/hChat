//! GPU/system metrics collectors.
//!
//! - **macmon** (Apple Silicon, no sudo): a dedicated OS thread owns the
//!   non-`Send` `Sampler` and publishes the latest reading into a shared slot.
//! - **agent**: poll a remote `hchat-agent` exposing nvidia-smi + /proc/meminfo
//!   as JSON (used for the DGX Spark, where nvidia-smi can't report unified VRAM).

use super::{GpuStats, LocalGpu};

/// Spawn the macmon sampler thread (macOS only). It writes the latest local-GPU
/// reading into `shared` roughly once per second. No-op on other platforms.
#[cfg(target_os = "macos")]
pub fn spawn_macmon(shared: LocalGpu) {
    std::thread::Builder::new()
        .name("macmon".into())
        .spawn(move || {
            let mut sampler = match macmon::Sampler::new() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("macmon: sampler init failed: {e:?}");
                    return;
                }
            };
            let soc = sampler.get_soc_info();
            let name = soc.chip_name.clone();
            let mem_total_gb = soc.memory_gb as f64;
            loop {
                // get_metrics blocks ~`duration` ms (sampling window), so this
                // loop self-paces near 1 Hz.
                match sampler.get_metrics(1000) {
                    Ok(m) => {
                        let used_gb = m.memory.ram_usage as f64 / 1e9;
                        let total_gb = if m.memory.ram_total > 0 {
                            m.memory.ram_total as f64 / 1e9
                        } else {
                            mem_total_gb
                        };
                        let util = (m.gpu_usage.1 as f64 * 100.0).clamp(0.0, 100.0);
                        let stats = GpuStats {
                            source: "macmon".into(),
                            vram_used_gb: Some(used_gb),
                            vram_total_gb: Some(total_gb),
                            power_w: Some(m.gpu_power as f64),
                            power_limit_w: None,
                            temp_c: Some(m.temp.gpu_temp_avg as f64),
                            util_pct: Some(util),
                            devices: vec![super::GpuDevice {
                                name: name.clone(),
                                util_pct: util,
                                mem_used_gb: used_gb,
                                mem_total_gb: total_gb,
                                temp_c: m.temp.gpu_temp_avg as f64,
                                power_w: m.gpu_power as f64,
                                power_limit_w: 0.0,
                            }],
                        };
                        *shared.lock().unwrap() = Some(stats);
                    }
                    Err(e) => eprintln!("macmon: sample error: {e:?}"),
                }
            }
        })
        .ok();
}

#[cfg(not(target_os = "macos"))]
pub fn spawn_macmon(_shared: LocalGpu) {}

/// Fetch GPU stats from a remote `hchat-agent`. The agent returns a JSON body
/// matching `GpuStats` (minus `source`, which we stamp here).
pub async fn fetch_agent(client: &reqwest::Client, base_url: &str) -> Option<GpuStats> {
    let url = format!("{}/gpu", base_url.trim_end_matches('/'));
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let mut stats: GpuStats = resp.json().await.ok()?;
    stats.source = "agent".into();
    Some(stats)
}
