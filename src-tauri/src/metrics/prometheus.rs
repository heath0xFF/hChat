//! Minimal Prometheus text-exposition parser + server-metric derivation for
//! vLLM and llama.cpp `/metrics` endpoints. We don't pull a Prometheus client
//! crate — the format is simple enough to scan directly.

use crate::config::Runtime;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;

/// Parse a Prometheus text body into two maps keyed by metric base name:
/// - `all`: values summed across all label sets (right for vLLM, whose series
///   are split by `model_name`/`gpu`/`finish_reason` and have no global line).
/// - `bare`: only the value of the unlabeled (global-aggregate) series, when
///   present (right for llama-swap, which emits both a global line and per-model
///   lines — summing would double-count).
///
/// Histogram `_sum` / `_count` suffixes are kept as distinct names.
pub fn parse(body: &str) -> (HashMap<String, f64>, HashMap<String, f64>) {
    let mut all: HashMap<String, f64> = HashMap::new();
    let mut bare: HashMap<String, f64> = HashMap::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // `name{labels} value [timestamp]` or `name value`
        let brace = line.find('{');
        let ws = line.find(char::is_whitespace);
        let name_end = match (brace, ws) {
            (Some(b), Some(w)) => b.min(w),
            (Some(b), None) => b,
            (None, Some(w)) => w,
            (None, None) => line.len(),
        };
        let name = &line[..name_end];
        let has_label = brace.is_some_and(|b| ws.is_none_or(|w| b < w));
        let rest = if has_label {
            match line[name_end..].find('}') {
                Some(close) => &line[name_end + close + 1..],
                None => continue,
            }
        } else {
            &line[name_end..]
        };
        let Some(val_str) = rest.split_whitespace().next() else {
            continue;
        };
        let Ok(val) = val_str.parse::<f64>() else {
            continue;
        };
        *all.entry(name.to_string()).or_insert(0.0) += val;
        if !has_label {
            bare.insert(name.to_string(), val);
        }
    }
    (all, bare)
}

/// A scraped sample plus the time it was taken, for rate derivation.
#[derive(Clone)]
pub struct Sample {
    pub at: Instant,
    /// Summed-across-labels values.
    pub all: HashMap<String, f64>,
    /// Unlabeled (global-aggregate) values only.
    pub bare: HashMap<String, f64>,
}

#[derive(Default, Clone, Serialize)]
pub struct ServerStats {
    pub decode_tok_s: Option<f64>,
    pub prefill_tok_s: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub requests_running: Option<f64>,
    pub requests_waiting: Option<f64>,
    pub kv_cache_pct: Option<f64>,
    pub prompt_tokens_total: Option<f64>,
    pub generation_tokens_total: Option<f64>,
}

fn rate(prev: &Sample, cur: &Sample, key: &str) -> Option<f64> {
    let (a, b) = (prev.all.get(key)?, cur.all.get(key)?);
    let dt = cur.at.duration_since(prev.at).as_secs_f64();
    if dt <= 0.0 || b < a {
        return None; // counter reset or no elapsed time
    }
    Some((b - a) / dt)
}

fn hist_avg_ms(prev: &Sample, cur: &Sample, base: &str) -> Option<f64> {
    let sum = cur.all.get(&format!("{base}_sum"))?
        - prev.all.get(&format!("{base}_sum")).copied().unwrap_or(0.0);
    let count = cur.all.get(&format!("{base}_count"))?
        - prev.all.get(&format!("{base}_count")).copied().unwrap_or(0.0);
    if count <= 0.0 || sum < 0.0 {
        return None;
    }
    Some((sum / count) * 1000.0) // seconds → ms
}

/// Map parsed samples onto `ServerStats` per runtime dialect. `prev` enables
/// rate/delta derivation for counters and histograms.
pub fn derive(runtime: Runtime, prev: Option<&Sample>, cur: &Sample) -> ServerStats {
    let g = |k: &str| cur.all.get(k).copied();
    // For llama-swap, prefer the unlabeled global series (it also emits
    // per-model lines; summing both would double-count).
    let gb = |k: &str| cur.bare.get(k).or_else(|| cur.all.get(k)).copied();
    match runtime {
        Runtime::LlamaSwap => ServerStats {
            // llama-swap exposes rolling tok/s gauges (1/5/15-request windows).
            decode_tok_s: gb("llama_swap_generate_tokens_per_second_last_5"),
            prefill_tok_s: gb("llama_swap_prompt_tokens_per_second_last_5"),
            ttft_ms: None,
            requests_running: None, // no concurrency gauge exposed
            requests_waiting: None,
            kv_cache_pct: None,
            prompt_tokens_total: gb("llama_swap_input_tokens_total"),
            generation_tokens_total: gb("llama_swap_output_tokens_total"),
        },
        Runtime::Vllm => ServerStats {
            decode_tok_s: prev.and_then(|p| rate(p, cur, "vllm:generation_tokens_total")),
            prefill_tok_s: prev.and_then(|p| rate(p, cur, "vllm:prompt_tokens_total")),
            ttft_ms: prev.and_then(|p| hist_avg_ms(p, cur, "vllm:time_to_first_token_seconds")),
            requests_running: g("vllm:num_requests_running"),
            requests_waiting: g("vllm:num_requests_waiting"),
            kv_cache_pct: g("vllm:gpu_cache_usage_perc").map(|v| v * 100.0),
            prompt_tokens_total: g("vllm:prompt_tokens_total"),
            generation_tokens_total: g("vllm:generation_tokens_total"),
        },
        Runtime::LlamaCpp => ServerStats {
            // llama.cpp exposes instantaneous tok/s gauges directly.
            decode_tok_s: g("llamacpp:predicted_tokens_seconds"),
            prefill_tok_s: g("llamacpp:prompt_tokens_seconds"),
            ttft_ms: None,
            requests_running: g("llamacpp:requests_processing"),
            requests_waiting: g("llamacpp:requests_deferred"),
            kv_cache_pct: g("llamacpp:kv_cache_usage_ratio").map(|v| v * 100.0),
            prompt_tokens_total: g("llamacpp:prompt_tokens_total"),
            generation_tokens_total: g("llamacpp:tokens_predicted_total"),
        },
        _ => ServerStats::default(),
    }
}
