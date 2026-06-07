//! Tauri command layer. Replaces the egui `app.rs` orchestration: each command
//! locks `AppState` briefly for synchronous DB/config work and drops the guard
//! before any `.await`. Streaming chat forwards `StreamEvent`s from the core
//! `api::stream_chat` onto a frontend `Channel<ChatEvent>`.

use crate::api::{self, ChatParams, StreamEvent};
use crate::config::Config;
use crate::message::{ContentPart, ImageUrl, Message, Role, ToolCall};
use crate::state::AppState;
use crate::storage::ConversationSettings;
use crate::tools::{self, Handler, Safety, ToolDef};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tauri::State;
use tauri::ipc::Channel;
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// Max tool-call → re-stream cycles per user turn (matches the egui app).
const MAX_TOOL_ITERATIONS: usize = 8;

// ---------- DTOs ----------

#[derive(Serialize)]
pub struct ConversationDto {
    pub id: i64,
    pub title: String,
    pub pinned: bool,
    pub project_id: Option<i64>,
    /// Short runtime code of the conversation's endpoint (omlx/llamaswap/…),
    /// for the sidebar backend icon.
    pub runtime: Option<String>,
}

#[derive(Serialize)]
pub struct ProjectDto {
    pub id: i64,
    pub name: String,
    pub pinned: bool,
}

#[derive(Serialize)]
pub struct UsageStatsDto {
    pub total_requests: i64,
    pub ok_requests: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub total_cost: f64,
    pub ttft_p50_ms: Option<f64>,
    pub ttft_p95_ms: Option<f64>,
    pub decode_p50_tok_s: Option<f64>,
    pub decode_p95_tok_s: Option<f64>,
    pub by_model: Vec<UsageByModelDto>,
    pub daily: Vec<UsageDailyDto>,
}

#[derive(Serialize)]
pub struct UsageByModelDto {
    pub model: String,
    pub endpoint: String,
    pub requests: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub cost: f64,
    pub avg_ttft_ms: Option<f64>,
    pub avg_decode_tok_s: Option<f64>,
}

#[derive(Serialize)]
pub struct UsageDailyDto {
    pub date: String,
    pub total_tokens: i64,
}

#[derive(Deserialize)]
pub struct BenchParams {
    pub endpoint: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub concurrency: u32,
    pub total_requests: u32,
}

/// avg / p50 / p95 of a metric across the benchmark's successful requests.
#[derive(Serialize, Default)]
pub struct BenchAgg {
    pub avg: Option<f64>,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
}

#[derive(Serialize)]
pub struct BenchResult {
    pub requests: u32,
    pub ok: u32,
    pub errors: u32,
    pub wall_ms: f64,
    pub ttft_ms: BenchAgg,
    pub decode_tok_s: BenchAgg,
    /// Aggregate decode throughput: completion tokens across all successful
    /// requests divided by wall-clock time (the headline concurrency number).
    pub agg_decode_tok_s: f64,
    pub total_completion_tokens: u64,
    /// Per-request points (successful only), in completion order, for charts.
    pub ttft_series: Vec<f64>,
    pub decode_series: Vec<f64>,
}

#[derive(Serialize)]
pub struct ToolCallDto {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize)]
pub struct MessageDto {
    pub id: Option<i64>,
    pub role: String,
    /// Concatenated text content (may contain `<think>` reasoning blocks and
    /// fenced code — the frontend segments + renders it).
    pub text: String,
    /// Data/HTTP image URLs attached to this message, in order.
    pub images: Vec<String>,
    pub tool_calls: Option<Vec<ToolCallDto>>,
    pub tool_call_id: Option<String>,
    /// Unix epoch ms (session-local for live messages, persisted for loaded ones).
    pub created_at: Option<i64>,
}

#[derive(Serialize)]
pub struct ConversationData {
    pub messages: Vec<MessageDto>,
    pub settings: SettingsDto,
    pub draft: Option<String>,
}

/// Effective per-conversation settings (conversation overrides folded over the
/// global config defaults), ready for the UI to display as concrete values.
#[derive(Serialize)]
pub struct SettingsDto {
    pub model: Option<String>,
    pub endpoint: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub use_max_tokens: bool,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Vec<String>,
    pub working_dir: Option<String>,
}

/// Model + endpoint + sampling knobs shared by send/regenerate/edit. The API
/// key is *not* here — it's resolved server-side from `config.saved_endpoints`
/// so secrets never round-trip through JS. Flattened into the request structs
/// so the JS payload stays a single flat object.
#[derive(Deserialize, Clone)]
pub struct GenParams {
    pub endpoint: String,
    pub model: String,
    pub system_prompt: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub use_max_tokens: bool,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

#[derive(Deserialize)]
pub struct SendParams {
    pub conversation_id: Option<i64>,
    #[serde(flatten)]
    pub gp: GenParams,
    pub user_text: String,
    /// Image attachments as data/HTTP URLs, rendered into `image_url` content
    /// parts on the user message.
    #[serde(default)]
    pub images: Vec<String>,
}

#[derive(Deserialize)]
pub struct RegenerateParams {
    pub conversation_id: i64,
    #[serde(flatten)]
    pub gp: GenParams,
}

#[derive(Deserialize)]
pub struct EditParams {
    pub conversation_id: i64,
    pub message_id: i64,
    pub new_text: String,
    #[serde(flatten)]
    pub gp: GenParams,
}

/// Streaming events pushed to the frontend over the per-request `Channel`.
#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// Conversation id, emitted first (covers the create-on-first-send case so
    /// the UI can adopt the new id immediately).
    Started { conversation_id: i64 },
    /// A new assistant turn is starting — the frontend pushes a fresh assistant
    /// placeholder. Emitted before the first turn and before each tool-driven
    /// continuation turn.
    TurnStart,
    Token { text: String },
    Reasoning { text: String },
    /// The model requested a tool call (attached to the current assistant
    /// message as a chip).
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    /// A Confirm-safety tool needs the user's go-ahead before it runs. The
    /// frontend shows an approval card and calls `resolve_tool`.
    ToolApproval {
        id: String,
        name: String,
        arguments: String,
    },
    /// A tool finished (or was denied) — the frontend appends a tool result
    /// bubble.
    ToolResult {
        id: String,
        name: String,
        result: String,
        is_error: bool,
    },
    Usage {
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
        total_tokens: Option<u32>,
        cost: Option<f64>,
    },
    /// Per-request client-measured metrics (feed the Status dashboard tiles).
    RequestMetrics {
        ttft_ms: Option<f64>,
        decode_tok_s: Option<f64>,
        prefill_tok_s: Option<f64>,
        duration_ms: f64,
    },
    Done { message_id: Option<i64> },
    Error { message: String },
}

// ---------- helpers ----------

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn message_to_dto(m: &Message) -> MessageDto {
    let images = m
        .content
        .iter()
        .filter_map(|p| match p {
            ContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
            _ => None,
        })
        .collect();
    let tool_calls = m.tool_calls.as_ref().map(|calls| {
        calls
            .iter()
            .map(|c| ToolCallDto {
                id: c.id.clone(),
                name: c.function.name.clone(),
                arguments: c.function.arguments.clone(),
            })
            .collect()
    });
    MessageDto {
        id: m.id,
        role: role_str(&m.role).to_string(),
        text: m.text_str(),
        images,
        tool_calls,
        tool_call_id: m.tool_call_id.clone(),
        created_at: m.created_at,
    }
}

/// Fold conversation overrides over global config defaults.
fn effective_settings(cfg: &Config, cs: &ConversationSettings) -> SettingsDto {
    SettingsDto {
        model: cs.model.clone(),
        endpoint: cs.endpoint.clone().unwrap_or_else(|| cfg.default_endpoint.clone()),
        system_prompt: cs
            .system_prompt
            .clone()
            .unwrap_or_else(|| cfg.system_prompt.clone()),
        temperature: cs.temperature.unwrap_or(cfg.temperature),
        max_tokens: cs.max_tokens.unwrap_or(cfg.max_tokens),
        use_max_tokens: cs.use_max_tokens,
        top_p: cs.top_p.or(cfg.top_p),
        frequency_penalty: cs.frequency_penalty.or(cfg.frequency_penalty),
        presence_penalty: cs.presence_penalty.or(cfg.presence_penalty),
        stop_sequences: if cs.stop_sequences.is_empty() {
            cfg.stop_sequences.clone()
        } else {
            cs.stop_sequences.clone()
        },
        working_dir: cs.working_dir.clone(),
    }
}

// ---------- config / models ----------

#[tauri::command]
pub fn get_config(state: State<'_, AppState>) -> Config {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
pub fn save_config(state: State<'_, AppState>, mut config: Config) -> Result<(), String> {
    config.sanitize();
    config.save()?;
    *state.config.lock().unwrap() = config;
    Ok(())
}

#[tauri::command]
pub async fn fetch_models(
    state: State<'_, AppState>,
    endpoint: String,
) -> Result<Vec<String>, String> {
    let api_key = state.api_key_for(&endpoint);
    api::fetch_models(&endpoint, api_key.as_deref()).await
}

// ---------- conversations ----------

fn runtime_code(r: crate::config::Runtime) -> &'static str {
    use crate::config::Runtime::*;
    match r {
        Vllm => "vllm",
        Omlx => "omlx",
        LlamaCpp => "llamacpp",
        LlamaSwap => "llamaswap",
        Openai => "openai",
    }
}

#[tauri::command]
pub fn list_conversations(state: State<'_, AppState>) -> Vec<ConversationDto> {
    let convs = state.storage.lock().unwrap().list_conversations();
    let cfg = state.config.lock().unwrap();
    convs
        .into_iter()
        .map(|c| {
            let runtime = c
                .endpoint
                .as_ref()
                .and_then(|ep| cfg.saved_endpoints.iter().find(|e| &e.url == ep))
                .map(|e| runtime_code(e.runtime).to_string());
            ConversationDto {
                id: c.id,
                title: c.title,
                pinned: c.pinned,
                project_id: c.project_id,
                runtime,
            }
        })
        .collect()
}

// ---------- usage ----------

#[tauri::command]
pub fn usage_stats(state: State<'_, AppState>) -> UsageStatsDto {
    let retention = state.config.lock().unwrap().usage_retention_days;
    let s = {
        let storage = state.storage.lock().unwrap();
        storage.prune_usage(retention);
        storage.usage_stats()
    };
    UsageStatsDto {
        total_requests: s.total_requests,
        ok_requests: s.ok_requests,
        prompt_tokens: s.prompt_tokens,
        completion_tokens: s.completion_tokens,
        total_tokens: s.total_tokens,
        total_cost: s.total_cost,
        ttft_p50_ms: s.ttft_p50_ms,
        ttft_p95_ms: s.ttft_p95_ms,
        decode_p50_tok_s: s.decode_p50_tok_s,
        decode_p95_tok_s: s.decode_p95_tok_s,
        by_model: s
            .by_model
            .into_iter()
            .map(|m| UsageByModelDto {
                model: m.model,
                endpoint: m.endpoint,
                requests: m.requests,
                prompt_tokens: m.prompt_tokens,
                completion_tokens: m.completion_tokens,
                total_tokens: m.total_tokens,
                cost: m.cost,
                avg_ttft_ms: m.avg_ttft_ms,
                avg_decode_tok_s: m.avg_decode_tok_s,
            })
            .collect(),
        daily: s
            .daily
            .into_iter()
            .map(|d| UsageDailyDto {
                date: d.date,
                total_tokens: d.total_tokens,
            })
            .collect(),
    }
}

#[tauri::command]
pub fn clear_usage(state: State<'_, AppState>) {
    state.storage.lock().unwrap().clear_usage();
}

// ---------- benchmark ----------

struct BenchSample {
    ttft_ms: Option<f64>,
    decode_tok_s: Option<f64>,
    completion: u32,
    ok: bool,
}

/// Nearest-rank percentile of a pre-sorted slice (`q` in 0.0..=1.0).
fn bench_percentile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    Some(sorted[idx])
}

fn bench_agg(vals: &[f64]) -> BenchAgg {
    if vals.is_empty() {
        return BenchAgg::default();
    }
    let avg = vals.iter().sum::<f64>() / vals.len() as f64;
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    BenchAgg {
        avg: Some(avg),
        p50: bench_percentile(&sorted, 0.50),
        p95: bench_percentile(&sorted, 0.95),
    }
}

/// One benchmark request: stream a single completion and measure TTFT + decode.
async fn bench_once(
    endpoint: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    api_key: Option<String>,
) -> BenchSample {
    let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();
    let params = ChatParams {
        base_url: endpoint,
        model,
        messages: vec![Message::text(Role::User, prompt)],
        temperature: None,
        max_tokens: Some(max_tokens),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop_sequences: None,
        api_key,
        tools: None,
    };
    let start = Instant::now();
    api::stream_chat(params, tx, CancellationToken::new());

    let mut first_token: Option<Instant> = None;
    let mut completion: u32 = 0;
    let mut errored = false;
    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Token(_) | StreamEvent::Reasoning(_) => {
                first_token.get_or_insert_with(Instant::now);
            }
            StreamEvent::UsageInfo(u) => {
                if let Some(c) = u.completion_tokens {
                    completion = c;
                }
            }
            StreamEvent::Error(_) => errored = true,
            StreamEvent::Done => break,
            _ => {}
        }
    }
    if errored {
        return BenchSample {
            ttft_ms: None,
            decode_tok_s: None,
            completion: 0,
            ok: false,
        };
    }
    let now = Instant::now();
    let ttft_ms = first_token.map(|f| f.duration_since(start).as_secs_f64() * 1000.0);
    let decode_tok_s = match first_token {
        Some(f) if completion > 0 => {
            let secs = now.duration_since(f).as_secs_f64();
            (secs > 0.0).then(|| completion as f64 / secs)
        }
        _ => None,
    };
    BenchSample {
        ttft_ms,
        decode_tok_s,
        completion,
        ok: true,
    }
}

/// Fire `total_requests` completions at an endpoint (≤ `concurrency` in flight)
/// and report TTFT/decode percentiles plus aggregate throughput.
#[tauri::command]
pub async fn run_benchmark(
    state: State<'_, AppState>,
    params: BenchParams,
) -> Result<BenchResult, String> {
    if params.model.trim().is_empty() {
        return Err("Pick a model to benchmark.".into());
    }
    let concurrency = params.concurrency.clamp(1, 64) as usize;
    let total = params.total_requests.clamp(1, 500);
    let max_tokens = params.max_tokens.clamp(1, 4096);
    let prompt = if params.prompt.trim().is_empty() {
        "Write a few sentences about the history of computing.".to_string()
    } else {
        params.prompt.clone()
    };
    let api_key = state.api_key_for(&params.endpoint);

    let sem = Arc::new(Semaphore::new(concurrency));
    let mut handles = Vec::with_capacity(total as usize);
    let wall_start = Instant::now();
    for _ in 0..total {
        let permit_src = sem.clone();
        let endpoint = params.endpoint.clone();
        let model = params.model.clone();
        let prompt = prompt.clone();
        let key = api_key.clone();
        handles.push(tokio::spawn(async move {
            let _permit = permit_src.acquire_owned().await.ok()?;
            Some(bench_once(endpoint, model, prompt, max_tokens, key).await)
        }));
    }

    let mut samples = Vec::with_capacity(total as usize);
    for h in handles {
        if let Ok(Some(s)) = h.await {
            samples.push(s);
        }
    }
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

    let ok_count = samples.iter().filter(|s| s.ok).count() as u32;
    let errors = samples.len() as u32 - ok_count;
    let total_completion: u64 = samples
        .iter()
        .filter(|s| s.ok)
        .map(|s| s.completion as u64)
        .sum();
    let ttft_series: Vec<f64> = samples.iter().filter_map(|s| s.ttft_ms).collect();
    let decode_series: Vec<f64> = samples.iter().filter_map(|s| s.decode_tok_s).collect();
    let agg_decode_tok_s = if wall_ms > 0.0 {
        total_completion as f64 / (wall_ms / 1000.0)
    } else {
        0.0
    };

    Ok(BenchResult {
        requests: total,
        ok: ok_count,
        errors,
        wall_ms,
        ttft_ms: bench_agg(&ttft_series),
        decode_tok_s: bench_agg(&decode_series),
        agg_decode_tok_s,
        total_completion_tokens: total_completion,
        ttft_series,
        decode_series,
    })
}

// ---------- projects ----------

#[tauri::command]
pub fn list_projects(state: State<'_, AppState>) -> Vec<ProjectDto> {
    state
        .storage
        .lock()
        .unwrap()
        .list_projects()
        .into_iter()
        .map(|p| ProjectDto {
            id: p.id,
            name: p.name,
            pinned: p.pinned,
        })
        .collect()
}

#[tauri::command]
pub fn create_project(state: State<'_, AppState>, name: String) -> Result<i64, String> {
    state.storage.lock().unwrap().create_project(&name)
}

#[tauri::command]
pub fn rename_project(state: State<'_, AppState>, id: i64, name: String) {
    state.storage.lock().unwrap().rename_project(id, &name);
}

#[tauri::command]
pub fn delete_project(state: State<'_, AppState>, id: i64) {
    state.storage.lock().unwrap().delete_project(id);
}

#[tauri::command]
pub fn set_project_pinned(state: State<'_, AppState>, id: i64, pinned: bool) {
    state.storage.lock().unwrap().set_project_pinned(id, pinned);
}

#[tauri::command]
pub fn set_conversation_project(
    state: State<'_, AppState>,
    conversation_id: i64,
    project_id: Option<i64>,
) {
    state
        .storage
        .lock()
        .unwrap()
        .set_conversation_project(conversation_id, project_id);
}

#[tauri::command]
pub fn load_conversation(state: State<'_, AppState>, id: i64) -> ConversationData {
    let storage = state.storage.lock().unwrap();
    let cfg = state.config.lock().unwrap();
    let messages = storage
        .load_messages(id)
        .iter()
        .map(message_to_dto)
        .collect();
    let settings = effective_settings(&cfg, &storage.load_conversation_settings(id));
    let draft = storage.load_draft(id).filter(|d| !d.is_empty());
    ConversationData {
        messages,
        settings,
        draft,
    }
}

#[tauri::command]
pub fn save_draft(state: State<'_, AppState>, id: i64, text: String) {
    state.storage.lock().unwrap().save_draft(id, &text);
}

// ---------- ~/.agents (commands + skills) ----------

#[derive(Serialize)]
pub struct AgentsDto {
    pub commands: Vec<crate::agents::AgentCommand>,
    pub skills: Vec<crate::agents::Skill>,
}

/// List slash commands + skills discovered from `~/.agents` (and the project's
/// `.agents/` under `working_dir`). Tools from the same dirs are loaded into the
/// model's tool set during a turn, not surfaced here.
#[tauri::command]
pub fn list_agents(working_dir: Option<String>) -> AgentsDto {
    let wd = working_dir.map(PathBuf::from);
    let bundle = crate::agents::load(wd.as_deref());
    AgentsDto {
        commands: bundle.commands,
        skills: bundle.skills,
    }
}

// ---------- MCP ----------

#[tauri::command]
pub async fn list_mcp_servers(
    state: State<'_, AppState>,
) -> Result<Vec<crate::mcp::McpStatus>, String> {
    Ok(state.mcp.status().await)
}

/// Reconnect all configured MCP servers (after editing config).
#[tauri::command]
pub async fn reconnect_mcp(state: State<'_, AppState>) -> Result<(), String> {
    let servers = state.config.lock().unwrap().mcp_servers.clone();
    state.mcp.connect_all(servers).await;
    Ok(())
}

#[tauri::command]
pub fn delete_conversation(state: State<'_, AppState>, id: i64) {
    state.storage.lock().unwrap().delete_conversation(id);
}

#[tauri::command]
pub fn delete_all_conversations(state: State<'_, AppState>) -> Result<(), String> {
    state.storage.lock().unwrap().delete_all_conversations()
}

#[tauri::command]
pub fn rename_conversation(state: State<'_, AppState>, id: i64, title: String) {
    let storage = state.storage.lock().unwrap();
    storage.update_conversation_title(id, &title);
    // A manual rename is intentional — mark it so the one-shot AI auto-titler
    // won't overwrite the user's chosen name on the next turn.
    storage.mark_auto_titled(id);
}

#[tauri::command]
pub fn set_pinned(state: State<'_, AppState>, id: i64, pinned: bool) {
    state.storage.lock().unwrap().set_pinned(id, pinned);
}

#[tauri::command]
pub fn search_conversations(
    state: State<'_, AppState>,
    query: String,
) -> Vec<(i64, String, String)> {
    state.storage.lock().unwrap().search(&query)
}

#[tauri::command]
pub fn export_conversation(state: State<'_, AppState>, id: i64) -> String {
    state.storage.lock().unwrap().export_markdown(id)
}

/// Write a conversation's markdown to the user's download dir; returns the path.
#[tauri::command]
pub fn export_conversation_file(state: State<'_, AppState>, id: i64) -> Result<String, String> {
    let (markdown, title) = {
        let s = state.storage.lock().unwrap();
        let md = s.export_markdown(id);
        let title = s
            .list_conversations()
            .into_iter()
            .find(|c| c.id == id)
            .map(|c| c.title)
            .unwrap_or_default();
        (md, title)
    };
    let safe: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    let safe = safe.trim_matches('-');
    let name = if safe.is_empty() {
        format!("fornax-{id}")
    } else {
        safe.to_string()
    };
    let dir = dirs::download_dir()
        .or_else(dirs::home_dir)
        .ok_or("Could not find a download directory")?;
    let path = dir.join(format!("{name}.md"));
    std::fs::write(&path, markdown).map_err(|e| format!("Write failed: {e}"))?;
    Ok(path.display().to_string())
}

// ---------- presets ----------

#[derive(Serialize)]
pub struct PresetDto {
    pub id: i64,
    pub name: String,
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub use_max_tokens: bool,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Vec<String>,
}

#[tauri::command]
pub fn list_presets(state: State<'_, AppState>) -> Vec<PresetDto> {
    state
        .storage
        .lock()
        .unwrap()
        .list_presets()
        .into_iter()
        .map(|p| {
            let s = p.settings;
            PresetDto {
                id: p.id,
                name: p.name,
                endpoint: s.endpoint,
                model: s.model,
                system_prompt: s.system_prompt,
                temperature: s.temperature,
                max_tokens: s.max_tokens,
                use_max_tokens: s.use_max_tokens,
                top_p: s.top_p,
                frequency_penalty: s.frequency_penalty,
                presence_penalty: s.presence_penalty,
                stop_sequences: s.stop_sequences,
            }
        })
        .collect()
}

#[tauri::command]
pub fn create_preset(
    state: State<'_, AppState>,
    name: String,
    gp: GenParams,
) -> Result<i64, String> {
    let cs = settings_from_gen(&gp, None);
    state.storage.lock().unwrap().create_preset(&name, &cs)
}

#[tauri::command]
pub fn delete_preset(state: State<'_, AppState>, id: i64) {
    state.storage.lock().unwrap().delete_preset(id);
}

// ---------- metrics ----------

/// Point the metrics dashboard at `endpoint`. Resolves the runtime / prometheus
/// URL / GPU source from `config.saved_endpoints`; unknown endpoints fall back
/// to a generic OpenAI backend (no GPU/server metrics, client-measured only).
#[tauri::command]
pub fn set_metrics_target(
    state: State<'_, AppState>,
    metrics: State<'_, crate::metrics::MetricsHandle>,
    endpoint: String,
) {
    let mut target = {
        let cfg = state.config.lock().unwrap();
        cfg.saved_endpoints
            .iter()
            .find(|e| e.url == endpoint)
            .map(|e| crate::metrics::MetricsTarget {
                endpoint: e.url.clone(),
                runtime: e.runtime,
                prometheus_url: e.prometheus_url.clone(),
                gpu: e.gpu,
                agent_url: e.agent_url.clone(),
            })
            .unwrap_or_else(|| crate::metrics::MetricsTarget {
                endpoint: endpoint.clone(),
                runtime: crate::config::Runtime::Openai,
                prometheus_url: None,
                gpu: crate::config::GpuKind::None,
                agent_url: None,
            })
    };

    // Convenience default: a local endpoint on macOS with no explicit GPU source
    // shows this Mac's unified-memory GPU via macmon, no config needed.
    #[cfg(target_os = "macos")]
    if target.gpu == crate::config::GpuKind::None && is_local_host(&target.endpoint) {
        target.gpu = crate::config::GpuKind::Macmon;
    }

    *metrics.0.lock().unwrap() = Some(target);
}

#[cfg(target_os = "macos")]
fn is_local_host(url: &str) -> bool {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_ascii_lowercase()))
        .is_some_and(|h| h == "localhost" || h == "127.0.0.1" || h == "0.0.0.0")
}

// ---------- streaming chat ----------

#[tauri::command]
pub fn cancel_stream(state: State<'_, AppState>, conversation_id: i64) {
    if let Some(token) = state.cancel.lock().unwrap().remove(&conversation_id) {
        token.cancel();
    }
}

/// Resolve a parked tool approval. `decision` is "approve", "approve_all"
/// (also adds the tool to this conversation's auto-approve allowlist), or
/// anything else (deny).
#[tauri::command]
pub fn resolve_tool(state: State<'_, AppState>, call_id: String, decision: String) {
    let pending = state.pending_approvals.lock().unwrap().remove(&call_id);
    if let Some(p) = pending {
        let approved = match decision.as_str() {
            "approve" => true,
            "approve_all" => {
                state
                    .auto_approved
                    .lock()
                    .unwrap()
                    .insert((p.conversation_id, p.tool.clone()));
                true
            }
            _ => false,
        };
        let _ = p.tx.send(approved);
    }
}

fn settings_from_gen(gp: &GenParams, working_dir: Option<String>) -> ConversationSettings {
    ConversationSettings {
        model: Some(gp.model.clone()),
        system_prompt: Some(gp.system_prompt.clone()),
        temperature: gp.temperature,
        max_tokens: gp.max_tokens,
        use_max_tokens: gp.use_max_tokens,
        top_p: gp.top_p,
        frequency_penalty: gp.frequency_penalty,
        presence_penalty: gp.presence_penalty,
        stop_sequences: gp.stop_sequences.clone(),
        endpoint: Some(gp.endpoint.clone()),
        working_dir,
    }
}

fn resolve_wd(working_dir: Option<String>) -> PathBuf {
    working_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")))
}

/// Synthetic tool the model calls to pull a skill's full instructions on demand.
fn use_skill_tool() -> ToolDef {
    ToolDef {
        name: "use_skill".to_string(),
        description:
            "Load the full instructions for one of the available skills by name, then follow them."
                .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "The skill name to load." }
            },
            "required": ["name"]
        }),
        handler: Handler::Builtin(tools::BuiltinRef("use_skill".to_string())),
        safety: Safety::Auto,
    }
}

/// System message advertising the available skills to the model.
fn skills_system_prompt(skills: &[crate::agents::Skill]) -> String {
    let mut s = String::from(
        "You have access to the following skills. When one is relevant to the user's \
         request, call the `use_skill` tool with its name to load the full instructions, \
         then follow them.\n\nAvailable skills:\n",
    );
    for sk in skills {
        if sk.description.is_empty() {
            s.push_str(&format!("- {}\n", sk.name));
        } else {
            s.push_str(&format!("- {}: {}\n", sk.name, sk.description));
        }
    }
    s
}

#[tauri::command]
pub async fn send_message(
    state: State<'_, AppState>,
    params: SendParams,
    on_event: Channel<ChatEvent>,
) -> Result<i64, String> {
    let (conversation_id, messages, api_key, working_dir, first_branch) = {
        let storage = state.storage.lock().unwrap();
        let conversation_id = match params.conversation_id {
            Some(id) => id,
            None => storage.create_conversation("untitled chat")?,
        };
        let mut messages = storage.load_messages(conversation_id);
        let mut parts = vec![ContentPart::Text {
            text: params.user_text.clone(),
        }];
        for url in &params.images {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: url.clone(),
                    detail: None,
                },
            });
        }
        messages.push(Message::from_parts(Role::User, parts));

        let prior_wd = storage.load_conversation_settings(conversation_id).working_dir;
        storage.save_messages(conversation_id, &mut messages)?;
        storage.save_conversation_settings(
            conversation_id,
            &settings_from_gen(&params.gp, prior_wd.clone()),
        );

        let parent = messages.last().and_then(|m| m.id);
        let first_branch = storage.next_branch_index(conversation_id, parent);
        (
            conversation_id,
            messages,
            state.api_key_for(&params.gp.endpoint),
            resolve_wd(prior_wd),
            first_branch,
        )
    };

    on_event.send(ChatEvent::Started { conversation_id }).ok();
    let (final_id, errored) = run_turn(
        &state, conversation_id, messages, &params.gp, api_key, working_dir, first_branch,
        &on_event,
    )
    .await;
    if !errored {
        maybe_auto_title(&state, conversation_id, &params.gp, &params.user_text).await;
    }
    on_event
        .send(ChatEvent::Done {
            message_id: final_id,
        })
        .ok();
    Ok(conversation_id)
}

/// Regenerate the last assistant turn as a new sibling branch under the same
/// user message. The previous reply is kept on disk and reachable via the
/// ◀ N/M ▶ navigator.
#[tauri::command]
pub async fn regenerate(
    state: State<'_, AppState>,
    params: RegenerateParams,
    on_event: Channel<ChatEvent>,
) -> Result<i64, String> {
    let conversation_id = params.conversation_id;
    let (messages, api_key, working_dir, first_branch) = {
        let storage = state.storage.lock().unwrap();
        let full = storage.load_messages(conversation_id);
        let Some(idx) = full.iter().rposition(|m| m.role == Role::User) else {
            return Err("Nothing to regenerate".into());
        };
        let messages: Vec<Message> = full[..=idx].to_vec();
        let parent = messages.last().and_then(|m| m.id);
        let first_branch = storage.next_branch_index(conversation_id, parent);
        let prior_wd = storage.load_conversation_settings(conversation_id).working_dir;
        storage.save_conversation_settings(
            conversation_id,
            &settings_from_gen(&params.gp, prior_wd.clone()),
        );
        (
            messages,
            state.api_key_for(&params.gp.endpoint),
            resolve_wd(prior_wd),
            first_branch,
        )
    };

    on_event.send(ChatEvent::Started { conversation_id }).ok();
    let (final_id, _) = run_turn(
        &state, conversation_id, messages, &params.gp, api_key, working_dir, first_branch,
        &on_event,
    )
    .await;
    on_event
        .send(ChatEvent::Done {
            message_id: final_id,
        })
        .ok();
    Ok(conversation_id)
}

/// Edit a (user) message and resend: the edited text becomes a new sibling
/// under the original's parent, and a fresh assistant reply streams from there.
#[tauri::command]
pub async fn edit_message(
    state: State<'_, AppState>,
    params: EditParams,
    on_event: Channel<ChatEvent>,
) -> Result<i64, String> {
    let conversation_id = params.conversation_id;
    let (messages, api_key, working_dir) = {
        let storage = state.storage.lock().unwrap();
        let full = storage.load_messages(conversation_id);
        let Some(idx) = full.iter().position(|m| m.id == Some(params.message_id)) else {
            return Err("Message not found".into());
        };
        let original = &full[idx];
        let parent = original.parent_id;
        let branch = storage.next_branch_index(conversation_id, parent);

        let mut messages: Vec<Message> = full[..idx].to_vec();
        let mut edited = Message::text(original.role.clone(), params.new_text.clone());
        edited.parent_id = parent;
        edited.branch_index = branch;
        messages.push(edited);

        let prior_wd = storage.load_conversation_settings(conversation_id).working_dir;
        storage.save_conversation_settings(
            conversation_id,
            &settings_from_gen(&params.gp, prior_wd.clone()),
        );
        storage.save_messages(conversation_id, &mut messages)?;
        (
            messages,
            state.api_key_for(&params.gp.endpoint),
            resolve_wd(prior_wd),
        )
    };

    on_event.send(ChatEvent::Started { conversation_id }).ok();
    // The assistant reply is the first child of the new user sibling → branch 0.
    let (final_id, _) = run_turn(
        &state, conversation_id, messages, &params.gp, api_key, working_dir, 0, &on_event,
    )
    .await;
    on_event
        .send(ChatEvent::Done {
            message_id: final_id,
        })
        .ok();
    Ok(conversation_id)
}

/// Drive one user turn through the tool loop: stream → persist assistant →
/// execute tools → re-stream, up to `MAX_TOOL_ITERATIONS`. `messages` must
/// already contain the persisted history + the triggering user message.
/// Returns `(final_assistant_message_id, errored)`.
#[allow(clippy::too_many_arguments)]
async fn run_turn(
    state: &State<'_, AppState>,
    conversation_id: i64,
    mut messages: Vec<Message>,
    gp: &GenParams,
    api_key: Option<String>,
    working_dir: PathBuf,
    first_assistant_branch_index: i64,
    on_event: &Channel<ChatEvent>,
) -> (Option<i64>, bool) {
    // Register the cancellation token up front (before any `.await`) so a
    // `cancel_stream` arriving during tool/MCP discovery isn't a no-op.
    let cancel = CancellationToken::new();
    state
        .cancel
        .lock()
        .unwrap()
        .insert(conversation_id, cancel.clone());

    // User tools (loaded fresh each turn so edits hot-reload) + tools/skills
    // from the ~/.agents convention (user-level and project-local).
    let mut tool_defs: Vec<ToolDef> = tools::load_from_dir(&tools::user_tools_dir());
    let bundle = crate::agents::load(Some(&working_dir));
    tool_defs.extend(bundle.tools);
    let skills = bundle.skills;
    if !skills.is_empty() {
        tool_defs.push(use_skill_tool());
    }
    let skills_prompt = (!skills.is_empty()).then(|| skills_system_prompt(&skills));
    // Native + agent tools, plus tools discovered from connected MCP servers.
    let mut api_specs = tools::to_api_shape(&tool_defs);
    api_specs.extend(state.mcp.tool_specs().await);
    let tools_api = if api_specs.is_empty() {
        None
    } else {
        Some(api_specs)
    };

    let mut final_message_id = None;
    let mut errored = false;

    for iteration in 0..MAX_TOOL_ITERATIONS {
        if cancel.is_cancelled() {
            break;
        }
        on_event.send(ChatEvent::TurnStart).ok();

        let mut wire = Vec::with_capacity(messages.len() + 2);
        if !gp.system_prompt.trim().is_empty() {
            wire.push(Message::text(Role::System, gp.system_prompt.clone()));
        }
        if let Some(sp) = &skills_prompt {
            wire.push(Message::text(Role::System, sp.clone()));
        }
        wire.extend(messages.iter().cloned());

        let outcome =
            stream_once(gp, wire, api_key.clone(), tools_api.clone(), cancel.clone(), on_event)
                .await;
        // Record this turn's token spend for the Usage page (tool-loop
        // continuations each get their own row; failed turns count as !ok so
        // the success rate is meaningful).
        {
            let storage = state.storage.lock().unwrap();
            storage.record_usage(&crate::storage::UsageRecord {
                endpoint: &gp.endpoint,
                model: &gp.model,
                prompt_tokens: outcome.prompt_tokens,
                completion_tokens: outcome.completion_tokens,
                ttft_ms: outcome.ttft_ms,
                decode_tok_s: outcome.decode_tok_s,
                cost: outcome.cost,
                ok: !outcome.errored,
            });
        }

        if outcome.errored {
            errored = true;
            break;
        }

        let stored = combine_stored(&outcome.reasoning, &outcome.content);
        let mut assistant = Message::text(Role::Assistant, stored);
        if iteration == 0 {
            assistant.branch_index = first_assistant_branch_index;
        }
        if !outcome.tool_calls.is_empty() {
            assistant.tool_calls = Some(outcome.tool_calls.clone());
        }
        messages.push(assistant);
        if let Err(e) = {
            let storage = state.storage.lock().unwrap();
            storage.save_messages(conversation_id, &mut messages)
        } {
            on_event.send(ChatEvent::Error { message: e }).ok();
            errored = true;
            break;
        }
        final_message_id = messages.last().and_then(|m| m.id);

        if outcome.tool_calls.is_empty() || iteration + 1 >= MAX_TOOL_ITERATIONS {
            break;
        }

        for call in &outcome.tool_calls {
            on_event
                .send(ChatEvent::ToolCall {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.clone(),
                })
                .ok();

            // `use_skill` is handled in-process: return the named skill's body.
            if call.function.name == "use_skill" {
                let name = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                    .ok()
                    .and_then(|v| v.get("name").and_then(|n| n.as_str()).map(String::from))
                    .unwrap_or_default();
                let (result, is_error) = match skills.iter().find(|s| s.name == name) {
                    Some(s) => (s.body.clone(), false),
                    None => (format!("unknown skill: {name}"), true),
                };
                on_event
                    .send(ChatEvent::ToolResult {
                        id: call.id.clone(),
                        name: call.function.name.clone(),
                        result: result.clone(),
                        is_error,
                    })
                    .ok();
                messages.push(Message::tool_result(call.id.clone(), result));
                continue;
            }

            // MCP tools are routed to their server (honoring auto-approve and the
            // per-conversation allowlist; otherwise the normal approval flow).
            if crate::mcp::McpManager::is_mcp_tool(&call.function.name) {
                let pre_approved = state.mcp.auto_approve(&call.function.name).await
                    || state
                        .auto_approved
                        .lock()
                        .unwrap()
                        .contains(&(conversation_id, call.function.name.clone()));
                let approved = if pre_approved {
                    true
                } else {
                    request_approval(state, on_event, call, conversation_id, &cancel).await
                };
                let (result, is_error) = if !approved {
                    ("Tool call denied by user.".to_string(), true)
                } else {
                    let args = serde_json::from_str(&call.function.arguments)
                        .unwrap_or_else(|_| serde_json::json!({}));
                    state.mcp.call(&call.function.name, args).await
                };
                on_event
                    .send(ChatEvent::ToolResult {
                        id: call.id.clone(),
                        name: call.function.name.clone(),
                        result: result.clone(),
                        is_error,
                    })
                    .ok();
                messages.push(Message::tool_result(call.id.clone(), result));
                continue;
            }

            let def = tool_defs.iter().find(|d| d.name == call.function.name).cloned();
            let (result, is_error) = match def {
                None => (format!("unknown tool: {}", call.function.name), true),
                Some(def) => {
                    let pre_approved = def.safety == Safety::Auto
                        || state
                            .auto_approved
                            .lock()
                            .unwrap()
                            .contains(&(conversation_id, def.name.clone()));
                    let approved = if pre_approved {
                        true
                    } else {
                        request_approval(state, on_event, call, conversation_id, &cancel).await
                    };
                    if !approved {
                        ("Tool call denied by user.".to_string(), true)
                    } else {
                        execute_tool(def, call, working_dir.clone()).await
                    }
                }
            };

            on_event
                .send(ChatEvent::ToolResult {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    result: result.clone(),
                    is_error,
                })
                .ok();
            messages.push(Message::tool_result(call.id.clone(), result));
        }
        if let Err(e) = {
            let storage = state.storage.lock().unwrap();
            storage.save_messages(conversation_id, &mut messages)
        } {
            on_event.send(ChatEvent::Error { message: e }).ok();
            errored = true;
            break;
        }
    }

    state.cancel.lock().unwrap().remove(&conversation_id);
    (final_message_id, errored)
}

#[derive(Serialize)]
pub struct SiblingInfo {
    pub index: usize,
    pub total: usize,
    pub ids: Vec<i64>,
}

/// Sibling-branch info for the `◀ N/M ▶` navigator at a branch point.
#[tauri::command]
pub fn message_siblings(state: State<'_, AppState>, message_id: i64) -> SiblingInfo {
    let ids: Vec<i64> = state
        .storage
        .lock()
        .unwrap()
        .siblings_of(message_id)
        .into_iter()
        .map(|h| h.id)
        .collect();
    let index = ids.iter().position(|&id| id == message_id).unwrap_or(0);
    SiblingInfo {
        total: ids.len(),
        index,
        ids,
    }
}

/// Rebuild the active path suffix starting from `start_id` (the chosen sibling),
/// picking the newest child at each fork. The frontend splices this onto the
/// unchanged prefix before the branch point.
#[tauri::command]
pub fn walk_from(state: State<'_, AppState>, start_id: i64) -> Vec<MessageDto> {
    state
        .storage
        .lock()
        .unwrap()
        .walk_from(start_id)
        .iter()
        .map(message_to_dto)
        .collect()
}

/// Reasoning is persisted inline as a leading `<think>` block so it round-trips
/// and the frontend can collapse it on reload.
fn combine_stored(reasoning: &str, content: &str) -> String {
    if reasoning.is_empty() {
        content.to_string()
    } else {
        format!("<think>{reasoning}</think>\n{content}")
    }
}

struct StreamOutcome {
    content: String,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
    errored: bool,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    ttft_ms: Option<f64>,
    decode_tok_s: Option<f64>,
    cost: Option<f64>,
}

/// Run one streaming completion, forwarding tokens/reasoning/usage/metrics to
/// the frontend and returning the assembled result for the orchestration loop.
async fn stream_once(
    gp: &GenParams,
    wire: Vec<Message>,
    api_key: Option<String>,
    tools: Option<Vec<serde_json::Value>>,
    cancel: CancellationToken,
    on_event: &Channel<ChatEvent>,
) -> StreamOutcome {
    let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();
    let chat_params = ChatParams {
        base_url: gp.endpoint.clone(),
        model: gp.model.clone(),
        messages: wire,
        temperature: gp.temperature,
        max_tokens: if gp.use_max_tokens {
            gp.max_tokens
        } else {
            None
        },
        top_p: gp.top_p,
        frequency_penalty: gp.frequency_penalty,
        presence_penalty: gp.presence_penalty,
        stop_sequences: if gp.stop_sequences.is_empty() {
            None
        } else {
            Some(gp.stop_sequences.clone())
        },
        api_key,
        tools,
    };
    api::stream_chat(chat_params, tx, cancel);

    let start = Instant::now();
    let mut first_token_at: Option<Instant> = None;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut usage_prompt: Option<u32> = None;
    let mut usage_completion: Option<u32> = None;
    let mut usage_cost: Option<f64> = None;
    let mut errored = false;

    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Token(t) => {
                first_token_at.get_or_insert_with(Instant::now);
                content.push_str(&t);
                on_event.send(ChatEvent::Token { text: t }).ok();
            }
            StreamEvent::Reasoning(r) => {
                first_token_at.get_or_insert_with(Instant::now);
                reasoning.push_str(&r);
                on_event.send(ChatEvent::Reasoning { text: r }).ok();
            }
            StreamEvent::ToolCalls(calls) => {
                tool_calls = calls;
            }
            StreamEvent::UsageInfo(u) => {
                usage_prompt = u.prompt_tokens;
                usage_completion = u.completion_tokens;
                usage_cost = u.cost;
                on_event
                    .send(ChatEvent::Usage {
                        prompt_tokens: u.prompt_tokens,
                        completion_tokens: u.completion_tokens,
                        total_tokens: u.total_tokens,
                        cost: u.cost,
                    })
                    .ok();
            }
            StreamEvent::Error(msg) => {
                errored = true;
                on_event.send(ChatEvent::Error { message: msg }).ok();
            }
            StreamEvent::Done => break,
            StreamEvent::ModelsLoaded { .. } => {}
        }
    }

    let now = Instant::now();
    let duration_ms = now.duration_since(start).as_secs_f64() * 1000.0;
    let ttft_ms = first_token_at.map(|t| t.duration_since(start).as_secs_f64() * 1000.0);
    let decode_tok_s = match (usage_completion, first_token_at) {
        (Some(ct), Some(ft)) if ct > 0 => {
            let decode_s = now.duration_since(ft).as_secs_f64();
            (decode_s > 0.0).then(|| ct as f64 / decode_s)
        }
        _ => None,
    };
    let prefill_tok_s = match (usage_prompt, ttft_ms) {
        (Some(pt), Some(ttft)) if ttft > 0.0 => Some(pt as f64 / (ttft / 1000.0)),
        _ => None,
    };
    on_event
        .send(ChatEvent::RequestMetrics {
            ttft_ms,
            decode_tok_s,
            prefill_tok_s,
            duration_ms,
        })
        .ok();

    StreamOutcome {
        content,
        reasoning,
        tool_calls,
        errored,
        prompt_tokens: usage_prompt,
        completion_tokens: usage_completion,
        ttft_ms,
        decode_tok_s,
        cost: usage_cost,
    }
}

/// Park a oneshot keyed by the tool_call id and emit a `tool_approval` event;
/// resolves when `resolve_tool` fires or the stream is cancelled (→ denied).
async fn request_approval(
    state: &State<'_, AppState>,
    on_event: &Channel<ChatEvent>,
    call: &ToolCall,
    conversation_id: i64,
    cancel: &CancellationToken,
) -> bool {
    let (tx, rx) = oneshot::channel::<bool>();
    state.pending_approvals.lock().unwrap().insert(
        call.id.clone(),
        crate::state::PendingApproval {
            tx,
            conversation_id,
            tool: call.function.name.clone(),
        },
    );
    on_event
        .send(ChatEvent::ToolApproval {
            id: call.id.clone(),
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
        })
        .ok();
    tokio::select! {
        r = rx => r.unwrap_or(false),
        _ = cancel.cancelled() => {
            state.pending_approvals.lock().unwrap().remove(&call.id);
            false
        }
    }
}

/// Dispatch a tool call to its handler on a blocking thread (builtins and
/// shell tools are synchronous and can run for a while). Returns
/// `(output, is_error)`.
async fn execute_tool(def: ToolDef, call: &ToolCall, wd: PathBuf) -> (String, bool) {
    let args: serde_json::Value =
        serde_json::from_str(&call.function.arguments).unwrap_or_else(|_| serde_json::json!({}));
    let handler = def.handler.clone();
    let res = tokio::task::spawn_blocking(move || match handler {
        Handler::Builtin(b) => tools::run_builtin(&b.0, &args, &wd),
        Handler::Shell { shell } => tools::run_shell_tool(&shell, &args, &wd),
    })
    .await;
    match res {
        Ok(Ok(out)) => (out, false),
        Ok(Err(e)) => (e, true),
        Err(e) => (format!("tool task failed: {e}"), true),
    }
}

/// Generate a short title from the first exchange, once, in the background.
async fn maybe_auto_title(
    state: &State<'_, AppState>,
    conversation_id: i64,
    gp: &GenParams,
    user_text: &str,
) {
    let needs = {
        let storage = state.storage.lock().unwrap();
        storage.needs_auto_title(conversation_id)
    };
    if !needs {
        return;
    }
    let api_key = state.api_key_for(&gp.endpoint);
    let prompt = vec![
        Message::text(
            Role::System,
            "Give a concise 3-6 word title for this conversation. Reply with only the title, no quotes."
                .to_string(),
        ),
        Message::text(Role::User, user_text.to_string()),
    ];
    let title = api::complete_once(
        &gp.endpoint,
        &gp.model,
        &prompt,
        api_key.as_deref(),
        // Reasoning models (e.g. nemotron) spend tokens inside a <think> block
        // before emitting the title; a tight cap leaves the block unclosed and
        // strip_reasoning yields an empty string. Give them room to finish.
        Some(512),
        Some(0.3),
    )
    .await;
    if let Ok(raw) = title {
        let clean = crate::markdown::strip_reasoning(&raw);
        let clean = clean.trim().trim_matches('"').trim();
        if !clean.is_empty() {
            let storage = state.storage.lock().unwrap();
            storage.update_conversation_title(conversation_id, clean);
            storage.mark_auto_titled(conversation_id);
        }
    }
}
