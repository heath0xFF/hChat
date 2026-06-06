//! Tauri command layer. Replaces the egui `app.rs` orchestration: each command
//! locks `AppState` briefly for synchronous DB/config work and drops the guard
//! before any `.await`. Streaming chat forwards `StreamEvent`s from the core
//! `api::stream_chat` onto a frontend `Channel<ChatEvent>`.

use crate::api::{self, ChatParams, StreamEvent};
use crate::config::Config;
use crate::message::{ContentPart, Message, Role};
use crate::state::AppState;
use crate::storage::ConversationSettings;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tauri::State;
use tauri::ipc::Channel;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// ---------- DTOs ----------

#[derive(Serialize)]
pub struct ConversationDto {
    pub id: i64,
    pub title: String,
    pub pinned: bool,
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
}

#[derive(Serialize)]
pub struct ConversationData {
    pub messages: Vec<MessageDto>,
    pub settings: SettingsDto,
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

/// Everything the frontend sends to start a generation. The API key is *not*
/// here — it's resolved server-side from `config.saved_endpoints` so secrets
/// never round-trip through JS.
#[derive(Deserialize)]
pub struct SendParams {
    pub conversation_id: Option<i64>,
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
    pub user_text: String,
}

/// Streaming events pushed to the frontend over the per-request `Channel`.
#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// Conversation id, emitted first (covers the create-on-first-send case so
    /// the UI can adopt the new id immediately).
    Started { conversation_id: i64 },
    Token { text: String },
    Reasoning { text: String },
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

#[tauri::command]
pub fn list_conversations(state: State<'_, AppState>) -> Vec<ConversationDto> {
    state
        .storage
        .lock()
        .unwrap()
        .list_conversations()
        .into_iter()
        .map(|c| ConversationDto {
            id: c.id,
            title: c.title,
            pinned: c.pinned,
        })
        .collect()
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
    ConversationData { messages, settings }
}

#[tauri::command]
pub fn delete_conversation(state: State<'_, AppState>, id: i64) {
    state.storage.lock().unwrap().delete_conversation(id);
}

#[tauri::command]
pub fn rename_conversation(state: State<'_, AppState>, id: i64, title: String) {
    state
        .storage
        .lock()
        .unwrap()
        .update_conversation_title(id, &title);
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

// ---------- streaming chat ----------

#[tauri::command]
pub fn cancel_stream(state: State<'_, AppState>) {
    if let Some(token) = state.cancel.lock().unwrap().take() {
        token.cancel();
    }
}

#[tauri::command]
pub async fn send_message(
    state: State<'_, AppState>,
    params: SendParams,
    on_event: Channel<ChatEvent>,
) -> Result<i64, String> {
    // --- phase 1: synchronous setup (no await while locked) ---
    let (conversation_id, mut messages, wire_messages, settings_to_save, api_key) = {
        let storage = state.storage.lock().unwrap();

        let conversation_id = match params.conversation_id {
            Some(id) => id,
            None => storage.create_conversation("New chat")?,
        };

        // Existing active-path history (each carries its DB id).
        let mut messages = storage.load_messages(conversation_id);
        // Append the new user turn (unsaved → id None).
        messages.push(Message::text(Role::User, params.user_text.clone()));

        // Build the wire payload: optional system prompt + full history.
        let mut wire_messages = Vec::with_capacity(messages.len() + 1);
        if !params.system_prompt.trim().is_empty() {
            wire_messages.push(Message::text(Role::System, params.system_prompt.clone()));
        }
        wire_messages.extend(messages.iter().cloned());

        let settings_to_save = ConversationSettings {
            model: Some(params.model.clone()),
            system_prompt: Some(params.system_prompt.clone()),
            temperature: params.temperature,
            max_tokens: params.max_tokens,
            use_max_tokens: params.use_max_tokens,
            top_p: params.top_p,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            stop_sequences: params.stop_sequences.clone(),
            endpoint: Some(params.endpoint.clone()),
            working_dir: None,
        };

        let api_key = {
            let cfg = state.config.lock().unwrap();
            cfg.saved_endpoints
                .iter()
                .find(|e| e.url == params.endpoint)
                .and_then(|e| e.api_key.clone())
        };

        (conversation_id, messages, wire_messages, settings_to_save, api_key)
    };

    on_event.send(ChatEvent::Started { conversation_id }).ok();

    // Persist the user message + settings now so it survives a mid-stream quit.
    {
        let storage = state.storage.lock().unwrap();
        storage.save_messages(conversation_id, &mut messages)?;
        storage.save_conversation_settings(conversation_id, &settings_to_save);
    }

    // --- phase 2: stream ---
    let cancel = CancellationToken::new();
    *state.cancel.lock().unwrap() = Some(cancel.clone());

    let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();
    let chat_params = ChatParams {
        base_url: params.endpoint.clone(),
        model: params.model.clone(),
        messages: wire_messages,
        temperature: params.temperature,
        max_tokens: if params.use_max_tokens {
            params.max_tokens
        } else {
            None
        },
        top_p: params.top_p,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        stop_sequences: if params.stop_sequences.is_empty() {
            None
        } else {
            Some(params.stop_sequences.clone())
        },
        api_key,
        tools: None, // tool calling layered in next
    };
    api::stream_chat(chat_params, tx, cancel.clone());

    let start = Instant::now();
    let mut first_token_at: Option<Instant> = None;
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut usage_prompt: Option<u32> = None;
    let mut usage_completion: Option<u32> = None;
    let mut errored = false;

    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Token(t) => {
                if first_token_at.is_none() {
                    first_token_at = Some(Instant::now());
                }
                content.push_str(&t);
                on_event.send(ChatEvent::Token { text: t }).ok();
            }
            StreamEvent::Reasoning(r) => {
                if first_token_at.is_none() {
                    first_token_at = Some(Instant::now());
                }
                reasoning.push_str(&r);
                on_event.send(ChatEvent::Reasoning { text: r }).ok();
            }
            StreamEvent::ToolCalls(_) => {
                // Tool execution loop is added in the next milestone; ignore for now.
            }
            StreamEvent::UsageInfo(u) => {
                usage_prompt = u.prompt_tokens;
                usage_completion = u.completion_tokens;
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

    *state.cancel.lock().unwrap() = None;

    // --- phase 3: metrics + persist assistant turn ---
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

    let mut message_id = None;
    if !errored && !(content.is_empty() && reasoning.is_empty()) {
        // Persist reasoning inline as a leading <think> block so it round-trips
        // and the frontend can collapse it on reload.
        let stored = if reasoning.is_empty() {
            content.clone()
        } else {
            format!("<think>{reasoning}</think>\n{content}")
        };
        messages.push(Message::text(Role::Assistant, stored));
        {
            let storage = state.storage.lock().unwrap();
            storage.save_messages(conversation_id, &mut messages)?;
        }
        message_id = messages.last().and_then(|m| m.id);

        maybe_auto_title(&state, conversation_id, &params).await;
    }

    on_event.send(ChatEvent::Done { message_id }).ok();
    Ok(conversation_id)
}

/// Generate a short title from the first exchange, once, in the background.
async fn maybe_auto_title(state: &State<'_, AppState>, conversation_id: i64, params: &SendParams) {
    let needs = {
        let storage = state.storage.lock().unwrap();
        storage.needs_auto_title(conversation_id)
    };
    if !needs {
        return;
    }
    let api_key = state.api_key_for(&params.endpoint);
    let prompt = vec![
        Message::text(
            Role::System,
            "Give a concise 3-6 word title for this conversation. Reply with only the title, no quotes."
                .to_string(),
        ),
        Message::text(Role::User, params.user_text.clone()),
    ];
    let title = api::complete_once(
        &params.endpoint,
        &params.model,
        &prompt,
        api_key.as_deref(),
        Some(32),
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
