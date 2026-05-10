use crate::api::{self, ChatParams, StreamEvent, Usage};
use crate::commands::{self, Command, ParseResult};
use crate::config::{Config, Endpoint};
use crate::markdown::{self, Segment};
use crate::message::{ContentPart, ImageUrl, Message, Role};
use crate::storage::{ConversationSettings, Preset, Storage};
use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const STARTER_PROMPTS: &[(&str, &str)] = &[
    ("✏️ Brainstorm", "Help me brainstorm ideas for "),
    ("🐛 Debug", "Help me debug this issue:\n\n"),
    ("📚 Explain", "Explain like I'm an experienced engineer: "),
    ("🔧 Refactor", "Refactor this code for clarity:\n\n```\n\n```"),
];

pub struct ChatApp {
    messages: Vec<Message>,
    input: String,
    base_url: String,
    models: Vec<String>,
    selected_model: usize,
    streaming: bool,
    rx: Option<mpsc::UnboundedReceiver<StreamEvent>>,
    /// Separate channel for model fetching so it doesn't conflict with chat streaming
    models_rx: Option<mpsc::UnboundedReceiver<StreamEvent>>,
    cancel_token: Option<CancellationToken>,
    runtime: tokio::runtime::Runtime,
    models_loading: bool,
    error: Option<String>,
    system_prompt: String,
    temperature: f32,
    max_tokens: u32,
    use_max_tokens: bool,
    dark_mode: bool,
    commonmark_cache: CommonMarkCache,
    editing_message: Option<usize>,
    edit_buffer: String,
    show_settings: bool,
    // Presets — saved settings bundles. Cache is loaded lazily.
    show_presets: bool,
    new_preset_name: String,
    presets_cache: Vec<Preset>,
    // Persistence
    storage: Storage,
    current_conversation_id: Option<i64>,
    /// (id, title, pinned). Pinned conversations sort to the top.
    conversation_list: Vec<(i64, String, bool)>,
    /// Hash of the live settings that we last persisted to the current conv.
    /// When the live hash differs, we know to flush per-conv settings.
    settings_signature: u64,
    /// When loading a conversation whose endpoint differs from the current one,
    /// we switch the endpoint and re-fetch models. The conv's stored model name
    /// goes here so we can restore it once the new model list arrives.
    pending_model_after_models_load: Option<String>,
    /// Channel for one-shot auto-title responses. The payload is
    /// `(conversation_id, Result<title, error>)` — the conv id is on the
    /// payload (not pulled from `current_conversation_id`) so a switch in
    /// the meantime doesn't cross-contaminate titles.
    title_rx: Option<mpsc::UnboundedReceiver<(i64, Result<String, String>)>>,
    // Draft persistence — debounce input changes and write to the conv row
    // 500ms after the user stops typing. `last_seen_input` detects per-frame
    // edits; `persisted_draft` is what we last wrote (so we skip a write
    // when nothing's actually changed since the last persist).
    last_seen_input: String,
    persisted_draft: String,
    draft_dirty_since: Option<std::time::Instant>,
    show_sidebar: bool,
    show_context_sidebar: bool,
    // Search
    search_query: String,
    search_results: Vec<(i64, String, String)>,
    show_search: bool,
    // Endpoints
    saved_endpoints: Vec<Endpoint>,
    show_endpoints: bool,
    new_endpoint: String,
    new_endpoint_key: String,
    // Token usage
    last_usage: Option<Usage>,
    session_cost: f64,
    // Rename
    renaming_conversation: Option<i64>,
    rename_buffer: String,
    // Config
    config: Config,
    // Sampling
    top_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    stop_sequences: Vec<String>,
    // Find-in-conversation
    show_find: bool,
    find_query: String,
    /// Last query we already scrolled the first match into view for. When this
    /// matches `find_query`, suppress further `scroll_to_me` calls so the user
    /// can scroll past the first match without being yanked back every frame.
    last_scrolled_find_query: Option<String>,
    // Token-estimate cache. Key is a cheap signature of (system + msgs + input);
    // recompute only when it changes. Without this we re-encode the entire
    // conversation through tiktoken every frame.
    token_cache: Option<(u64, usize)>,
    // Reasoning streaming state — true when we've emitted <think> for the
    // current assistant message but haven't closed it yet.
    reasoning_open: bool,
    // Toast — short-lived feedback (e.g. unknown slash command).
    toast: Option<(String, std::time::Instant)>,
    /// Image attachments staged for the next sent message. Each entry is
    /// `(display_name, ContentPart::ImageUrl)`. Cleared after send_message
    /// builds the user message.
    pending_attachments: Vec<(String, ContentPart)>,
    /// Working directory passed to tool handlers in this conversation.
    /// Per-conversation override; defaults to the user's home dir on a
    /// fresh chat. Persists alongside the rest of `ConversationSettings`.
    working_dir: String,
    /// Tools loaded from `~/.config/hchat/tools/` at startup. Sent with
    /// every chat request so the model knows what it can call. Reload via
    /// re-launch (Phase 5a; a /tools-reload command can come later).
    loaded_tools: Vec<crate::tools::ToolDef>,
    /// Tool calls queued from the most recent assistant turn that haven't
    /// been executed yet. Drained left-to-right by `process_pending_tool_calls`.
    pending_tool_calls: Vec<crate::message::ToolCall>,
    /// One tool call awaiting user click on the approval card. Set when
    /// the queue head is `safety = "confirm"` and the user hasn't
    /// pre-approved this tool name. The execution loop is paused while
    /// this is `Some`.
    pending_approval: Option<crate::message::ToolCall>,
    /// Tool names the user pre-approved for the rest of this session via
    /// the "Allow all in this conv" button. Not persisted — fresh per
    /// app launch; per the design, doesn't mutate the TOML safety field.
    auto_approved_tools: std::collections::HashSet<String>,
    /// How many tool-then-restream cycles we've completed for the current
    /// user turn. Hard cap (see `MAX_TOOL_ITERATIONS`) prevents a runaway
    /// model from looping forever. Reset on the next `send_message`.
    tool_iterations: u32,
    /// Per-frame cache of (current_branch_position, total_siblings) per
    /// message — feeds the ◀ N/M ▶ navigator. Without this we'd hit the
    /// DB twice per message per frame just to render arrows that change
    /// almost never. Invalidated by a cheap signature compare.
    sibling_info_cache: Vec<Option<(usize, usize)>>,
    sibling_info_signature: u64,
}

impl ChatApp {
    pub fn new(cc: &eframe::CreationContext<'_>, config: Config) -> Self {
        configure_fonts(&cc.egui_ctx, &config);
        apply_theme(&cc.egui_ctx, config.dark_mode);
        cc.egui_ctx.set_zoom_factor(config.ui_scale);

        let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        let storage = Storage::new();
        let conversation_list = storage
            .list_conversations()
            .into_iter()
            .map(|c| (c.id, c.title, c.pinned))
            .collect();

        let mut app = Self {
            messages: Vec::new(),
            input: String::new(),
            base_url: config.default_endpoint.clone(),
            models: Vec::new(),
            selected_model: 0,
            streaming: false,
            rx: None,
            models_rx: None,
            cancel_token: None,
            runtime,
            models_loading: false,
            error: None,
            system_prompt: config.system_prompt.clone(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            use_max_tokens: config.use_max_tokens,
            dark_mode: config.dark_mode,
            commonmark_cache: CommonMarkCache::default(),
            editing_message: None,
            edit_buffer: String::new(),
            show_settings: false,
            show_presets: false,
            new_preset_name: String::new(),
            presets_cache: Vec::new(),
            storage,
            current_conversation_id: None,
            conversation_list,
            settings_signature: 0,
            pending_model_after_models_load: None,
            title_rx: None,
            last_seen_input: String::new(),
            persisted_draft: String::new(),
            draft_dirty_since: None,
            show_sidebar: true,
            show_context_sidebar: true,
            search_query: String::new(),
            search_results: Vec::new(),
            show_search: false,
            saved_endpoints: config.saved_endpoints.clone(),
            show_endpoints: false,
            new_endpoint: String::new(),
            new_endpoint_key: String::new(),
            last_usage: None,
            session_cost: 0.0,
            renaming_conversation: None,
            rename_buffer: String::new(),
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            stop_sequences: config.stop_sequences.clone(),
            show_find: false,
            find_query: String::new(),
            last_scrolled_find_query: None,
            token_cache: None,
            reasoning_open: false,
            toast: None,
            pending_attachments: Vec::new(),
            working_dir: default_working_dir(),
            loaded_tools: {
                let dir = crate::tools::user_tools_dir();
                crate::tools::seed_defaults_if_empty(&dir);
                crate::tools::load_from_dir(&dir)
            },
            pending_tool_calls: Vec::new(),
            pending_approval: None,
            auto_approved_tools: std::collections::HashSet::new(),
            tool_iterations: 0,
            sibling_info_cache: Vec::new(),
            sibling_info_signature: 0,
            config,
        };

        app.fetch_models();
        app
    }

    fn show_toast(&mut self, msg: impl Into<String>) {
        self.toast = Some((msg.into(), std::time::Instant::now()));
    }

    fn fetch_models(&mut self) {
        // Always start a fresh fetch — replacing `models_rx` orphans any
        // in-flight task's sender, so its result silently disappears (the
        // dropped receiver causes `tx.send` to return Err). This is what we
        // want when the user switches endpoint mid-fetch: the old endpoint's
        // result must not land as the new endpoint's model list.
        let base_url = self.base_url.clone();
        let api_key = self.current_api_key().map(|s| s.to_string());
        let (tx, rx) = mpsc::unbounded_channel();

        let url_for_event = base_url.clone();
        self.runtime.spawn(async move {
            match api::fetch_models(&base_url, api_key.as_deref()).await {
                Ok(models) => {
                    let _ = tx.send(StreamEvent::ModelsLoaded {
                        url: url_for_event,
                        models,
                    });
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(e));
                }
            }
            let _ = tx.send(StreamEvent::Done);
        });

        self.models_loading = true;
        self.models_rx = Some(rx);
    }

    fn selected_model_name(&self) -> &str {
        self.models
            .get(self.selected_model)
            .map(|s| s.as_str())
            .unwrap_or("(no models)")
    }

    fn current_api_key(&self) -> Option<&str> {
        self.saved_endpoints
            .iter()
            .find(|ep| ep.url == self.base_url)
            .and_then(|ep| ep.api_key.as_deref())
    }

    fn new_conversation(&mut self) {
        if self.streaming {
            self.stop_streaming();
        }
        self.save_current();
        self.clear_tool_execution_state();
        self.messages.clear();
        self.current_conversation_id = None;
        self.error = None;
        self.last_usage = None;
        self.session_cost = 0.0;
        self.input.clear();
        // Critical: clear edit state. Otherwise Ctrl+N during an in-progress
        // edit leaves `editing_message = Some(N)` pointing into the new
        // (smaller) conversation, and the Nth message of the next chat will
        // render in edit mode with the prior chat's edit_buffer contents.
        self.editing_message = None;
        self.edit_buffer.clear();
        // Reset draft tracking — there's no conv to persist into yet.
        self.last_seen_input.clear();
        self.persisted_draft.clear();
        self.draft_dirty_since = None;
        self.pending_attachments.clear();
        self.reset_settings_to_defaults();
    }

    fn save_current(&mut self) {
        if self.messages.is_empty() {
            return;
        }

        let save_result: Result<(), String> = match self.current_conversation_id {
            Some(id) => self.storage.save_messages(id, &mut self.messages),
            None => {
                let title = self
                    .messages
                    .iter()
                    .find(|m| m.role == Role::User)
                    .map(|m| {
                        let s = m.text_str();
                        let t = s.trim();
                        if t.len() > 50 {
                            // Find a char boundary for truncation
                            let end = t
                                .char_indices()
                                .take_while(|(i, _)| *i <= 47)
                                .last()
                                .map(|(i, c)| i + c.len_utf8())
                                .unwrap_or(47);
                            format!("{}...", &t[..end])
                        } else {
                            t.to_string()
                        }
                    })
                    .unwrap_or_else(|| "New chat".to_string());

                match self.storage.create_conversation(&title) {
                    Ok(id) => {
                        let r = self.storage.save_messages(id, &mut self.messages);
                        if r.is_ok() {
                            self.current_conversation_id = Some(id);
                            // Persist the live settings as the conv's initial
                            // snapshot so reloading later restores them.
                            let snap = self.current_settings_snapshot();
                            self.storage.save_conversation_settings(id, &snap);
                            self.settings_signature = settings_signature(&snap);
                        }
                        r
                    }
                    Err(e) => Err(e),
                }
            }
        };

        if let Err(e) = save_result {
            // Surface DB errors so the user knows their conversation didn't
            // persist. Previously these were silently swallowed.
            self.error = Some(format!("Save failed: {e}"));
        }

        // Auto-title trigger: only fire once per conversation. Requires at
        // least one user + one non-empty assistant message in scope (the
        // initial save from the very first send_message would have only the
        // user message — that path doesn't generate a title).
        if let Some(id) = self.current_conversation_id
            && self.storage.needs_auto_title(id)
            && self
                .messages
                .iter()
                .any(|m| m.role == Role::Assistant && !m.is_empty_content())
        {
            self.request_auto_title(id);
        }

        self.refresh_conversation_list();
    }

    fn load_conversation(&mut self, id: i64) {
        if self.streaming {
            self.stop_streaming();
        }
        self.save_current();
        self.clear_tool_execution_state();
        self.messages = self.storage.load_messages(id);
        self.current_conversation_id = Some(id);
        self.error = None;
        self.last_usage = None;
        self.session_cost = 0.0;
        self.editing_message = None;
        self.edit_buffer.clear();
        let stored = self.storage.load_conversation_settings(id);
        self.apply_settings(stored);
        self.input = self.storage.load_draft(id).unwrap_or_default();
        // Seed the draft tracker so we don't immediately re-persist what we
        // just loaded.
        self.last_seen_input = self.input.clone();
        self.persisted_draft = self.input.clone();
        self.draft_dirty_since = None;
    }

    fn delete_conversation(&mut self, id: i64) {
        self.storage.delete_conversation(id);
        if self.current_conversation_id == Some(id) {
            if self.streaming {
                self.stop_streaming();
            }
            self.clear_tool_execution_state();
            self.messages.clear();
            self.current_conversation_id = None;
            self.editing_message = None;
            self.edit_buffer.clear();
        }
        self.refresh_conversation_list();
    }

    fn refresh_conversation_list(&mut self) {
        self.conversation_list = self
            .storage
            .list_conversations()
            .into_iter()
            .map(|c| (c.id, c.title, c.pinned))
            .collect();
    }

    /// Snapshot the live per-conversation state into a `ConversationSettings`.
    /// `model` is the currently selected model name (looked up via the index).
    fn current_settings_snapshot(&self) -> ConversationSettings {
        ConversationSettings {
            model: self.models.get(self.selected_model).cloned(),
            system_prompt: if self.system_prompt.is_empty() {
                None
            } else {
                Some(self.system_prompt.clone())
            },
            temperature: Some(self.temperature),
            max_tokens: Some(self.max_tokens),
            use_max_tokens: self.use_max_tokens,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            stop_sequences: self.stop_sequences.clone(),
            endpoint: Some(self.base_url.clone()),
            working_dir: Some(self.working_dir.clone()),
        }
    }

    /// Load a `ConversationSettings` into the live fields. `None` fields fall
    /// back to the global config defaults so a conv created before a setting
    /// existed picks up sensible values.
    fn apply_settings(&mut self, s: ConversationSettings) {
        // If the conv stored a different endpoint, switch to it and queue
        // model fetch. Subsequent ModelsLoaded restores the model selection
        // by name via `pending_model_after_models_load`.
        if let Some(ep) = s.endpoint
            && ep != self.base_url
        {
            self.base_url = ep;
            self.fetch_models();
        }

        if let Some(name) = s.model {
            if let Some(idx) = self.models.iter().position(|m| m == &name) {
                self.selected_model = idx;
            } else {
                // Models not loaded yet (or this model isn't currently
                // available). Stash for later.
                self.pending_model_after_models_load = Some(name);
            }
        }

        self.system_prompt = s.system_prompt.unwrap_or_else(|| self.config.system_prompt.clone());
        self.temperature = s.temperature.unwrap_or(self.config.temperature);
        self.max_tokens = s.max_tokens.unwrap_or(self.config.max_tokens);
        self.use_max_tokens = s.use_max_tokens;
        self.top_p = s.top_p.or(self.config.top_p);
        self.frequency_penalty = s.frequency_penalty.or(self.config.frequency_penalty);
        self.presence_penalty = s.presence_penalty.or(self.config.presence_penalty);
        self.stop_sequences = if s.stop_sequences.is_empty() {
            self.config.stop_sequences.clone()
        } else {
            s.stop_sequences
        };
        self.working_dir = s.working_dir.unwrap_or_else(default_working_dir);
        // Mark settings as freshly loaded so the dirty-check below doesn't
        // immediately re-save what we just read.
        self.settings_signature = settings_signature(&self.current_settings_snapshot());
    }

    /// Reset the live settings to the global config defaults. Called when
    /// starting a brand-new conversation.
    fn reset_settings_to_defaults(&mut self) {
        self.system_prompt = self.config.system_prompt.clone();
        self.temperature = self.config.temperature;
        self.max_tokens = self.config.max_tokens;
        self.use_max_tokens = self.config.use_max_tokens;
        self.top_p = self.config.top_p;
        self.frequency_penalty = self.config.frequency_penalty;
        self.presence_penalty = self.config.presence_penalty;
        self.stop_sequences = self.config.stop_sequences.clone();
        self.working_dir = default_working_dir();
        self.settings_signature = settings_signature(&self.current_settings_snapshot());
    }

    /// Persist the live settings to the current conversation row if they've
    /// changed since the last persist. Cheap to call every frame: the hash
    /// compare short-circuits when nothing's moved.
    fn persist_settings_if_dirty(&mut self) {
        let Some(id) = self.current_conversation_id else {
            return;
        };
        let snap = self.current_settings_snapshot();
        let sig = settings_signature(&snap);
        if sig != self.settings_signature {
            self.storage.save_conversation_settings(id, &snap);
            self.settings_signature = sig;
        }
    }

    /// Process files dropped onto the window. Image files are encoded as
    /// base64 data URLs and staged for the next message; text files are
    /// inlined into the input as fenced code. Anything else gets a toast.
    fn handle_dropped_files(&mut self, files: Vec<egui::DroppedFile>) {
        const MAX_TEXT_BYTES: usize = 256 * 1024;
        const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
        for f in files {
            // egui delivers either a path (native) or in-memory bytes (web).
            // We only target native, but handle both shapes defensively.
            let (name, bytes): (String, Vec<u8>) = if let Some(path) = &f.path {
                let n = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.to_string_lossy().to_string());
                match std::fs::read(path) {
                    Ok(b) => (n, b),
                    Err(e) => {
                        self.show_toast(format!("Failed to read {n}: {e}"));
                        continue;
                    }
                }
            } else if !f.bytes.as_ref().map(|b| b.is_empty()).unwrap_or(true) {
                let n = if f.name.is_empty() {
                    "dropped".to_string()
                } else {
                    f.name.clone()
                };
                let b = f.bytes.as_ref().map(|b| b.to_vec()).unwrap_or_default();
                (n, b)
            } else {
                continue;
            };

            let lower = name.to_ascii_lowercase();
            let ext = std::path::Path::new(&lower)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            if let Some(mime) = image_mime_for_ext(ext) {
                if bytes.len() > MAX_IMAGE_BYTES {
                    self.show_toast(format!(
                        "{name}: image larger than {} MB — skipped",
                        MAX_IMAGE_BYTES / (1024 * 1024)
                    ));
                    continue;
                }
                use base64::Engine as _;
                let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                let data_url = format!("data:{mime};base64,{encoded}");
                self.pending_attachments.push((
                    name,
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: data_url,
                            detail: None,
                        },
                    },
                ));
            } else if is_text_extension(ext) {
                if bytes.len() > MAX_TEXT_BYTES {
                    self.show_toast(format!(
                        "{name}: text larger than {} KB — skipped",
                        MAX_TEXT_BYTES / 1024
                    ));
                    continue;
                }
                let text = match String::from_utf8(bytes) {
                    Ok(t) => t,
                    Err(_) => {
                        self.show_toast(format!("{name}: not valid UTF-8 — skipped"));
                        continue;
                    }
                };
                let lang = lang_for_ext(ext);
                if !self.input.is_empty() && !self.input.ends_with('\n') {
                    self.input.push('\n');
                }
                self.input
                    .push_str(&format!("```{lang}\n// {name}\n{text}\n```\n"));
            } else {
                self.show_toast(format!("{name}: unsupported file type"));
            }
        }
    }

    fn show_presets_panel(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label(egui::RichText::new("Presets").strong());
        ui.add_space(4.0);

        // Save current settings as a new preset.
        ui.horizontal(|ui| {
            ui.label("New preset:");
            ui.add(
                egui::TextEdit::singleline(&mut self.new_preset_name)
                    .hint_text("Preset name")
                    .desired_width(200.0),
            );
            let can_save = !self.new_preset_name.trim().is_empty();
            if ui
                .add_enabled(can_save, egui::Button::new("Save current"))
                .on_hover_text("Snapshot the current chat's settings as a new preset")
                .clicked()
            {
                let snap = self.current_settings_snapshot();
                let name = self.new_preset_name.trim().to_string();
                if self.storage.create_preset(&name, &snap).is_ok() {
                    self.new_preset_name.clear();
                    self.presets_cache = self.storage.list_presets();
                }
            }
        });

        ui.add_space(4.0);
        if self.presets_cache.is_empty() {
            ui.label(
                egui::RichText::new("No presets yet. Save the current chat's settings above.")
                    .italics()
                    .weak(),
            );
        } else {
            // Two-pass: render rows and collect the chosen action, then act
            // (avoids overlapping borrows on `self`).
            enum PresetAction {
                Apply(i64),
                NewFrom(i64),
                Delete(i64),
            }
            let mut action: Option<PresetAction> = None;
            for preset in &self.presets_cache {
                ui.horizontal(|ui| {
                    ui.label(&preset.name);
                    if let Some(model) = &preset.settings.model {
                        ui.weak(format!("({model})"));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("🗑")
                            .on_hover_text("Delete preset")
                            .clicked()
                        {
                            action = Some(PresetAction::Delete(preset.id));
                        }
                        if ui
                            .small_button("New chat")
                            .on_hover_text("Open a new conversation seeded with this preset")
                            .clicked()
                        {
                            action = Some(PresetAction::NewFrom(preset.id));
                        }
                        if ui
                            .small_button("Apply")
                            .on_hover_text("Apply this preset to the current chat")
                            .clicked()
                        {
                            action = Some(PresetAction::Apply(preset.id));
                        }
                    });
                });
            }
            if let Some(act) = action {
                let preset = match act {
                    PresetAction::Apply(id)
                    | PresetAction::NewFrom(id)
                    | PresetAction::Delete(id) => self
                        .presets_cache
                        .iter()
                        .find(|p| p.id == id)
                        .cloned(),
                };
                match act {
                    PresetAction::Apply(_) => {
                        if let Some(p) = preset {
                            self.apply_settings(p.settings);
                        }
                    }
                    PresetAction::NewFrom(_) => {
                        if let Some(p) = preset {
                            self.new_conversation();
                            self.apply_settings(p.settings);
                        }
                    }
                    PresetAction::Delete(id) => {
                        self.storage.delete_preset(id);
                        self.presets_cache = self.storage.list_presets();
                    }
                }
            }
        }
    }

    /// Fire a one-shot title-generation request for the given conversation.
    /// Result lands on `title_rx` later; `process_events` applies it. Marks
    /// the conv as `auto_titled` immediately on dispatch so a second send
    /// before the response lands doesn't fire a duplicate request.
    fn request_auto_title(&mut self, conv_id: i64) {
        let messages: Vec<Message> = self
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .take(4) // first user + first assistant pair, plus one more in case
            .cloned()
            .collect();
        if messages.is_empty() {
            return;
        }
        // Claim the slot now. If the call fails the title stays as the
        // truncated first user message — the user can rename manually.
        self.storage.mark_auto_titled(conv_id);
        // Append the title-instruction as the final user turn.
        let mut ctx_messages = messages;
        ctx_messages.push(Message::text(
            Role::User,
            "Provide a title for this conversation in 6 words or fewer. \
             Reply with only the title text — no quotes, no punctuation at \
             the end, no preamble."
                .to_string(),
        ));

        let model = match self.models.get(self.selected_model) {
            Some(m) => m.clone(),
            None => return,
        };
        let base_url = self.base_url.clone();
        let api_key = self.current_api_key().map(|s| s.to_string());

        let (tx, rx) = mpsc::unbounded_channel();
        self.title_rx = Some(rx);

        self.runtime.spawn(async move {
            let res = api::complete_once(
                &base_url,
                &model,
                &ctx_messages,
                api_key.as_deref(),
                Some(32),
                Some(0.3),
            )
            .await;
            let _ = tx.send((conv_id, res));
        });
    }

    /// Debounced draft persist: writes `input` to `conversations.draft` 500ms
    /// after the last keystroke. Returns the soonest the caller should request
    /// a repaint at — pass through to `ctx.request_repaint_after` so the timer
    /// fires even if egui is otherwise idle.
    fn persist_draft_if_idle(&mut self) -> Option<std::time::Duration> {
        const DEBOUNCE: std::time::Duration = std::time::Duration::from_millis(500);
        if self.input != self.last_seen_input {
            self.last_seen_input = self.input.clone();
            self.draft_dirty_since = Some(std::time::Instant::now());
        }
        let t = self.draft_dirty_since?;
        let elapsed = t.elapsed();
        if elapsed >= DEBOUNCE {
            if self.input != self.persisted_draft
                && let Some(id) = self.current_conversation_id
            {
                self.storage.save_draft(id, &self.input);
                self.persisted_draft = self.input.clone();
            }
            self.draft_dirty_since = None;
            None
        } else {
            Some(DEBOUNCE - elapsed)
        }
    }

    fn send_message(&mut self) {
        let raw = self.input.trim().to_string();
        // Allow send when there's text OR at least one attachment (image-only
        // turns are valid for vision models). Block sends while a tool call
        // is parked for approval or queued — letting the user send a new
        // turn would orphan the parked call's `tool_call_id`, breaking the
        // assistant↔tool message chain on the next API request.
        if (raw.is_empty() && self.pending_attachments.is_empty())
            || self.streaming
            || self.pending_approval.is_some()
            || !self.pending_tool_calls.is_empty()
        {
            return;
        }

        // Slash commands intercept before model availability check, so /help
        // and /clear work even when no models are loaded. Attachments are
        // only sent with a real chat turn — a slash command shouldn't
        // accidentally consume them.
        if !raw.is_empty() {
            match commands::parse(&raw) {
                ParseResult::Command(cmd) => {
                    self.input.clear();
                    self.error = None;
                    self.dispatch_command(cmd);
                    return;
                }
                ParseResult::Unknown(verb) => {
                    self.show_toast(format!("Unknown command: /{verb} — try /help"));
                    return;
                }
                ParseResult::BadArgs { verb: _, reason } => {
                    self.show_toast(reason);
                    return;
                }
                ParseResult::NotACommand => {}
            }
        }

        if self.models.is_empty() {
            self.error = Some("No models available".to_string());
            return;
        }

        self.input.clear();
        self.error = None;

        // Build the user message: text part (if any) followed by image parts.
        let mut parts: Vec<ContentPart> = Vec::new();
        if !raw.is_empty() {
            parts.push(ContentPart::Text { text: raw });
        }
        for (_, part) in self.pending_attachments.drain(..) {
            parts.push(part);
        }
        self.messages.push(Message::from_parts(Role::User, parts));
        // New user turn — reset the tool iteration budget. The previous
        // assistant turn may have used some of it; that turn is over.
        self.tool_iterations = 0;
        self.start_streaming();
    }

    fn dispatch_command(&mut self, cmd: Command) {
        match cmd {
            Command::Model(needle) => {
                let needle_lower = needle.to_ascii_lowercase();
                let hit = self
                    .models
                    .iter()
                    .position(|m| m.to_ascii_lowercase().contains(&needle_lower));
                match hit {
                    Some(i) => {
                        self.selected_model = i;
                        let name = self.models[i].clone();
                        self.show_toast(format!("Model: {name}"));
                    }
                    None => self.show_toast(format!("No model matches '{needle}'")),
                }
            }
            Command::Temperature(v) => {
                let clamped = v.clamp(0.0, 2.0);
                self.temperature = clamped;
                self.show_toast(format!("Temperature: {clamped:.2}"));
            }
            Command::System(text) => {
                self.system_prompt = text.clone();
                let preview = if text.is_empty() {
                    "(cleared)".to_string()
                } else {
                    truncate_chars(&text, 40)
                };
                self.show_toast(format!("System prompt: {preview}"));
            }
            Command::Clear => self.new_conversation(),
            Command::Copy => {
                if let Some(last) = self
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == Role::Assistant && !m.is_empty_content())
                {
                    self.copy_to_clipboard(&last.text_str());
                    self.show_toast("Copied last reply");
                } else {
                    self.show_toast("No reply to copy yet");
                }
            }
            Command::Help => self.show_toast(commands::help_text()),
        }
    }

    fn start_streaming(&mut self) {
        if self.models.is_empty() {
            self.error = Some("No models available".to_string());
            return;
        }

        // If the caller (regenerate, edit_and_resend) already prepared an
        // empty assistant placeholder with the right parent_id /
        // branch_index, don't double-push. Otherwise add a fresh one;
        // save_messages will auto-link its parent_id to the preceding user
        // message in the slice.
        let needs_placeholder = match self.messages.last() {
            Some(m) => m.role != Role::Assistant || !m.is_empty_content(),
            None => true,
        };
        if needs_placeholder {
            self.messages
                .push(Message::text(Role::Assistant, String::new()));
        }
        self.reasoning_open = false;

        let (tx, rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        self.rx = Some(rx);
        self.cancel_token = Some(cancel.clone());
        self.streaming = true;
        self.last_usage = None;

        // Clamp selected_model to valid range (saturating_sub avoids underflow on empty vec)
        self.selected_model = self.selected_model.min(self.models.len().saturating_sub(1));
        let model = self.models[self.selected_model].clone();
        let base_url = self.base_url.clone();

        let mut messages: Vec<Message> = Vec::new();
        if !self.system_prompt.trim().is_empty() {
            messages.push(Message::text(Role::System, self.system_prompt.clone()));
        }
        // All messages except the empty assistant one we just added
        messages.extend_from_slice(&self.messages[..self.messages.len() - 1]);

        let temperature = Some(self.temperature);
        let max_tokens = if self.use_max_tokens {
            Some(self.max_tokens)
        } else {
            None
        };

        let api_key = self.current_api_key().map(|s| s.to_string());

        let stop = if self.stop_sequences.is_empty() {
            None
        } else {
            Some(self.stop_sequences.clone())
        };

        let tools = if self.loaded_tools.is_empty() {
            None
        } else {
            Some(crate::tools::to_api_shape(&self.loaded_tools))
        };
        let params = ChatParams {
            base_url,
            model,
            messages,
            temperature,
            max_tokens,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            stop_sequences: stop,
            api_key,
            tools,
        };
        self.runtime
            .spawn(async move { api::stream_chat(params, tx, cancel) });
    }

    fn stop_streaming(&mut self) {
        if let Some(cancel) = self.cancel_token.take() {
            cancel.cancel();
        }
        self.streaming = false;
        self.rx = None;
        if let Some(last) = self.messages.last()
            && last.role == Role::Assistant
            && last.is_empty_content()
        {
            self.messages.pop();
        }
    }

    fn regenerate(&mut self) {
        if self.streaming || self.models.is_empty() {
            return;
        }
        // Treat regenerate as a fresh attempt: reset the tool iteration
        // budget so the new branch isn't capped by what the previous
        // assistant turn burned through.
        self.tool_iterations = 0;
        let Some(last) = self.messages.last() else {
            return;
        };
        if last.role != Role::Assistant {
            return;
        }
        let parent_id = last.parent_id;

        // Make sure the conversation has a parent_id chain. Legacy 0.7.0
        // conversations didn't, so the first regenerate triggers a one-shot
        // backfill — the resulting siblings then have a real ancestor to
        // attach to.
        if let Some(conv_id) = self.current_conversation_id {
            let _ = self.storage.backfill_parent_ids(conv_id);
            // The backfill rewrote rows in-place; refresh the in-memory
            // copy so subsequent navigation/saves see the new parent_ids.
            // (The active path doesn't change — just the linkage metadata.)
            self.messages = self.storage.load_messages(conv_id);
        }

        // After the (possible) reload, find the assistant we want to
        // regenerate and the parent it's attached to.
        let (parent_for_new, branch_index) = if let Some(conv_id) = self.current_conversation_id {
            let p = self
                .messages
                .last()
                .and_then(|m| m.parent_id)
                .or(parent_id);
            let bi = self.storage.next_branch_index(conv_id, p);
            (p, bi)
        } else {
            // Unsaved conversation — no DB to consult. Pop the assistant
            // and let start_streaming push a fresh one as before. Branch
            // tracking only kicks in once the conv is persisted.
            self.messages.pop();
            self.start_streaming();
            return;
        };

        // Pop the active assistant from the in-memory path; the row stays
        // in the DB so the user can navigate back to it via the sibling
        // arrows.
        self.messages.pop();
        let mut placeholder = Message::text(Role::Assistant, String::new());
        placeholder.parent_id = parent_for_new;
        placeholder.branch_index = branch_index;
        self.messages.push(placeholder);
        self.start_streaming();
    }

    fn edit_and_resend(&mut self, index: usize) {
        if self.streaming || index >= self.messages.len() {
            return;
        }
        // Edit creates a fresh user-driven turn — reset the iteration
        // budget so the new branch starts with the full cap available.
        self.tool_iterations = 0;
        let content = self.edit_buffer.trim().to_string();
        if content.is_empty() {
            return;
        }

        // Preserve any non-text parts (image attachments) on the original
        // message. The new sibling user message inherits them so editing a
        // multimodal turn doesn't silently drop the images.
        let preserved: Vec<ContentPart> = self.messages[index]
            .content
            .iter()
            .filter(|p| !matches!(p, ContentPart::Text { .. }))
            .cloned()
            .collect();
        let mut new_parts = vec![ContentPart::Text { text: content }];
        new_parts.extend(preserved);

        let original_parent = self.messages[index].parent_id;

        // Backfill legacy conversations so the new sibling slots into a
        // coherent parent chain. Then reload to pick up the linkage.
        if let Some(conv_id) = self.current_conversation_id {
            let _ = self.storage.backfill_parent_ids(conv_id);
            self.messages = self.storage.load_messages(conv_id);
            // If reload changed indices (it shouldn't for a clean backfill),
            // bail safely rather than corrupt state.
            if index >= self.messages.len() {
                self.editing_message = None;
                self.edit_buffer.clear();
                return;
            }
        }

        let parent_for_sibling = self
            .messages
            .get(index)
            .and_then(|m| m.parent_id)
            .or(original_parent);

        let branch_index = if let Some(conv_id) = self.current_conversation_id {
            self.storage
                .next_branch_index(conv_id, parent_for_sibling)
        } else {
            0
        };

        // Truncate the in-memory active path at `index` (the rows beyond
        // stay on disk as a sibling branch). Push the edited user message
        // as a new sibling — it gets a fresh id on save.
        self.messages.truncate(index);
        let mut new_user = Message::from_parts(Role::User, new_parts);
        new_user.parent_id = parent_for_sibling;
        new_user.branch_index = branch_index;
        self.messages.push(new_user);

        self.editing_message = None;
        self.edit_buffer.clear();

        self.start_streaming();
    }

    fn copy_to_clipboard(&self, text: &str) {
        if let Ok(mut clipboard) = arboard::Clipboard::new() {
            let _ = clipboard.set_text(text);
        }
    }

    /// Switch to the prev/next sibling at branch point `index`. Direction is
    /// `-1` for previous, `+1` for next. Splices the new branch into
    /// `self.messages` from `index` onward by walking the tree from the
    /// chosen sibling. No-op if the message has no id (unsaved) or no
    /// sibling exists in the requested direction.
    fn navigate_sibling(&mut self, index: usize, direction: i64) {
        if self.streaming || index >= self.messages.len() {
            return;
        }
        let Some(current_id) = self.messages[index].id else {
            return;
        };
        let sibs = self.storage.siblings_of(current_id);
        if sibs.len() <= 1 {
            return;
        }
        let Some(cur_pos) = sibs.iter().position(|s| s.id == current_id) else {
            return;
        };
        let new_pos: usize = match direction.cmp(&0) {
            std::cmp::Ordering::Less if cur_pos > 0 => cur_pos - 1,
            std::cmp::Ordering::Greater if cur_pos + 1 < sibs.len() => cur_pos + 1,
            _ => return,
        };
        let target_id = sibs[new_pos].id;
        let new_path = self.storage.walk_from(target_id);
        // Truncate everything from `index` down (it belonged to the old
        // branch) and replace with the walked path from the chosen sibling.
        self.messages.truncate(index);
        self.messages.extend(new_path);
    }

    fn process_events(&mut self) {
        // Auto-title responses. Drain into a local vec so we don't hold a
        // mutable borrow on self across the storage / refresh calls below.
        let title_results: Vec<(i64, Result<String, String>)> = match &mut self.title_rx {
            Some(rx) => {
                let mut out = Vec::new();
                while let Ok(item) = rx.try_recv() {
                    out.push(item);
                }
                out
            }
            None => Vec::new(),
        };
        if !title_results.is_empty() {
            // `mark_auto_titled` was already called at dispatch (see
            // `request_auto_title`) — we only update the title text here.
            for (conv_id, res) in title_results {
                if let Ok(title) = res {
                    let cleaned = sanitize_title(&title);
                    if !cleaned.is_empty() {
                        self.storage.update_conversation_title(conv_id, &cleaned);
                    }
                }
            }
            self.refresh_conversation_list();
            self.title_rx = None;
        }

        // Process model fetch events (separate channel)
        // Drain models_rx events into a Vec first so we can call &mut self
        // helpers (show_toast, current_settings_snapshot) below without
        // overlapping with the receiver borrow.
        let model_events: Vec<StreamEvent> = match &mut self.models_rx {
            Some(rx) => {
                let mut out = Vec::new();
                while let Ok(ev) = rx.try_recv() {
                    out.push(ev);
                }
                out
            }
            None => Vec::new(),
        };
        for event in model_events {
            match event {
                StreamEvent::ModelsLoaded { url, models }
                    if !models.is_empty() && url == self.base_url =>
                {
                    self.models = models;
                    // If a conversation we just loaded asked for a specific
                    // model, restore that selection now that names are
                    // available. Otherwise clamp the existing index.
                    if let Some(name) = self.pending_model_after_models_load.take() {
                        if let Some(idx) = self.models.iter().position(|m| m == &name) {
                            self.selected_model = idx;
                        } else {
                            // Stored model isn't in this endpoint's list
                            // anymore (renamed/removed/different endpoint).
                            // Surface it instead of silently picking a
                            // different model.
                            let fallback = self
                                .models
                                .first()
                                .cloned()
                                .unwrap_or_else(|| "?".to_string());
                            self.show_toast(format!(
                                "Model '{name}' not available; using '{fallback}'"
                            ));
                            self.selected_model =
                                self.selected_model.min(self.models.len().saturating_sub(1));
                        }
                        // The model index just changed (or was confirmed) —
                        // re-baseline the settings signature so the
                        // next-frame dirty check doesn't write back
                        // identical settings and bump updated_at, which
                        // would shuffle the sidebar order.
                        let snap = self.current_settings_snapshot();
                        self.settings_signature = settings_signature(&snap);
                    } else {
                        self.selected_model =
                            self.selected_model.min(self.models.len().saturating_sub(1));
                    }
                }
                // Result for an endpoint we're no longer on — silently drop.
                StreamEvent::ModelsLoaded { .. } => {}
                StreamEvent::Done => {
                    self.models_loading = false;
                    self.models_rx = None;
                    break;
                }
                StreamEvent::Error(e) => {
                    self.error = Some(e);
                    self.models_loading = false;
                    self.models_rx = None;
                    break;
                }
                _ => {}
            }
        }

        // Process chat stream events
        let mut just_finished_streaming = false;

        if let Some(rx) = &mut self.rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    StreamEvent::Reasoning(text) => {
                        // Empty deltas should never get here (api.rs filters
                        // them) but if one slips through, drop it — it would
                        // open a <think> with no body.
                        if text.is_empty() {
                            continue;
                        }
                        if let Some(last) = self.messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            if !self.reasoning_open {
                                last.append_text("<think>\n");
                                self.reasoning_open = true;
                            }
                            last.append_text(&text);
                        }
                    }
                    StreamEvent::Token(token) => {
                        // Empty content tokens must not close a streaming
                        // <think> block — see api.rs comment for the failure
                        // mode this prevents.
                        if token.is_empty() {
                            continue;
                        }
                        if let Some(last) = self.messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            if self.reasoning_open {
                                last.append_text("\n</think>\n\n");
                                self.reasoning_open = false;
                            }
                            last.append_text(&token);
                        }
                    }
                    StreamEvent::UsageInfo(usage) => {
                        if let Some(cost) = usage.cost {
                            self.session_cost += cost;
                        }
                        self.last_usage = Some(usage);
                    }
                    StreamEvent::ToolCalls(calls) => {
                        // Attach to the streaming assistant message. The
                        // execution loop kicks in on Done — running
                        // mid-stream would race with later content tokens
                        // (some providers send a closing content delta
                        // after the tool_calls).
                        if let Some(last) = self.messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            last.tool_calls = Some(calls);
                        }
                    }
                    StreamEvent::Done => {
                        // If reasoning was streaming and the provider never sent
                        // a content delta, close the <think> tag explicitly so
                        // the saved message has a balanced block.
                        if self.reasoning_open
                            && let Some(last) = self.messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            last.append_text("\n</think>\n");
                        }
                        self.reasoning_open = false;
                        just_finished_streaming = true;
                        self.streaming = false;
                        self.cancel_token = None;
                        self.rx = None;
                        break;
                    }
                    StreamEvent::Error(e) => {
                        self.error = Some(e);
                        // Close any open reasoning block so the partial message
                        // we keep doesn't render as "streaming…" forever.
                        if self.reasoning_open
                            && let Some(last) = self.messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            last.append_text("\n</think>\n");
                        }
                        self.reasoning_open = false;
                        self.streaming = false;
                        self.cancel_token = None;
                        self.rx = None;
                        if let Some(last) = self.messages.last()
                            && last.role == Role::Assistant
                            && last.is_empty_content()
                        {
                            self.messages.pop();
                        }
                        // Save the partial response. Without this, a stream
                        // that errors out leaves an unsaved partial assistant
                        // message in memory that gets bundled into the next
                        // successful save — or lost entirely if the user
                        // closes the app before another Done.
                        just_finished_streaming = true;
                        break;
                    }
                    _ => {}
                }
            }
        }

        if just_finished_streaming && !self.messages.is_empty() {
            self.save_current();
            // After persisting the assistant turn (which may carry
            // tool_calls), start draining the queue. If any call needs
            // user approval, we'll park in `pending_approval`; otherwise
            // we'll execute everything and re-fire start_streaming.
            self.kick_off_tool_execution();
        }
    }

    /// Reset all tool-execution state. Called when the user switches/
    /// creates/deletes a conversation — without this a parked approval
    /// card from one conv would still be resolvable from another, pushing
    /// a tool_result into the wrong message chain.
    fn clear_tool_execution_state(&mut self) {
        self.pending_approval = None;
        self.pending_tool_calls.clear();
        self.tool_iterations = 0;
        // `auto_approved_tools` is documented as per-conversation in the
        // approval card hover text — clear it on switch so the user's
        // "approve all" decision in chat A doesn't carry into chat B.
        self.auto_approved_tools.clear();
    }

    /// If the just-finished assistant turn emitted tool_calls and we're
    /// under the iteration cap, queue them and start processing. No-op
    /// otherwise.
    fn kick_off_tool_execution(&mut self) {
        let calls = self
            .messages
            .last()
            .and_then(|m| m.tool_calls.clone())
            .unwrap_or_default();
        if calls.is_empty() {
            return;
        }
        if self.tool_iterations >= MAX_TOOL_ITERATIONS {
            self.show_toast(format!(
                "Tool iteration cap ({MAX_TOOL_ITERATIONS}) reached \u{2014} stopping"
            ));
            return;
        }
        self.pending_tool_calls = calls;
        self.process_pending_tool_calls();
    }

    /// Drain auto-safety calls (and any pre-approved confirm calls) from
    /// the queue. Park the first confirm call we hit in
    /// `pending_approval` and return; the UI's approval card will resume
    /// us via `resolve_pending_approval`.
    fn process_pending_tool_calls(&mut self) {
        while let Some(call) = self.pending_tool_calls.first().cloned() {
            let safety = self.safety_for(&call.function.name);
            if safety == crate::tools::Safety::Auto
                || self.auto_approved_tools.contains(&call.function.name)
            {
                self.pending_tool_calls.remove(0);
                let body = self.execute_tool_call(&call);
                self.messages
                    .push(Message::tool_result(call.id.clone(), body));
            } else {
                // Park for user approval.
                self.pending_approval = Some(self.pending_tool_calls.remove(0));
                return;
            }
        }
        // Queue empty — re-stream so the model sees the tool results.
        self.tool_iterations += 1;
        self.start_streaming();
    }

    /// Resolve the parked approval card: `Approve` runs the call and
    /// resumes the queue; `ApproveAll` adds the tool name to the
    /// per-session allowlist before doing the same; `Reject` pushes a
    /// "rejected by user" tool result and resumes.
    fn resolve_pending_approval(&mut self, decision: ApprovalDecision) {
        let Some(call) = self.pending_approval.take() else {
            return;
        };
        let body = match decision {
            ApprovalDecision::Approve => self.execute_tool_call(&call),
            ApprovalDecision::ApproveAll => {
                self.auto_approved_tools.insert(call.function.name.clone());
                self.execute_tool_call(&call)
            }
            ApprovalDecision::Reject => "Rejected by user.".to_string(),
        };
        self.messages
            .push(Message::tool_result(call.id.clone(), body));
        self.process_pending_tool_calls();
    }

    /// Render the parked tool-approval card if one's pending. Returns the
    /// user's decision so the caller can apply it without dragging an
    /// extra `&mut self` borrow through the UI closure.
    fn render_approval_card(&self, ui: &mut egui::Ui) -> Option<ApprovalDecision> {
        let call = self.pending_approval.as_ref()?;
        let mut decision: Option<ApprovalDecision> = None;
        egui::Frame::group(ui.style())
            .fill(ui.visuals().selection.bg_fill.linear_multiply(0.15))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("⚠ Tool requires approval").strong());
                    ui.weak(format!("({})", call.function.name));
                });
                // Pretty-print arguments JSON. Falls back to the raw string
                // if the model emitted invalid JSON (rare, but possible
                // during streaming if the loop fires before a closing brace).
                let pretty = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                    .and_then(|v| serde_json::to_string_pretty(&v))
                    .unwrap_or_else(|_| call.function.arguments.clone());
                egui::ScrollArea::vertical()
                    .max_height(160.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut pretty.as_str())
                                .code_editor()
                                .desired_width(f32::INFINITY)
                                .interactive(false),
                        );
                    });
                ui.horizontal(|ui| {
                    if ui.button("Approve").clicked() {
                        decision = Some(ApprovalDecision::Approve);
                    }
                    if ui
                        .button("Approve all in this conv")
                        .on_hover_text("Skip the prompt for any future call to this tool name in this conversation")
                        .clicked()
                    {
                        decision = Some(ApprovalDecision::ApproveAll);
                    }
                    if ui.button("Reject").clicked() {
                        decision = Some(ApprovalDecision::Reject);
                    }
                });
            });
        decision
    }

    /// Look up the configured safety for a tool by name. Defaults to
    /// `Confirm` if the tool isn't loaded — refusing to silently auto-
    /// execute a tool the user hasn't installed feels safer than the
    /// alternative.
    fn safety_for(&self, name: &str) -> crate::tools::Safety {
        self.loaded_tools
            .iter()
            .find(|d| d.name == name)
            .map(|d| d.safety)
            .unwrap_or(crate::tools::Safety::Confirm)
    }

    /// Run one tool call against its matching ToolDef. Returns the
    /// stringified result (or an error message); never panics.
    fn execute_tool_call(&self, call: &crate::message::ToolCall) -> String {
        use crate::tools::{Handler, run_builtin, run_shell_tool};
        let def = match self
            .loaded_tools
            .iter()
            .find(|d| d.name == call.function.name)
        {
            Some(d) => d,
            None => return format!("error: unknown tool '{}'", call.function.name),
        };
        let args: serde_json::Value = match serde_json::from_str(&call.function.arguments) {
            Ok(v) => v,
            Err(e) => return format!("error: invalid arguments JSON: {e}"),
        };
        let wd = std::path::Path::new(&self.working_dir);
        let result = match &def.handler {
            Handler::Builtin(b) => run_builtin(&b.0, &args, wd),
            Handler::Shell { shell } => run_shell_tool(shell, &args, wd),
        };
        result.unwrap_or_else(|e| format!("error: {e}"))
    }

    pub fn reload_config(&mut self, ctx: &egui::Context) {
        let new_config = match Config::try_load() {
            Ok(c) => c,
            Err(e) => {
                self.error = Some(format!("Failed to reload config: {e}"));
                return;
            }
        };

        // Reload fonts only if font settings actually changed
        let fonts_changed = self.config.font_family != new_config.font_family
            || self.config.mono_font_family != new_config.mono_font_family
            || (self.config.font_size - new_config.font_size).abs() > f32::EPSILON
            || (self.config.mono_font_size - new_config.mono_font_size).abs() > f32::EPSILON;
        if fonts_changed {
            configure_fonts(ctx, &new_config);
        }

        // Update theme
        if self.dark_mode != new_config.dark_mode {
            self.dark_mode = new_config.dark_mode;
            apply_theme(ctx, self.dark_mode);
        }

        // Update UI scale
        if (self.config.ui_scale - new_config.ui_scale).abs() > f32::EPSILON {
            ctx.set_zoom_factor(new_config.ui_scale);
        }

        // Apply new settings
        self.system_prompt = new_config.system_prompt.clone();
        self.temperature = new_config.temperature;
        self.max_tokens = new_config.max_tokens;
        self.use_max_tokens = new_config.use_max_tokens;
        self.top_p = new_config.top_p;
        self.frequency_penalty = new_config.frequency_penalty;
        self.presence_penalty = new_config.presence_penalty;
        self.stop_sequences = new_config.stop_sequences.clone();

        // Capture old endpoint state for change detection
        let old_url = self.base_url.clone();
        let old_api_key = self.current_api_key().map(|s| s.to_string());

        // Update endpoints list
        self.saved_endpoints = new_config.saved_endpoints.clone();

        // Apply default_endpoint (symmetric with save which writes base_url -> default_endpoint)
        self.base_url = new_config.default_endpoint.clone();

        // If the chosen endpoint isn't in the saved list, fallback to first
        if !self
            .saved_endpoints
            .iter()
            .any(|ep| ep.url == self.base_url)
        {
            if let Some(first) = self.saved_endpoints.first() {
                self.base_url = first.url.clone();
            }
        }

        // Only re-fetch models if the endpoint URL or API key actually changed
        let new_api_key = self.current_api_key().map(|s| s.to_string());
        if (self.base_url != old_url || new_api_key != old_api_key) && !self.streaming {
            self.fetch_models();
        }

        self.config = new_config;
        self.error = None;
        ctx.request_repaint();
    }
}

fn configure_fonts(ctx: &egui::Context, config: &Config) {
    load_custom_fonts(ctx, config);

    let mut style = (*ctx.global_style()).clone();
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(config.font_size, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(config.mono_font_size, egui::FontFamily::Monospace),
    );
    ctx.set_global_style(style);
}

fn load_custom_fonts(ctx: &egui::Context, config: &Config) {
    use font_kit::source::SystemSource;

    let mut fonts = egui::FontDefinitions::default();
    let source = SystemSource::new();

    if !config.font_family.is_empty() {
        if let Some(data) = lookup_font(&source, &config.font_family) {
            fonts
                .font_data
                .insert("custom_proportional".to_owned(), data.into());
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                family.insert(0, "custom_proportional".to_owned());
            }
        } else {
            eprintln!(
                "Warning: font '{}' not found, using default",
                config.font_family
            );
        }
    }

    if !config.mono_font_family.is_empty() {
        if let Some(data) = lookup_font(&source, &config.mono_font_family) {
            fonts
                .font_data
                .insert("custom_monospace".to_owned(), data.into());
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                family.insert(0, "custom_monospace".to_owned());
            }
        } else {
            eprintln!(
                "Warning: font '{}' not found, using default",
                config.mono_font_family
            );
        }
    }

    ctx.set_fonts(fonts);
}

fn lookup_font(source: &font_kit::source::SystemSource, name: &str) -> Option<egui::FontData> {
    use font_kit::family_name::FamilyName;
    use font_kit::properties::Properties;

    let handle = source
        .select_best_match(&[FamilyName::Title(name.to_string())], &Properties::new())
        .ok()?;
    let font = handle.load().ok()?;
    let data = font.copy_font_data()?;
    Some(egui::FontData::from_owned((*data).clone()))
}

fn apply_theme(ctx: &egui::Context, dark: bool) {
    if dark {
        ctx.set_visuals(egui::Visuals::dark());
    } else {
        ctx.set_visuals(egui::Visuals::light());
    }
}

impl eframe::App for ChatApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.process_events();

        // Drag-and-drop: read each frame, *not* per-panel — eframe delivers
        // dropped_files at the context level, so a single drain at the top
        // of the frame is the right shape.
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            self.handle_dropped_files(dropped);
        }

        if self.streaming || self.models_loading {
            ctx.request_repaint();
        }

        // Keyboard shortcuts. Gated on `wants_keyboard_input` so they don't
        // hijack typing into a focused TextEdit (system prompt, rename buffer,
        // edit buffer, stop sequences). Tradeoff: to use Ctrl+F or Ctrl+N
        // while a text field is focused, the user has to defocus first
        // (Escape, or click elsewhere). Worth it to avoid surprising
        // interrupts while editing.
        let typing = ctx.egui_wants_keyboard_input();
        let new_chat = !typing
            && ui.input(|i| i.key_pressed(egui::Key::N) && (i.modifiers.ctrl || i.modifiers.command));
        if new_chat {
            self.new_conversation();
        }
        let toggle_find = !typing
            && ui.input(|i| i.key_pressed(egui::Key::F) && (i.modifiers.ctrl || i.modifiers.command));
        if toggle_find {
            self.show_find = !self.show_find;
            if !self.show_find {
                self.find_query.clear();
                self.last_scrolled_find_query = None;
            }
        }
        // Escape closes find from anywhere — including from inside the find
        // input itself, which is exactly where you want it to work.
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) && self.show_find {
            self.show_find = false;
            self.find_query.clear();
            self.last_scrolled_find_query = None;
        }

        // Top bar
        egui::Panel::top("top_bar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                let sidebar_icon = if self.show_sidebar { "◀" } else { "▶" };
                if ui
                    .button(sidebar_icon)
                    .on_hover_text("Toggle sidebar")
                    .clicked()
                {
                    self.show_sidebar = !self.show_sidebar;
                }

                ui.separator();
                ui.label("Model:");
                egui::ComboBox::from_id_salt("model_selector")
                    .selected_text(self.selected_model_name())
                    .show_ui(ui, |ui| {
                        for (i, model) in self.models.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_model, i, model);
                        }
                    });

                ui.separator();
                ui.label("Endpoint:");

                if self.saved_endpoints.len() > 1 {
                    let mut new_url = None;
                    egui::ComboBox::from_id_salt("endpoint_selector")
                        .selected_text(&self.base_url)
                        .width(200.0)
                        .show_ui(ui, |ui| {
                            for ep in &self.saved_endpoints {
                                if ui
                                    .selectable_label(self.base_url == ep.url, &ep.url)
                                    .clicked()
                                {
                                    new_url = Some(ep.url.clone());
                                }
                            }
                        });
                    if let Some(url) = new_url {
                        self.base_url = url;
                        if !self.streaming {
                            self.fetch_models();
                        }
                    }
                } else {
                    ui.add(egui::TextEdit::singleline(&mut self.base_url).desired_width(200.0));
                }

                if ui
                    .add_enabled(!self.streaming, egui::Button::new("↻"))
                    .on_hover_text("Refresh models")
                    .clicked()
                {
                    self.fetch_models();
                }

                if ui.button("+").on_hover_text("Manage endpoints").clicked() {
                    self.show_endpoints = !self.show_endpoints;
                }

                ui.separator();
                
                let context_sidebar_icon = if self.show_context_sidebar { "☰▶" } else { "☰◀" };
                if ui
                    .button(context_sidebar_icon)
                    .on_hover_text("Toggle Context Sidebar")
                    .clicked()
                {
                    self.show_context_sidebar = !self.show_context_sidebar;
                }

                let settings_label = if self.show_settings { "⚙ ▼" } else { "⚙" };
                if ui
                    .button(settings_label)
                    .on_hover_text("Settings")
                    .clicked()
                {
                    self.show_settings = !self.show_settings;
                }

                let theme_icon = if self.dark_mode { "☀" } else { "🌙" };
                if ui
                    .button(theme_icon)
                    .on_hover_text("Toggle theme")
                    .clicked()
                {
                    self.dark_mode = !self.dark_mode;
                    self.config.dark_mode = self.dark_mode;
                    apply_theme(&ctx, self.dark_mode);
                }
            });

            // Endpoints manager
            if self.show_endpoints {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Add endpoint:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_endpoint)
                            .hint_text("http://localhost:8080/v1")
                            .desired_width(250.0),
                    );
                    ui.label("API key:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_endpoint_key)
                            .hint_text("(optional)")
                            .password(true)
                            .desired_width(150.0),
                    );
                    if ui.button("Add").clicked() && !self.new_endpoint.trim().is_empty() {
                        let url = self.new_endpoint.trim().to_string();
                        let api_key = if self.new_endpoint_key.trim().is_empty() {
                            None
                        } else {
                            Some(self.new_endpoint_key.trim().to_string())
                        };
                        if !self.saved_endpoints.iter().any(|ep| ep.url == url) {
                            self.saved_endpoints.push(Endpoint { url, api_key });
                        }
                        self.new_endpoint.clear();
                        self.new_endpoint_key.clear();
                    }
                });
                let mut to_remove = None;
                let mut key_edit: Option<(usize, Option<String>)> = None;
                for (i, ep) in self.saved_endpoints.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(&ep.url);
                        let key_icon = if ep.api_key.is_some() { "🔑" } else { "🔒" };
                        let tooltip = if ep.api_key.is_some() {
                            "API key set — click to clear"
                        } else {
                            "No API key"
                        };
                        if ui.small_button(key_icon).on_hover_text(tooltip).clicked()
                            && ep.api_key.is_some()
                        {
                            key_edit = Some((i, None));
                        }
                        if self.saved_endpoints.len() > 1 && ui.small_button("✕").clicked() {
                            to_remove = Some(i);
                        }
                    });
                }
                if let Some((i, new_key)) = key_edit {
                    self.saved_endpoints[i].api_key = new_key;
                }
                if let Some(i) = to_remove {
                    let removed = self.saved_endpoints.remove(i);
                    if self.base_url == removed.url {
                        if let Some(first) = self.saved_endpoints.first() {
                            self.base_url = first.url.clone();
                        }
                        if !self.streaming {
                            self.fetch_models();
                        }
                    }
                }
            }

            // Collapsible settings panel
            if self.show_settings {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("System prompt:");
                });
                ui.add(
                    egui::TextEdit::multiline(&mut self.system_prompt)
                        .hint_text("You are a helpful assistant...")
                        .desired_rows(2)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    ui.label("Working dir:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.working_dir)
                            .hint_text("~/projects/foo")
                            .desired_width(360.0),
                    )
                    .on_hover_text(
                        "Tools (read_file, run_shell, etc.) resolve relative paths against this directory",
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Temperature:");
                    ui.add(egui::Slider::new(&mut self.temperature, 0.0..=2.0).step_by(0.1));
                    ui.separator();
                    ui.checkbox(&mut self.use_max_tokens, "Max tokens:");
                    ui.add_enabled(
                        self.use_max_tokens,
                        egui::Slider::new(&mut self.max_tokens, 64..=16384).logarithmic(true),
                    );
                });

                // Advanced sampling params (collapsed by default — most users don't need them).
                egui::CollapsingHeader::new("Advanced sampling")
                    .id_salt("advanced_sampling")
                    .default_open(false)
                    .show(ui, |ui| {
                        // top_p (Option<f32>)
                        ui.horizontal(|ui| {
                            let mut enabled = self.top_p.is_some();
                            if ui.checkbox(&mut enabled, "top_p:").changed() {
                                self.top_p = if enabled { Some(1.0) } else { None };
                            }
                            if let Some(v) = self.top_p.as_mut() {
                                ui.add(egui::Slider::new(v, 0.0..=1.0).step_by(0.05));
                            }
                        });
                        ui.horizontal(|ui| {
                            let mut enabled = self.frequency_penalty.is_some();
                            if ui.checkbox(&mut enabled, "frequency_penalty:").changed() {
                                self.frequency_penalty = if enabled { Some(0.0) } else { None };
                            }
                            if let Some(v) = self.frequency_penalty.as_mut() {
                                ui.add(egui::Slider::new(v, -2.0..=2.0).step_by(0.1));
                            }
                        });
                        ui.horizontal(|ui| {
                            let mut enabled = self.presence_penalty.is_some();
                            if ui.checkbox(&mut enabled, "presence_penalty:").changed() {
                                self.presence_penalty = if enabled { Some(0.0) } else { None };
                            }
                            if let Some(v) = self.presence_penalty.as_mut() {
                                ui.add(egui::Slider::new(v, -2.0..=2.0).step_by(0.1));
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Stop sequences (one per line, max 4):");
                        });
                        let mut joined = self.stop_sequences.join("\n");
                        if ui
                            .add(
                                egui::TextEdit::multiline(&mut joined)
                                    .hint_text("e.g.\\nUser:\\nAssistant:")
                                    .desired_rows(2)
                                    .desired_width(f32::INFINITY),
                            )
                            .changed()
                        {
                            self.stop_sequences = joined
                                .lines()
                                .map(|s| s.to_string())
                                .filter(|s| !s.is_empty())
                                .take(4)
                                .collect();
                        }
                    });

                ui.separator();
                let mut fonts_changed = false;
                ui.horizontal(|ui| {
                    ui.label("Font:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.config.font_family)
                            .hint_text("default")
                            .desired_width(120.0),
                    );
                    ui.label("Size:");
                    if ui
                        .add(egui::Slider::new(&mut self.config.font_size, 8.0..=24.0).step_by(1.0))
                        .changed()
                    {
                        fonts_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Mono:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.config.mono_font_family)
                            .hint_text("default")
                            .desired_width(120.0),
                    );
                    ui.label("Size:");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.config.mono_font_size, 8.0..=24.0)
                                .step_by(1.0),
                        )
                        .changed()
                    {
                        fonts_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("UI scale:");
                    if ui
                        .add(egui::Slider::new(&mut self.config.ui_scale, 0.75..=2.0).step_by(0.05))
                        .changed()
                    {
                        ctx.set_zoom_factor(self.config.ui_scale);
                    }
                });
                if fonts_changed {
                    configure_fonts(&ctx, &self.config);
                }

                ui.horizontal(|ui| {
                    if ui.button("Save settings").clicked() {
                        self.config.dark_mode = self.dark_mode;
                        self.config.default_endpoint = self.base_url.clone();
                        self.config.system_prompt = self.system_prompt.clone();
                        self.config.temperature = self.temperature;
                        self.config.max_tokens = self.max_tokens;
                        self.config.use_max_tokens = self.use_max_tokens;
                        self.config.saved_endpoints = self.saved_endpoints.clone();
                        self.config.top_p = self.top_p;
                        self.config.frequency_penalty = self.frequency_penalty;
                        self.config.presence_penalty = self.presence_penalty;
                        self.config.stop_sequences = self.stop_sequences.clone();
                        if let Err(e) = self.config.save() {
                            self.error = Some(format!("Failed to save settings: {e}"));
                        }
                    }
                    if ui.button("Reload config").clicked() {
                        self.reload_config(&ctx);
                    }
                    if ui.button("Reset to defaults").clicked() {
                        self.config = Config::default();
                        self.dark_mode = self.config.dark_mode;
                        self.base_url = self.config.default_endpoint.clone();
                        self.system_prompt = self.config.system_prompt.clone();
                        self.temperature = self.config.temperature;
                        self.max_tokens = self.config.max_tokens;
                        self.use_max_tokens = self.config.use_max_tokens;
                        self.saved_endpoints = self.config.saved_endpoints.clone();
                        self.top_p = self.config.top_p;
                        self.frequency_penalty = self.config.frequency_penalty;
                        self.presence_penalty = self.config.presence_penalty;
                        self.stop_sequences = self.config.stop_sequences.clone();
                        configure_fonts(&ctx, &self.config);
                        ctx.set_zoom_factor(self.config.ui_scale);
                        apply_theme(&ctx, self.config.dark_mode);
                        if let Err(e) = self.config.save() {
                            self.error = Some(format!("Failed to save settings: {e}"));
                        }
                        self.fetch_models();
                    }
                    let presets_label = if self.show_presets {
                        "Presets ▼"
                    } else {
                        "Presets"
                    };
                    if ui.button(presets_label).clicked() {
                        self.show_presets = !self.show_presets;
                        if self.show_presets {
                            self.presets_cache = self.storage.list_presets();
                        }
                    }
                });

                if self.show_presets {
                    self.show_presets_panel(ui);
                }
            }
        });

        // Sidebar
        if self.show_sidebar {
            egui::Panel::left("sidebar")
                .default_size(200.0)
                .show_inside(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("+ New Chat").clicked() {
                            self.new_conversation();
                        }
                        if ui.button("🔍").on_hover_text("Search").clicked() {
                            self.show_search = !self.show_search;
                        }
                    });

                    if self.show_search {
                        ui.separator();
                        let search_changed = ui
                            .add(
                                egui::TextEdit::singleline(&mut self.search_query)
                                    .hint_text("Search conversations...")
                                    .desired_width(f32::INFINITY),
                            )
                            .changed();

                        if search_changed {
                            if self.search_query.is_empty() {
                                self.search_results.clear();
                            } else {
                                self.search_results = self.storage.search(&self.search_query);
                            }
                        }

                        if !self.search_query.is_empty() && !self.search_results.is_empty() {
                            let mut selected_conv = None;
                            for (conv_id, title, snippet) in &self.search_results {
                                if ui
                                    .button(title.as_str())
                                    .on_hover_text(snippet.as_str())
                                    .clicked()
                                {
                                    selected_conv = Some(*conv_id);
                                }
                            }
                            if let Some(conv_id) = selected_conv {
                                self.load_conversation(conv_id);
                                self.show_search = false;
                                self.search_query.clear();
                                self.search_results.clear();
                            }
                        }
                    }

                    ui.separator();

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let mut action: Option<SidebarAction> = None;

                        for (id, title, pinned) in &self.conversation_list {
                            let is_current = self.current_conversation_id == Some(*id);

                            // Inline rename mode
                            if self.renaming_conversation == Some(*id) {
                                ui.horizontal(|ui| {
                                    let resp = ui.text_edit_singleline(&mut self.rename_buffer);
                                    if resp.lost_focus()
                                        || ui.input(|i| i.key_pressed(egui::Key::Enter))
                                    {
                                        action = Some(SidebarAction::FinishRename(*id));
                                    }
                                    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                        self.renaming_conversation = None;
                                        self.rename_buffer.clear();
                                    }
                                });
                            } else {
                                // Kebab menu pinned right, title fills remaining space
                                let row_height = ui.spacing().interact_size.y;
                                let full_width = ui.available_width();
                                let (row_rect, _) = ui.allocate_exact_size(
                                    egui::vec2(full_width, row_height),
                                    egui::Sense::hover(),
                                );

                                // Kebab button on the right
                                let kebab_width = 20.0;
                                let kebab_rect = egui::Rect::from_min_size(
                                    egui::pos2(row_rect.right() - kebab_width, row_rect.top()),
                                    egui::vec2(kebab_width, row_height),
                                );
                                let mut kebab_ui =
                                    ui.new_child(egui::UiBuilder::new().max_rect(kebab_rect));
                                kebab_ui.menu_button("...", |ui| {
                                    let pin_label = if *pinned { "Unpin" } else { "Pin" };
                                    if ui.button(pin_label).clicked() {
                                        action = Some(SidebarAction::TogglePin(*id));
                                        ui.close();
                                    }
                                    if ui.button("Rename").clicked() {
                                        action = Some(SidebarAction::StartRename(*id));
                                        ui.close();
                                    }
                                    if ui.button("Export").clicked() {
                                        action = Some(SidebarAction::Export(*id));
                                        ui.close();
                                    }
                                    if ui.button("Delete").clicked() {
                                        action = Some(SidebarAction::Delete(*id));
                                        ui.close();
                                    }
                                });

                                // Title label fills the rest
                                let title_rect = egui::Rect::from_min_max(
                                    row_rect.min,
                                    egui::pos2(
                                        row_rect.right() - kebab_width - 4.0,
                                        row_rect.bottom(),
                                    ),
                                );
                                let mut title_ui =
                                    ui.new_child(egui::UiBuilder::new().max_rect(title_rect));
                                let display_title = if *pinned {
                                    format!("📌 {title}")
                                } else {
                                    title.clone()
                                };
                                let resp = title_ui.add(
                                    egui::Label::new(egui::RichText::new(display_title).color(
                                        if is_current {
                                            title_ui.visuals().strong_text_color()
                                        } else {
                                            title_ui.visuals().text_color()
                                        },
                                    ))
                                    .truncate()
                                    .selectable(false)
                                    .sense(egui::Sense::click()),
                                );

                                if resp.clicked() && !is_current {
                                    action = Some(SidebarAction::Load(*id));
                                }

                                // Right-click context menu
                                resp.context_menu(|ui| {
                                    let pin_label = if *pinned { "Unpin" } else { "Pin" };
                                    if ui.button(pin_label).clicked() {
                                        action = Some(SidebarAction::TogglePin(*id));
                                        ui.close();
                                    }
                                    if ui.button("Rename").clicked() {
                                        action = Some(SidebarAction::StartRename(*id));
                                        ui.close();
                                    }
                                    if ui.button("Export").clicked() {
                                        action = Some(SidebarAction::Export(*id));
                                        ui.close();
                                    }
                                    if ui.button("Delete").clicked() {
                                        action = Some(SidebarAction::Delete(*id));
                                        ui.close();
                                    }
                                });
                            }
                            ui.separator();
                        }

                        match action {
                            Some(SidebarAction::Load(id)) => self.load_conversation(id),
                            Some(SidebarAction::Delete(id)) => self.delete_conversation(id),
                            Some(SidebarAction::Export(id)) => {
                                let md = self.storage.export_markdown(id);
                                self.copy_to_clipboard(&md);
                            }
                            Some(SidebarAction::StartRename(id)) => {
                                if let Some((_, title, _)) = self
                                    .conversation_list
                                    .iter()
                                    .find(|(cid, _, _)| *cid == id)
                                {
                                    self.rename_buffer = title.clone();
                                }
                                self.renaming_conversation = Some(id);
                            }
                            Some(SidebarAction::FinishRename(id)) => {
                                let new_title = self.rename_buffer.trim().to_string();
                                if !new_title.is_empty() {
                                    self.storage.update_conversation_title(id, &new_title);
                                    self.refresh_conversation_list();
                                }
                                self.renaming_conversation = None;
                                self.rename_buffer.clear();
                            }
                            Some(SidebarAction::TogglePin(id)) => {
                                let was_pinned = self
                                    .conversation_list
                                    .iter()
                                    .find(|(cid, _, _)| *cid == id)
                                    .map(|(_, _, p)| *p)
                                    .unwrap_or(false);
                                self.storage.set_pinned(id, !was_pinned);
                                self.refresh_conversation_list();
                            }
                            None => {}
                        }
                    });
                });
        }

        if self.show_context_sidebar {
            egui::Panel::right("context_sidebar")
                .default_size(250.0)
                .show_inside(ui, |ui| {
                    ui.heading("Context & Stats");
                    ui.separator();
                    ui.add_space(8.0);

                    // Estimate tokens based on current conversation messages
                    // A very rough rule of thumb is ~4 characters per token
                    let mut char_count = 0;
                    for msg in &self.messages {
                        char_count += msg.text_str().len();
                    }
                    if !self.system_prompt.is_empty() {
                        char_count += self.system_prompt.len();
                    }
                    char_count += self.input.len();

                    // For accurate tokens, we could use tiktoken-rs if we know the model,
                    // but for local LLMs, a rough estimate is okay, or we can use tiktoken to guess
                    let approx_tokens = {
                        let mut full_text = self.system_prompt.clone();
                        full_text.push('\n');
                        for m in &self.messages {
                            full_text.push_str(&m.text_str());
                            full_text.push('\n');
                        }
                        full_text.push_str(&self.input);

                        let bpe = tiktoken_rs::cl100k_base_singleton();
                        bpe.encode_with_special_tokens(&full_text).len()
                    };

                    ui.label(egui::RichText::new("Context Size").strong());
                    ui.label(format!("{} Characters", char_count));
                    ui.label(format!("~{} Tokens", approx_tokens));
                    ui.add_space(8.0);

                    ui.label(egui::RichText::new("Chat Spend").strong());
                    if self.session_cost > 0.0 {
                        ui.label(format!("${:.6}", self.session_cost));
                    } else {
                        ui.label("$0.00");
                    }
                    ui.add_space(8.0);

                    if let Some(usage) = &self.last_usage {
                        ui.separator();
                        ui.label(egui::RichText::new("Last Request").strong());
                        if let Some(p) = usage.prompt_tokens {
                            ui.label(format!("Prompt: {}", p));
                        }
                        if let Some(c) = usage.completion_tokens {
                            ui.label(format!("Completion: {}", c));
                        }
                        if let Some(t) = usage.total_tokens {
                            ui.label(format!("Total: {}", t));
                        }
                        if let Some(cost) = usage.cost {
                            ui.label(format!("Cost: ${:.6}", cost));
                        }
                    }
                });
        }

        // Bottom input panel (only spans area between sidebars)
        egui::Panel::bottom("input_panel").show_inside(ui, |ui| {
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, format!("Error: {err}"));
            }

            // Tool-call approval card: rendered above the input area when
            // a confirm-safety call is parked. The execution loop is paused
            // until the user clicks one of the three buttons.
            let approval_decision = self.render_approval_card(ui);
            if let Some(d) = approval_decision {
                self.resolve_pending_approval(d);
            }

            // Pending attachment chips: render one row per attachment with a
            // remove (✕) button. Two-pass to avoid mutating during iteration.
            if !self.pending_attachments.is_empty() {
                let mut remove_idx: Option<usize> = None;
                ui.horizontal_wrapped(|ui| {
                    for (i, (name, _)) in self.pending_attachments.iter().enumerate() {
                        ui.group(|ui| {
                            ui.label(format!("📎 {name}"));
                            if ui.small_button("✕").on_hover_text("Remove").clicked() {
                                remove_idx = Some(i);
                            }
                        });
                    }
                });
                if let Some(i) = remove_idx {
                    self.pending_attachments.remove(i);
                }
            }

            let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
            let shift_held = ui.input(|i| i.modifiers.shift);

            ui.horizontal(|ui| {
                let response = ui.add_sized(
                    [ui.available_width() - 60.0, 90.0],
                    egui::TextEdit::multiline(&mut self.input)
                        .hint_text("Type a message... (Shift+Enter for newline)")
                        .desired_rows(4),
                );

                if enter_pressed && response.has_focus() {
                    if shift_held {
                        self.input.push('\n');
                    } else {
                        if self.input.ends_with('\n') {
                            self.input.pop();
                        }
                        self.send_message();
                    }
                }

                if self.streaming {
                    if ui.button("■ Stop").clicked() {
                        self.stop_streaming();
                    }
                } else {
                    let send_enabled = !self.input.trim().is_empty() && !self.models.is_empty();
                    if ui
                        .add_enabled(send_enabled, egui::Button::new("Send"))
                        .clicked()
                    {
                        self.send_message();
                        response.request_focus();
                    }
                }
            });

            // Live token estimate. Cached on a cheap content signature so we
            // don't re-encode the entire conversation through tiktoken every
            // frame; only when something materially changed.
            let signature = token_signature(&self.system_prompt, &self.messages, &self.input);
            let approx_tokens = match self.token_cache {
                Some((sig, count)) if sig == signature => count,
                _ => {
                    let count = estimate_tokens(&self.system_prompt, &self.messages, &self.input);
                    self.token_cache = Some((signature, count));
                    count
                }
            };

            ui.horizontal(|ui| {
                if self.streaming {
                    ui.spinner();
                }
                ui.weak(format!("~{approx_tokens} tokens"));
                if let Some(usage) = &self.last_usage {
                    let parts: Vec<String> = [
                        usage.prompt_tokens.map(|t| format!("prompt: {t}")),
                        usage.completion_tokens.map(|t| format!("completion: {t}")),
                        usage.total_tokens.map(|t| format!("total: {t}")),
                    ]
                    .into_iter()
                    .flatten()
                    .collect();
                    if !parts.is_empty() {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.weak(parts.join(" | "));
                        });
                    }
                }
            });
        });

        // Central message area
        let panel_fill = ui.visuals().panel_fill;
        let mut prefill_input: Option<String> = None;
        egui::Frame::new()
            .fill(panel_fill)
            .inner_margin(egui::Margin::symmetric(16, 0))
            .show(ui, |ui| {
                // Find bar (Ctrl+F)
                // Compute find matches once per frame: lowercased query,
                // per-message bool, first-match index. Reused by both the count
                // display and the message render loop.
                let find_q_lower: Option<String> = if self.show_find && !self.find_query.is_empty()
                {
                    Some(self.find_query.to_ascii_lowercase())
                } else {
                    None
                };
                let match_flags: Vec<bool> = match find_q_lower.as_deref() {
                    Some(q) => self
                        .messages
                        .iter()
                        .map(|m| m.text_str().to_ascii_lowercase().contains(q))
                        .collect(),
                    None => Vec::new(),
                };
                let first_match_idx: Option<usize> = match_flags.iter().position(|&b| b);

                if self.show_find {
                    ui.horizontal(|ui| {
                        ui.label("🔍");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.find_query)
                                .hint_text("Find in conversation…")
                                .desired_width(400.0),
                        );
                        if !self.find_query.is_empty() {
                            let matches = match_flags.iter().filter(|&&b| b).count();
                            ui.weak(format!(
                                "{matches} match{}",
                                if matches == 1 { "" } else { "es" }
                            ));
                        }
                        if ui.small_button("✕").clicked() {
                            self.show_find = false;
                            self.find_query.clear();
                            self.last_scrolled_find_query = None;
                        }
                    });
                    ui.separator();
                }

                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if self.messages.is_empty() {
                            ui.vertical_centered(|ui| {
                                ui.add_space(60.0);
                                ui.heading("hChat");
                                ui.label("Lightweight local LLM chat");
                                ui.add_space(20.0);
                                ui.weak("Try one of these to get started:");
                                ui.add_space(8.0);
                                for (label, prompt) in STARTER_PROMPTS {
                                    if ui
                                        .button(*label)
                                        .on_hover_text(*prompt)
                                        .clicked()
                                    {
                                        prefill_input = Some((*prompt).to_string());
                                    }
                                }
                                ui.add_space(20.0);
                                ui.weak("Ctrl+N for new chat · Ctrl+F to find · /help for commands");
                            });
                            return;
                        }

                        let mut action: Option<MessageAction> = None;
                        let msg_count = self.messages.len();

                        // If we issue a scroll_to_me this frame, store the query
                        // here so subsequent frames know not to fire it again.
                        let mut pending_scroll_query: Option<String> = None;

                        // Snapshot fields the inner closure-free loop needs without
                        // re-borrowing self.
                        let editing_message = self.editing_message;
                        let streaming = self.streaming;
                        let highlight_fill =
                            ui.visuals().selection.bg_fill.linear_multiply(0.25);

                        // (current_branch_position, total_siblings) per
                        // message — feeds the ◀ N/M ▶ navigator. Refreshed
                        // from the DB only when the active path's id
                        // signature actually changes (regenerate, edit,
                        // navigation, send). Without this we'd hit
                        // siblings_of twice per message every frame just
                        // to render arrows that almost never move.
                        let sig = sibling_signature(&self.messages);
                        if sig != self.sibling_info_signature
                            || self.sibling_info_cache.len() != msg_count
                        {
                            self.sibling_info_cache = (0..msg_count)
                                .map(|i| {
                                    let mid = self.messages[i].id?;
                                    let sibs = self.storage.siblings_of(mid);
                                    if sibs.len() <= 1 {
                                        return None;
                                    }
                                    let pos = sibs.iter().position(|s| s.id == mid)?;
                                    Some((pos, sibs.len()))
                                })
                                .collect();
                            self.sibling_info_signature = sig;
                        }
                        let sibling_info = &self.sibling_info_cache;

                        // We iterate by index to avoid holding a borrow on self.messages
                        // across the body, which needs &mut self for commonmark_cache.
                        for i in 0..msg_count {
                            let (role, content_clone, created_at, tool_calls_snap);
                            {
                                let msg = &self.messages[i];
                                role = msg.role.clone();
                                content_clone = msg.text_str();
                                created_at = msg.created_at;
                                tool_calls_snap = msg.tool_calls.clone();
                            }
                            let (prefix, color) = match role {
                                Role::User => ("You", egui::Color32::from_rgb(100, 180, 255)),
                                Role::Assistant => ("AI", egui::Color32::from_rgb(100, 255, 150)),
                                Role::System => ("System", egui::Color32::GRAY),
                                Role::Tool => ("Tool", egui::Color32::from_rgb(180, 140, 220)),
                            };

                            let matches_find = match_flags.get(i).copied().unwrap_or(false);
                            let sib = sibling_info.get(i).copied().flatten();

                            let render_inner = |ui: &mut egui::Ui,
                                                    cache: &mut CommonMarkCache,
                                                    edit_buffer: &mut String|
                             -> Option<MessageAction> {
                                let mut act: Option<MessageAction> = None;
                                ui.horizontal(|ui| {
                                    let role_resp = ui.colored_label(color, format!("{prefix}:"));
                                    if let Some(ts) = created_at {
                                        role_resp.on_hover_text(format_timestamp(ts));
                                    }
                                    // Sibling navigator (◀ N/M ▶) — only when
                                    // this message has alternate branches.
                                    if let Some((pos, total)) = sib {
                                        let prev_enabled = pos > 0;
                                        let next_enabled = pos + 1 < total;
                                        if ui
                                            .add_enabled(
                                                prev_enabled,
                                                egui::Button::new("◀").small(),
                                            )
                                            .on_hover_text("Previous branch")
                                            .clicked()
                                        {
                                            act = Some(MessageAction::PrevSibling(i));
                                        }
                                        ui.weak(format!("{}/{}", pos + 1, total));
                                        if ui
                                            .add_enabled(
                                                next_enabled,
                                                egui::Button::new("▶").small(),
                                            )
                                            .on_hover_text("Next branch")
                                            .clicked()
                                        {
                                            act = Some(MessageAction::NextSibling(i));
                                        }
                                    }
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui.small_button("📋").on_hover_text("Copy").clicked() {
                                                act = Some(MessageAction::Copy(i));
                                            }
                                            if role == Role::User
                                                && !streaming
                                                && ui.small_button("✏").on_hover_text("Edit").clicked()
                                            {
                                                act = Some(MessageAction::StartEdit(i));
                                            }
                                            if role == Role::Assistant
                                                && i == msg_count - 1
                                                && !streaming
                                                && ui
                                                    .small_button("🔄")
                                                    .on_hover_text("Regenerate")
                                                    .clicked()
                                            {
                                                act = Some(MessageAction::Regenerate);
                                            }
                                        },
                                    );
                                });

                                if editing_message == Some(i) {
                                    ui.add(
                                        egui::TextEdit::multiline(edit_buffer)
                                            .desired_width(f32::INFINITY)
                                            .desired_rows(3),
                                    );
                                    ui.horizontal(|ui| {
                                        if ui.button("Send").clicked() {
                                            act = Some(MessageAction::FinishEdit(i));
                                        }
                                        if ui.button("Cancel").clicked() {
                                            act = Some(MessageAction::CancelEdit);
                                        }
                                    });
                                } else if role == Role::Assistant && !content_clone.is_empty() {
                                    render_assistant_message(ui, cache, i, &content_clone);
                                } else if role == Role::Tool {
                                    render_tool_result(ui, i, &content_clone);
                                } else {
                                    ui.label(&content_clone);
                                }
                                // Tool calls (assistant turns that invoked
                                // one or more tools) render as a compact
                                // collapsible block beneath the assistant
                                // text.
                                if let Some(calls) = tool_calls_snap.as_ref()
                                    && role == Role::Assistant
                                    && !calls.is_empty()
                                {
                                    render_tool_calls(ui, i, calls);
                                }
                                act
                            };

                            let act_for_msg = if matches_find {
                                let inner_resp = egui::Frame::new()
                                    .fill(highlight_fill)
                                    .inner_margin(egui::Margin::same(4))
                                    .corner_radius(4.0)
                                    .show(ui, |ui| {
                                        render_inner(
                                            ui,
                                            &mut self.commonmark_cache,
                                            &mut self.edit_buffer,
                                        )
                                    });
                                // Only scroll once per query — repeated calls
                                // every frame would prevent the user from
                                // scrolling past the first match.
                                let already_scrolled = self
                                    .last_scrolled_find_query
                                    .as_deref()
                                    == Some(self.find_query.as_str());
                                if Some(i) == first_match_idx && !already_scrolled {
                                    inner_resp.response.scroll_to_me(Some(egui::Align::Center));
                                    pending_scroll_query = Some(self.find_query.clone());
                                }
                                inner_resp.inner
                            } else {
                                render_inner(
                                    ui,
                                    &mut self.commonmark_cache,
                                    &mut self.edit_buffer,
                                )
                            };

                            if act_for_msg.is_some() {
                                action = act_for_msg;
                            }

                            ui.add_space(8.0);
                            ui.separator();
                        }

                        if let Some(q) = pending_scroll_query {
                            self.last_scrolled_find_query = Some(q);
                        }

                        match action {
                            Some(MessageAction::Copy(i)) if i < self.messages.len() => {
                                self.copy_to_clipboard(&self.messages[i].text_str());
                            }
                            Some(MessageAction::StartEdit(i)) if i < self.messages.len() => {
                                self.edit_buffer = self.messages[i].text_str();
                                self.editing_message = Some(i);
                            }
                            Some(MessageAction::FinishEdit(i)) => {
                                self.edit_and_resend(i);
                            }
                            Some(MessageAction::CancelEdit) => {
                                self.editing_message = None;
                                self.edit_buffer.clear();
                            }
                            Some(MessageAction::Regenerate) => {
                                self.regenerate();
                            }
                            Some(MessageAction::PrevSibling(i)) => {
                                self.navigate_sibling(i, -1);
                            }
                            Some(MessageAction::NextSibling(i)) => {
                                self.navigate_sibling(i, 1);
                            }
                            _ => {}
                        }
                    });
            });

        if let Some(p) = prefill_input {
            self.input = p;
        }

        // Toast (auto-dismisses after 3.5s)
        let toast_expired = self
            .toast
            .as_ref()
            .map(|(_, started)| started.elapsed() >= std::time::Duration::from_millis(3500))
            .unwrap_or(false);
        if toast_expired {
            self.toast = None;
        }
        if let Some((msg, _)) = self.toast.as_ref() {
            egui::Area::new(egui::Id::new("hchat_toast"))
                .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -16.0))
                .show(&ctx, |ui| {
                    egui::Frame::popup(ui.style())
                        .inner_margin(egui::Margin::symmetric(12, 8))
                        .show(ui, |ui| {
                            // Cap width so a multi-line toast (like /help) stays
                            // narrow rather than stretching across the screen.
                            ui.set_max_width(360.0);
                            ui.label(msg);
                        });
                });
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }

        // Per-frame: persist any per-conversation setting changes the user
        // made via the settings panel (or other widgets). Cheap; the dirty
        // check short-circuits when nothing's moved.
        self.persist_settings_if_dirty();

        // Debounced draft save. Need to repaint when the debounce timer is
        // due, otherwise egui sleeps and the draft never lands.
        if let Some(remaining) = self.persist_draft_if_idle() {
            ctx.request_repaint_after(remaining);
        }
    }
}

/// Hard cap on tool-then-restream cycles per user turn. A model that keeps
/// calling tools forever will hit this and stop, surfacing a toast. Eight
/// is empirically generous — typical agentic flows (read → grep → read →
/// edit → run tests → read again → done) come in under that.
const MAX_TOOL_ITERATIONS: u32 = 8;

/// User decision on a parked tool approval card. `ApproveAll` adds the
/// tool name to a per-session allowlist so future calls of the same tool
/// in this conversation don't prompt again.
enum ApprovalDecision {
    Approve,
    ApproveAll,
    Reject,
}

enum MessageAction {
    Copy(usize),
    StartEdit(usize),
    FinishEdit(usize),
    CancelEdit,
    Regenerate,
    /// Navigate to the previous (or next) sibling at this branch point. The
    /// `usize` is the message index in `self.messages`; the handler queries
    /// the DB for siblings, picks the appropriate one, and re-walks the
    /// active path from there.
    PrevSibling(usize),
    NextSibling(usize),
}

enum SidebarAction {
    Load(i64),
    Delete(i64),
    Export(i64),
    StartRename(i64),
    FinishRename(i64),
    TogglePin(i64),
}

/// MIME type for an image file extension, or None if not a supported image.
/// Limited to the formats vision-capable APIs accept (OpenAI, OpenRouter,
/// Anthropic-via-proxy). HEIC etc. would need decoding.
fn image_mime_for_ext(ext: &str) -> Option<&'static str> {
    match ext {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        _ => None,
    }
}

/// True if the file extension looks like plain text we can safely inline as a
/// fenced code block. Conservative on purpose — anything binary-ish is
/// excluded so we don't dump random bytes into the chat.
fn is_text_extension(ext: &str) -> bool {
    matches!(
        ext,
        "txt"
            | "md"
            | "markdown"
            | "rs"
            | "py"
            | "js"
            | "jsx"
            | "ts"
            | "tsx"
            | "json"
            | "yaml"
            | "yml"
            | "toml"
            | "html"
            | "css"
            | "scss"
            | "go"
            | "java"
            | "kt"
            | "kts"
            | "swift"
            | "c"
            | "h"
            | "cc"
            | "cpp"
            | "hpp"
            | "rb"
            | "sh"
            | "bash"
            | "zsh"
            | "fish"
            | "sql"
            | "lua"
            | "php"
            | "pl"
            | "r"
            | "scala"
            | "ex"
            | "exs"
            | "elm"
            | "hs"
            | "clj"
            | "cs"
            | "fs"
            | "vue"
            | "svelte"
            | "xml"
            | "csv"
            | "tsv"
            | "log"
            | "ini"
            | "conf"
            // `env` deliberately excluded — `secrets.env` / `prod.env` etc.
            // commonly hold credentials. Don't auto-inline them into the chat.
            | "diff"
            | "patch"
    )
}

/// Best-effort markdown fence language for an extension. Returns the
/// extension itself when there's no remap, so unknown extensions still get
/// a (sometimes-syntactic-highlight-friendly) tag rather than a bare fence.
fn lang_for_ext(ext: &str) -> String {
    match ext {
        "rs" => "rust".to_string(),
        "py" => "python".to_string(),
        "js" | "jsx" => "javascript".to_string(),
        "ts" | "tsx" => "typescript".to_string(),
        "rb" => "ruby".to_string(),
        "sh" | "bash" => "bash".to_string(),
        "zsh" | "fish" => "sh".to_string(),
        "md" | "markdown" => "markdown".to_string(),
        "yml" => "yaml".to_string(),
        "kt" | "kts" => "kotlin".to_string(),
        "cs" => "csharp".to_string(),
        "fs" => "fsharp".to_string(),
        "ex" | "exs" => "elixir".to_string(),
        "hs" => "haskell".to_string(),
        "cc" | "cpp" | "hpp" => "cpp".to_string(),
        other => other.to_string(),
    }
}

/// Trim quotes, trailing punctuation, and whitespace from an LLM-generated
/// title. Models occasionally wrap their reply in quotes or add a trailing
/// period despite instructions; clean those up rather than store them
/// verbatim. Hard cap at 80 chars to keep the sidebar tidy.
fn sanitize_title(s: &str) -> String {
    let trimmed = s.trim();
    // Strip a single pair of surrounding quotes.
    let unquoted = if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        &trimmed[1..trimmed.len().saturating_sub(1)]
    } else {
        trimmed
    };
    let no_trailing = unquoted.trim_end_matches(['.', '!', '?', ',', ';']);
    let collapsed: String = no_trailing.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() > 80 {
        collapsed.chars().take(80).collect::<String>() + "…"
    } else {
        collapsed
    }
}

/// Default working directory for tool calls in a fresh conversation. Falls
/// back to `.` if the home dir lookup somehow fails — we still want a
/// usable path rather than crashing on an unwrap.
fn default_working_dir() -> String {
    dirs::home_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string())
}

/// Hash of the active path's structural identity (message ids + count).
/// When this signature is stable, the ◀ N/M ▶ navigator can reuse its
/// cached `sibling_info` without re-querying the DB. Streaming token
/// appends don't change ids, so they don't force a rebuild.
fn sibling_signature(messages: &[Message]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    messages.len().hash(&mut h);
    for m in messages {
        m.id.hash(&mut h);
    }
    h.finish()
}

/// Hash of the live per-conversation settings. Fed to a frame-by-frame dirty
/// check so settings persist whenever the user moves a slider, edits the
/// system prompt, etc., without a separate "save settings" button.
fn settings_signature(s: &ConversationSettings) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.model.hash(&mut h);
    s.system_prompt.hash(&mut h);
    // f32 isn't Hash; fold the bits.
    s.temperature.map(|v| v.to_bits()).hash(&mut h);
    s.max_tokens.hash(&mut h);
    s.use_max_tokens.hash(&mut h);
    s.top_p.map(|v| v.to_bits()).hash(&mut h);
    s.frequency_penalty.map(|v| v.to_bits()).hash(&mut h);
    s.presence_penalty.map(|v| v.to_bits()).hash(&mut h);
    s.stop_sequences.hash(&mut h);
    s.endpoint.hash(&mut h);
    s.working_dir.hash(&mut h);
    h.finish()
}

/// Cheap content signature for the token-estimate cache. Combines lengths and
/// a tail hash of each message; collisions only matter as missed updates and
/// any meaningful content change flips at least one byte at the tail.
fn token_signature(system: &str, messages: &[Message], pending: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    system.len().hash(&mut h);
    pending.len().hash(&mut h);
    pending.hash(&mut h);
    messages.len().hash(&mut h);
    for m in messages {
        let s = m.text_str();
        s.len().hash(&mut h);
        // Hash a small fingerprint of each message — full content hash would
        // be slower than just re-tokenizing, defeating the cache.
        let bytes = s.as_bytes();
        if !bytes.is_empty() {
            bytes[bytes.len() - 1].hash(&mut h);
            if bytes.len() >= 16 {
                bytes[bytes.len() - 16].hash(&mut h);
            }
        }
        // Image attachments invalidate the cache too — token cost depends on
        // the number and detail of image parts, not just text.
        m.images().count().hash(&mut h);
    }
    h.finish()
}

/// Token estimate using `cl100k_base` (the GPT-4 / GPT-3.5 tokenizer). It's an
/// approximation for non-OpenAI models (Claude, Llama, etc.) but the right
/// order of magnitude — accurate enough to drive a budget bar.
///
/// Concatenation order matches what we'd actually send: system prompt → each
/// message → pending input. We don't add per-message framing overhead the API
/// adds (ChatML tokens etc.), so the real prompt tokens will be a few percent
/// higher than this estimate.
fn estimate_tokens(system: &str, messages: &[Message], pending: &str) -> usize {
    let texts: Vec<String> = messages.iter().map(|m| m.text_str()).collect();
    let mut full = String::with_capacity(
        system.len() + pending.len() + texts.iter().map(|s| s.len() + 1).sum::<usize>(),
    );
    if !system.is_empty() {
        full.push_str(system);
        full.push('\n');
    }
    for s in &texts {
        full.push_str(s);
        full.push('\n');
    }
    full.push_str(pending);

    let bpe = tiktoken_rs::cl100k_base_singleton();
    bpe.encode_with_special_tokens(&full).len()
}

/// Truncate a string to at most `max_chars` characters and append `…`. Returns
/// the original string if it's already short enough. Char-aware so we never
/// slice through a multi-byte UTF-8 boundary.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_chars).collect();
        out.push('…');
        out
    }
}

/// Format a Unix-millis timestamp as `HH:MM:SS UTC`. Local-timezone formatting
/// would need chrono or similar; for Phase 1 hover tooltips, UTC is honest and
/// sufficient. Phase 2's schema migration adds proper persisted timestamps and
/// can move to localized formatting then.
fn format_timestamp(ms: i64) -> String {
    let secs = ms / 1000;
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02} UTC")
}

/// Render the tool_calls block under an assistant message. Each call gets
/// a collapsing header showing `name(args summary)`; opening it shows the
/// pretty-printed JSON arguments. Kept compact so a multi-call turn doesn't
/// dominate the scroll.
fn render_tool_calls(ui: &mut egui::Ui, msg_idx: usize, calls: &[crate::message::ToolCall]) {
    for (j, call) in calls.iter().enumerate() {
        // Short summary for the header: tool name + args truncated to a
        // single short line. The full pretty-printed JSON is in the body.
        let arg_summary = single_line_summary(&call.function.arguments, 60);
        let header = format!("🔧 {}({})", call.function.name, arg_summary);
        egui::CollapsingHeader::new(header)
            .id_salt(("tool_call", msg_idx, j))
            .default_open(false)
            .show(ui, |ui| {
                let pretty = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                    .and_then(|v| serde_json::to_string_pretty(&v))
                    .unwrap_or_else(|_| call.function.arguments.clone());
                ui.add(
                    egui::TextEdit::multiline(&mut pretty.as_str())
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .interactive(false),
                );
            });
    }
}

/// Render a Role::Tool message body as a collapsible "Tool result" block.
/// Closed by default — tool output (file dumps, shell stdout) is verbose
/// and shouldn't clutter the chat scroll.
fn render_tool_result(ui: &mut egui::Ui, msg_idx: usize, body: &str) {
    let preview = single_line_summary(body, 80);
    let header = format!("📤 Tool result — {preview}");
    egui::CollapsingHeader::new(header)
        .id_salt(("tool_result", msg_idx))
        .default_open(false)
        .show(ui, |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut body.to_string().as_str())
                    .code_editor()
                    .desired_width(f32::INFINITY)
                    .interactive(false),
            );
        });
}

/// Strip newlines + collapse whitespace + truncate. Used for the one-line
/// summary in tool-call / tool-result headers so they stay tidy.
fn single_line_summary(s: &str, max: usize) -> String {
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() > max {
        let truncated: String = collapsed.chars().take(max).collect();
        format!("{truncated}…")
    } else {
        collapsed
    }
}

/// Render an assistant message: walk the segment list, rendering prose with
/// `egui_commonmark` and code/reasoning blocks ourselves.
fn render_assistant_message(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    msg_idx: usize,
    content: &str,
) {
    for (block_idx, seg) in markdown::segments(content).into_iter().enumerate() {
        match seg {
            Segment::Markdown(text) => {
                CommonMarkViewer::new().show(ui, cache, text);
            }
            Segment::Code { lang, body } => {
                // Each code block gets its own id scope so two blocks with
                // identical bodies don't collide on auto-generated widget ids.
                ui.push_id(("code", msg_idx, block_idx), |ui| {
                    render_code_block(ui, lang, body);
                });
            }
            Segment::Reasoning { body, closed } => {
                render_reasoning_block(ui, msg_idx, block_idx, body, closed);
            }
        }
    }
}

fn render_code_block(ui: &mut egui::Ui, lang: &str, body: &str) {
    let visuals = ui.visuals();
    let bg = visuals.code_bg_color;
    let stroke_color = visuals.weak_text_color();

    egui::Frame::new()
        .fill(bg)
        .stroke(egui::Stroke::new(1.0, stroke_color))
        .inner_margin(egui::Margin::same(8))
        .corner_radius(4.0)
        .show(ui, |ui| {
            // Header strip: language pill (left) + copy button (right).
            ui.horizontal(|ui| {
                if !lang.is_empty() {
                    ui.label(
                        egui::RichText::new(lang)
                            .small()
                            .color(ui.visuals().weak_text_color())
                            .monospace(),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("📋")
                        .on_hover_text("Copy code")
                        .clicked()
                        && let Ok(mut cb) = arboard::Clipboard::new()
                    {
                        let _ = cb.set_text(body);
                    }
                });
            });
            // Use a selectable Label rather than a non-interactive TextEdit so
            // users can drag-select a snippet for partial copy. Label respects
            // newlines, so the visual layout of code is preserved.
            ui.add(
                egui::Label::new(egui::RichText::new(body).monospace())
                    .selectable(true)
                    .wrap_mode(egui::TextWrapMode::Extend),
            );
        });
}

fn render_reasoning_block(
    ui: &mut egui::Ui,
    msg_idx: usize,
    block_idx: usize,
    body: &str,
    closed: bool,
) {
    let label = if closed {
        "💭 Reasoning"
    } else {
        "💭 Reasoning (streaming…)"
    };

    // Include `closed` in the id salt so the CollapsingHeader's persisted
    // open/closed state resets when the block transitions from streaming to
    // finished — that lets `default_open(!closed)` actually apply on the
    // transition (egui only honors `default_open` for unseen ids). Trade-off:
    // if the user manually expanded it while streaming, that gets reset on
    // close — which is the behavior we want anyway (auto-collapse when done).
    egui::CollapsingHeader::new(
        egui::RichText::new(label)
            .small()
            .color(ui.visuals().weak_text_color()),
    )
    .id_salt(("reasoning", msg_idx, block_idx, closed))
    .default_open(!closed)
    .show(ui, |ui| {
        ui.label(
            egui::RichText::new(body)
                .small()
                .color(ui.visuals().weak_text_color()),
        );
    });
}
