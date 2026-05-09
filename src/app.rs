use crate::api::{self, ChatParams, StreamEvent, Usage};
use crate::commands::{self, Command, ParseResult};
use crate::config::{Config, Endpoint};
use crate::markdown::{self, Segment};
use crate::message::{Message, Role};
use crate::storage::Storage;
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
    // Persistence
    storage: Storage,
    current_conversation_id: Option<i64>,
    conversation_list: Vec<(i64, String)>,
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
            .map(|c| (c.id, c.title))
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
            storage,
            current_conversation_id: None,
            conversation_list,
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
            config,
        };

        app.fetch_models();
        app
    }

    fn show_toast(&mut self, msg: impl Into<String>) {
        self.toast = Some((msg.into(), std::time::Instant::now()));
    }

    fn fetch_models(&mut self) {
        if self.models_loading {
            return; // fetch already in progress
        }
        let base_url = self.base_url.clone();
        let api_key = self.current_api_key().map(|s| s.to_string());
        let (tx, rx) = mpsc::unbounded_channel();

        self.runtime.spawn(async move {
            match api::fetch_models(&base_url, api_key.as_deref()).await {
                Ok(models) => {
                    let _ = tx.send(StreamEvent::ModelsLoaded(models));
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
        self.messages.clear();
        self.current_conversation_id = None;
        self.error = None;
        self.last_usage = None;
        self.session_cost = 0.0;
    }

    fn save_current(&mut self) {
        if self.messages.is_empty() {
            return;
        }

        match self.current_conversation_id {
            Some(id) => {
                self.storage.save_messages(id, &self.messages);
            }
            None => {
                let title = self
                    .messages
                    .iter()
                    .find(|m| m.role == Role::User)
                    .map(|m| {
                        let t = m.content.trim();
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

                let id = self.storage.create_conversation(&title);
                self.storage.save_messages(id, &self.messages);
                self.current_conversation_id = Some(id);
            }
        };

        self.refresh_conversation_list();
    }

    fn load_conversation(&mut self, id: i64) {
        if self.streaming {
            self.stop_streaming();
        }
        self.save_current();
        self.messages = self.storage.load_messages(id);
        self.current_conversation_id = Some(id);
        self.error = None;
        self.last_usage = None;
        self.session_cost = 0.0;
        self.editing_message = None;
        self.edit_buffer.clear();
    }

    fn delete_conversation(&mut self, id: i64) {
        self.storage.delete_conversation(id);
        if self.current_conversation_id == Some(id) {
            if self.streaming {
                self.stop_streaming();
            }
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
            .map(|c| (c.id, c.title))
            .collect();
    }

    fn send_message(&mut self) {
        let raw = self.input.trim().to_string();
        if raw.is_empty() || self.streaming {
            return;
        }

        // Slash commands intercept before model availability check, so /help
        // and /clear work even when no models are loaded.
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

        if self.models.is_empty() {
            self.error = Some("No models available".to_string());
            return;
        }

        self.input.clear();
        self.error = None;

        self.messages.push(Message::new(Role::User, raw));
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
                    .find(|m| m.role == Role::Assistant && !m.content.is_empty())
                {
                    self.copy_to_clipboard(&last.content);
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

        self.messages
            .push(Message::new(Role::Assistant, String::new()));
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
            messages.push(Message::new(Role::System, self.system_prompt.clone()));
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
            && last.content.is_empty()
        {
            self.messages.pop();
        }
    }

    fn regenerate(&mut self) {
        if self.streaming || self.models.is_empty() {
            return;
        }
        if let Some(last) = self.messages.last()
            && last.role == Role::Assistant
        {
            self.messages.pop();
            self.start_streaming();
        }
    }

    fn edit_and_resend(&mut self, index: usize) {
        if self.streaming || index >= self.messages.len() {
            return;
        }
        let content = self.edit_buffer.trim().to_string();
        if content.is_empty() {
            return;
        }

        self.messages[index].content = content;
        self.messages.truncate(index + 1);
        self.editing_message = None;
        self.edit_buffer.clear();

        self.start_streaming();
    }

    fn copy_to_clipboard(&self, text: &str) {
        if let Ok(mut clipboard) = arboard::Clipboard::new() {
            let _ = clipboard.set_text(text);
        }
    }

    fn process_events(&mut self) {
        // Process model fetch events (separate channel)
        if let Some(rx) = &mut self.models_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    StreamEvent::ModelsLoaded(models) if !models.is_empty() => {
                        self.models = models;
                        self.selected_model =
                            self.selected_model.min(self.models.len().saturating_sub(1));
                    }
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
        }

        // Process chat stream events
        let mut just_finished_streaming = false;

        if let Some(rx) = &mut self.rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    StreamEvent::Reasoning(text) => {
                        if let Some(last) = self.messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            if !self.reasoning_open {
                                last.content.push_str("<think>\n");
                                self.reasoning_open = true;
                            }
                            last.content.push_str(&text);
                        }
                    }
                    StreamEvent::Token(token) => {
                        if let Some(last) = self.messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            if self.reasoning_open {
                                last.content.push_str("\n</think>\n\n");
                                self.reasoning_open = false;
                            }
                            last.content.push_str(&token);
                        }
                    }
                    StreamEvent::UsageInfo(usage) => {
                        if let Some(cost) = usage.cost {
                            self.session_cost += cost;
                        }
                        self.last_usage = Some(usage);
                    }
                    StreamEvent::Done => {
                        // If reasoning was streaming and the provider never sent
                        // a content delta, close the <think> tag explicitly so
                        // the saved message has a balanced block.
                        if self.reasoning_open
                            && let Some(last) = self.messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            last.content.push_str("\n</think>\n");
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
                            last.content.push_str("\n</think>\n");
                        }
                        self.reasoning_open = false;
                        self.streaming = false;
                        self.cancel_token = None;
                        self.rx = None;
                        if let Some(last) = self.messages.last()
                            && last.role == Role::Assistant
                            && last.content.is_empty()
                        {
                            self.messages.pop();
                        }
                        break;
                    }
                    _ => {}
                }
            }
        }

        if just_finished_streaming && !self.messages.is_empty() {
            self.save_current();
        }
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

        if self.streaming || self.models_loading {
            ctx.request_repaint();
        }

        // Keyboard shortcuts
        let new_chat =
            ui.input(|i| i.key_pressed(egui::Key::N) && (i.modifiers.ctrl || i.modifiers.command));
        if new_chat {
            self.new_conversation();
        }
        let toggle_find =
            ui.input(|i| i.key_pressed(egui::Key::F) && (i.modifiers.ctrl || i.modifiers.command));
        if toggle_find {
            self.show_find = !self.show_find;
            if !self.show_find {
                self.find_query.clear();
                self.last_scrolled_find_query = None;
            }
        }
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
                });
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

                        for (id, title) in &self.conversation_list {
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
                                let resp = title_ui.add(
                                    egui::Label::new(egui::RichText::new(title.as_str()).color(
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
                                if let Some((_, title)) =
                                    self.conversation_list.iter().find(|(cid, _)| *cid == id)
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
                        char_count += msg.content.len();
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
                            full_text.push_str(&m.content);
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
                        .map(|m| m.content.to_ascii_lowercase().contains(q))
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

                        // We iterate by index to avoid holding a borrow on self.messages
                        // across the body, which needs &mut self for commonmark_cache.
                        for i in 0..msg_count {
                            let (role, content_clone, created_at);
                            {
                                let msg = &self.messages[i];
                                role = msg.role.clone();
                                content_clone = msg.content.clone();
                                created_at = msg.created_at;
                            }
                            let (prefix, color) = match role {
                                Role::User => ("You", egui::Color32::from_rgb(100, 180, 255)),
                                Role::Assistant => ("AI", egui::Color32::from_rgb(100, 255, 150)),
                                Role::System => ("System", egui::Color32::GRAY),
                            };

                            let matches_find = match_flags.get(i).copied().unwrap_or(false);

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
                                } else {
                                    ui.label(&content_clone);
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
                                self.copy_to_clipboard(&self.messages[i].content);
                            }
                            Some(MessageAction::StartEdit(i)) if i < self.messages.len() => {
                                self.edit_buffer = self.messages[i].content.clone();
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
    }
}

enum MessageAction {
    Copy(usize),
    StartEdit(usize),
    FinishEdit(usize),
    CancelEdit,
    Regenerate,
}

enum SidebarAction {
    Load(i64),
    Delete(i64),
    Export(i64),
    StartRename(i64),
    FinishRename(i64),
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
        m.content.len().hash(&mut h);
        // Hash a small fingerprint of each message — full content hash would
        // be slower than just re-tokenizing, defeating the cache.
        let bytes = m.content.as_bytes();
        if !bytes.is_empty() {
            bytes[bytes.len() - 1].hash(&mut h);
            if bytes.len() >= 16 {
                bytes[bytes.len() - 16].hash(&mut h);
            }
        }
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
    let mut full = String::with_capacity(
        system.len()
            + pending.len()
            + messages.iter().map(|m| m.content.len() + 1).sum::<usize>(),
    );
    if !system.is_empty() {
        full.push_str(system);
        full.push('\n');
    }
    for m in messages {
        full.push_str(&m.content);
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
