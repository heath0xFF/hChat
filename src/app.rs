use crate::api::{self, ChatParams, StreamEvent, Usage};
use crate::config::{Config, Endpoint};
use crate::message::{Message, Role};
use crate::storage::Storage;
use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

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
    // Rename
    renaming_conversation: Option<i64>,
    rename_buffer: String,
    // Config
    config: Config,
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
            renaming_conversation: None,
            rename_buffer: String::new(),
            config,
        };

        app.fetch_models();
        app
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
        let content = self.input.trim().to_string();
        if content.is_empty() || self.streaming || self.models.is_empty() {
            return;
        }

        self.input.clear();
        self.error = None;

        self.messages.push(Message {
            role: Role::User,
            content,
        });

        self.start_streaming();
    }

    fn start_streaming(&mut self) {
        if self.models.is_empty() {
            self.error = Some("No models available".to_string());
            return;
        }

        self.messages.push(Message {
            role: Role::Assistant,
            content: String::new(),
        });

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
            messages.push(Message {
                role: Role::System,
                content: self.system_prompt.clone(),
            });
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

        let params = ChatParams {
            base_url,
            model,
            messages,
            temperature,
            max_tokens,
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
                    StreamEvent::ModelsLoaded(models) => {
                        if !models.is_empty() {
                            self.models = models;
                            self.selected_model =
                                self.selected_model.min(self.models.len().saturating_sub(1));
                        }
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
                    StreamEvent::Token(token) => {
                        if let Some(last) = self.messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            last.content.push_str(&token);
                        }
                    }
                    StreamEvent::UsageInfo(usage) => {
                        self.last_usage = Some(usage);
                    }
                    StreamEvent::Done => {
                        just_finished_streaming = true;
                        self.streaming = false;
                        self.cancel_token = None;
                        self.rx = None;
                        break;
                    }
                    StreamEvent::Error(e) => {
                        self.error = Some(e);
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

        // Bottom input panel
        egui::Panel::bottom("input_panel").show_inside(ui, |ui| {
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, format!("Error: {err}"));
            }

            let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
            let shift_held = ui.input(|i| i.modifiers.shift);

            ui.horizontal(|ui| {
                let response = ui.add_sized(
                    [ui.available_width() - 60.0, 35.0],
                    egui::TextEdit::multiline(&mut self.input)
                        .hint_text("Type a message... (Shift+Enter for newline)")
                        .desired_rows(1),
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

            ui.horizontal(|ui| {
                if self.streaming {
                    ui.spinner();
                }
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

                    ui.label(egui::RichText::new("Cost Estimate").strong());
                    
                    // Simple heuristic to guess if the endpoint is paid vs local
                    let is_paid_endpoint = self.base_url.contains("openrouter.ai") || self.base_url.contains("api.openai.com") || self.base_url.contains("api.anthropic.com");
                    
                    if is_paid_endpoint {
                        // We could use an exact mapping, but for now we'll estimate generic cost
                        // based on $0.01 per 1k context tokens
                        let estimated_cost = (approx_tokens as f64 / 1000.0) * 0.01;
                        ui.label(format!("~${:.4} (context)", estimated_cost));
                    } else {
                        ui.label("Local models: $0.00");
                    }
                    ui.add_space(8.0);

                    if let Some(usage) = &self.last_usage {
                        ui.separator();
                        ui.label(egui::RichText::new("Last API Response").strong());
                        if let Some(p) = usage.prompt_tokens {
                            ui.label(format!("Prompt: {}", p));
                        }
                        if let Some(c) = usage.completion_tokens {
                            ui.label(format!("Completion: {}", c));
                        }
                        if let Some(t) = usage.total_tokens {
                            ui.label(format!("Total: {}", t));
                        }
                        if let Some(cost) = usage.total_cost {
                            ui.label(format!("Cost: ${:.4}", cost));
                        }
                    }
                });
        }

        // Central message area
        let panel_fill = ui.visuals().panel_fill;
        egui::Frame::new()
            .fill(panel_fill)
            .inner_margin(egui::Margin::symmetric(16, 0))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if self.messages.is_empty() {
                            ui.vertical_centered(|ui| {
                                ui.add_space(100.0);
                                ui.heading("hChat");
                                ui.label("Lightweight local LLM chat");
                                ui.label("Ctrl+N to start a new conversation");
                            });
                            return;
                        }

                        let mut action: Option<MessageAction> = None;
                        let msg_count = self.messages.len();

                        for (i, msg) in self.messages.iter().enumerate() {
                            let (prefix, color) = match msg.role {
                                Role::User => ("You", egui::Color32::from_rgb(100, 180, 255)),
                                Role::Assistant => ("AI", egui::Color32::from_rgb(100, 255, 150)),
                                Role::System => ("System", egui::Color32::GRAY),
                            };

                            ui.horizontal(|ui| {
                                ui.colored_label(color, format!("{prefix}:"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.small_button("📋").on_hover_text("Copy").clicked() {
                                            action = Some(MessageAction::Copy(i));
                                        }
                                        if msg.role == Role::User
                                            && !self.streaming
                                            && ui.small_button("✏").on_hover_text("Edit").clicked()
                                        {
                                            action = Some(MessageAction::StartEdit(i));
                                        }
                                        if msg.role == Role::Assistant
                                            && i == msg_count - 1
                                            && !self.streaming
                                            && ui
                                                .small_button("🔄")
                                                .on_hover_text("Regenerate")
                                                .clicked()
                                        {
                                            action = Some(MessageAction::Regenerate);
                                        }
                                    },
                                );
                            });

                            if self.editing_message == Some(i) {
                                ui.add(
                                    egui::TextEdit::multiline(&mut self.edit_buffer)
                                        .desired_width(f32::INFINITY)
                                        .desired_rows(3),
                                );
                                ui.horizontal(|ui| {
                                    if ui.button("Send").clicked() {
                                        action = Some(MessageAction::FinishEdit(i));
                                    }
                                    if ui.button("Cancel").clicked() {
                                        action = Some(MessageAction::CancelEdit);
                                    }
                                });
                            } else if msg.role == Role::Assistant && !msg.content.is_empty() {
                                CommonMarkViewer::new().show(
                                    ui,
                                    &mut self.commonmark_cache,
                                    &msg.content,
                                );
                            } else {
                                ui.label(&msg.content);
                            }

                            ui.add_space(8.0);
                            ui.separator();
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
