use crate::api::{self, StreamEvent};
use crate::message::{Message, Role};
use eframe::egui;
use tokio::sync::mpsc;

pub struct ChatApp {
    messages: Vec<Message>,
    input: String,
    base_url: String,
    models: Vec<String>,
    selected_model: usize,
    streaming: bool,
    rx: Option<mpsc::UnboundedReceiver<StreamEvent>>,
    runtime: tokio::runtime::Runtime,
    models_loading: bool,
    error: Option<String>,
}

impl ChatApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_fonts(&cc.egui_ctx);

        let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        let base_url = "http://localhost:11434/v1".to_string();

        let mut app = Self {
            messages: Vec::new(),
            input: String::new(),
            base_url,
            models: vec!["loading...".to_string()],
            selected_model: 0,
            streaming: false,
            rx: None,
            runtime,
            models_loading: true,
            error: None,
        };

        app.fetch_models();
        app
    }

    fn fetch_models(&mut self) {
        let base_url = self.base_url.clone();
        let (tx, rx) = std::sync::mpsc::channel();

        self.runtime.spawn(async move {
            let result = api::fetch_models(&base_url).await;
            let _ = tx.send(result);
        });

        // Check synchronously (non-blocking) in update loop via models_loading flag
        // Store the receiver to check later
        self.models_loading = true;
        std::thread::spawn(move || {
            // This is handled in the update loop via the models_loading state
            drop(rx);
        });

        // Actually, let's use a simpler approach: spawn and use a channel
        let base_url = self.base_url.clone();
        let (tx, rx) = mpsc::unbounded_channel();

        self.runtime.spawn(async move {
            match api::fetch_models(&base_url).await {
                Ok(models) => {
                    let _ = tx.send(StreamEvent::Token(serde_json::to_string(&models).unwrap()));
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(e));
                }
            }
            let _ = tx.send(StreamEvent::Done);
        });

        self.rx = Some(rx);
    }

    fn send_message(&mut self) {
        let content = self.input.trim().to_string();
        if content.is_empty() || self.streaming {
            return;
        }

        self.input.clear();
        self.error = None;

        self.messages.push(Message {
            role: Role::User,
            content,
        });

        // Add empty assistant message to fill via streaming
        self.messages.push(Message {
            role: Role::Assistant,
            content: String::new(),
        });

        let (tx, rx) = mpsc::unbounded_channel();
        self.rx = Some(rx);
        self.streaming = true;

        let model = self.models[self.selected_model].clone();
        let base_url = self.base_url.clone();
        // Send all messages except the empty assistant one
        let messages: Vec<Message> = self.messages[..self.messages.len() - 1].to_vec();

        self.runtime
            .spawn(async move { api::stream_chat(base_url, model, messages, tx) });
    }

    fn process_events(&mut self) {
        if let Some(rx) = &mut self.rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    StreamEvent::Token(token) => {
                        if self.models_loading {
                            // This is a model list response
                            if let Ok(models) = serde_json::from_str::<Vec<String>>(&token) {
                                if !models.is_empty() {
                                    self.models = models;
                                    self.selected_model = 0;
                                }
                            }
                        } else if let Some(last) = self.messages.last_mut() {
                            last.content.push_str(&token);
                        }
                    }
                    StreamEvent::Done => {
                        if self.models_loading {
                            self.models_loading = false;
                        }
                        self.streaming = false;
                        self.rx = None;
                        return;
                    }
                    StreamEvent::Error(e) => {
                        self.error = Some(e);
                        self.streaming = false;
                        self.models_loading = false;
                        self.rx = None;
                        // Remove empty assistant message on error
                        if let Some(last) = self.messages.last() {
                            if last.role == Role::Assistant && last.content.is_empty() {
                                self.messages.pop();
                            }
                        }
                        return;
                    }
                }
            }
        }
    }
}

fn configure_fonts(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(14.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(13.0, egui::FontFamily::Monospace),
    );
    ctx.set_style(style);
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_events();

        // Request repaint while streaming for live updates
        if self.streaming || self.models_loading {
            ctx.request_repaint();
        }

        // Top bar with model selector and endpoint
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Model:");
                egui::ComboBox::from_id_salt("model_selector")
                    .selected_text(&self.models[self.selected_model])
                    .show_ui(ui, |ui| {
                        for (i, model) in self.models.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_model, i, model);
                        }
                    });

                ui.separator();
                ui.label("Endpoint:");
                ui.add(egui::TextEdit::singleline(&mut self.base_url).desired_width(250.0));

                if ui.button("↻").on_hover_text("Refresh models").clicked() {
                    self.fetch_models();
                }
            });
        });

        // Bottom input panel
        egui::TopBottomPanel::bottom("input_panel").show(ctx, |ui| {
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, format!("Error: {err}"));
            }

            // Check for Enter key press before rendering the text edit,
            // so we can intercept it before egui inserts a newline.
            let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
            let modifier_held = ui.input(|i| i.modifiers.shift);

            ui.horizontal(|ui| {
                let response = ui.add_sized(
                    [ui.available_width() - 60.0, 35.0],
                    egui::TextEdit::multiline(&mut self.input)
                        .hint_text("Type a message... (Shift+Enter for newline)")
                        .desired_rows(1),
                );

                // Enter sends, Ctrl/Cmd+Enter inserts newline
                if enter_pressed && response.has_focus() {
                    if modifier_held {
                        self.input.push('\n');
                    } else {
                        // Remove the newline that egui already inserted
                        if self.input.ends_with('\n') {
                            self.input.pop();
                        }
                        self.send_message();
                    }
                }

                let send_enabled = !self.streaming && !self.input.trim().is_empty();
                if ui
                    .add_enabled(send_enabled, egui::Button::new("Send"))
                    .clicked()
                {
                    self.send_message();
                    response.request_focus();
                }
            });

            if self.streaming {
                ui.spinner();
            }
        });

        // Central message area
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if self.messages.is_empty() {
                        ui.vertical_centered(|ui| {
                            ui.add_space(100.0);
                            ui.heading("hChat");
                            ui.label("Lightweight local LLM chat");
                        });
                    }

                    for msg in &self.messages {
                        let (prefix, color) = match msg.role {
                            Role::User => ("You", egui::Color32::from_rgb(100, 180, 255)),
                            Role::Assistant => ("AI", egui::Color32::from_rgb(100, 255, 150)),
                            Role::System => ("System", egui::Color32::GRAY),
                        };

                        ui.horizontal_wrapped(|ui| {
                            ui.colored_label(color, format!("{prefix}:"));
                            ui.label(&msg.content);
                        });

                        ui.add_space(8.0);
                    }
                });
        });
    }
}
