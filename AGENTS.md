# hChat Agent Instructions

## Commands
- **Run (dev)**: `cargo run` (debug build, faster compile)
- **Run (release)**: `cargo run --release`
- **Build optimized**: `cargo build --release` (Uses `opt-level = "s"`, LTO, and symbol stripping)
- **Lint**: `cargo clippy -- -D warnings`
- **Format**: `cargo fmt`

## Architecture & Conventions
- **`src/main.rs`**: Entry point. Configures `eframe` window and launches `ChatApp`.
- **`src/app.rs`**: Core UI logic. `ChatApp` struct holds all state. Uses `egui` immediate-mode rendering. State updates must be drained each frame.
- **`src/api.rs`**: OpenAI-compatible API client via `reqwest`. Note: `fetch_models()` explicitly hits Ollama's `/api/tags`, while `stream_chat()` sends to `/v1/chat/completions`. SSE chunks are parsed incrementally from a byte stream buffer.
- **`src/message.rs`**: `Message` and `Role` types, serde-compatible with OpenAI chat format.
- **`src/storage.rs`**: SQLite persistence via `rusqlite` using a bundled sqlite feature.
- **`src/config.rs`**: Settings persistence via TOML (`~/.config/hchat/config.toml`).

## Concurrency Model
- **Async Bridging**: A `tokio::runtime::Runtime` lives directly inside `ChatApp`. API calls are spawned onto this runtime, and results flow back via a `tokio::sync::mpsc::UnboundedReceiver`. These results are then drained each frame in `process_events()` within `app.rs`.
- **Dual-channel pattern**: There are two independent receivers drained in `process_events()`: `rx` for chat streaming and `models_rx` for model fetches. They use separate channels so a model refresh doesn't interfere with an active chat stream. If you add a new async operation, give it its own channel and drain it in `process_events()`.

## State Duplication Between `ChatApp` and `Config`
Several settings exist as both top-level `ChatApp` fields and inside `self.config`: `dark_mode`, `system_prompt`, `temperature`, `max_tokens`, `use_max_tokens`, `saved_endpoints`, and `base_url` (mapped to `config.default_endpoint`). The UI reads/writes the `ChatApp` fields at runtime. The two sync points are:
- **Save** (`Save settings` button): copies `ChatApp` fields → `self.config`, then writes to disk.
- **Reload** (`reload_config()`): reads disk → `Config`, then copies into `ChatApp` fields.

When adding new settings, you must update both sync paths or the value will silently drift between runtime state and persisted config. Font fields (`font_family`, `font_size`, `mono_font_family`, `mono_font_size`) are an exception — they live only on `self.config` and are read directly from there by the UI.

## UI & Cross-Platform
- Built with `eframe`/`egui`. It handles platform differences natively (Metal on macOS, Vulkan/OpenGL on Linux). Do not use conditional compilation for rendering differences.
