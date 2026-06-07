# Fornax Agent Instructions

Fornax is a **Tauri** app (Rust backend + React/TS frontend) — a local-LLM
workstation: chat client + live inference-metrics dashboard + artifacts panel.
Migrated from the original Rust/egui app; chat parity, the metrics dashboard, the
artifacts panel, sidebar projects, and themes (dark/light/Catppuccin) are all
live.

## Commands

- **Run (dev)**: `npm run tauri dev` (Vite dev server + cargo, hot reload)
- **Build bundle**: `npm run tauri build`
- **Backend build**: `cargo build --manifest-path src-tauri/Cargo.toml`
- **Backend lint**: `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings`
- **Backend format**: `cargo fmt --manifest-path src-tauri/Cargo.toml`
- **Frontend typecheck**: `npx tsc --noEmit`
- **Frontend bundle check**: `npx vite build`

First run needs `npm install`. If the global npm cache errors out (EACCES/EEXIST),
use a project-local cache: `npm install --cache "$PWD/.npm-cache"`.

## Architecture & Conventions

### Rust backend — `src-tauri/src/`
- **`main.rs` / `lib.rs`**: entry point + `tauri::Builder`. `lib.rs` declares the
  core modules at the crate root via `#[path = "core/…"]` and registers every
  `#[tauri::command]`.
- **`state.rs`**: `AppState` (managed by Tauri) — `Mutex<Storage>`,
  `Mutex<Config>`, the stream `CancellationToken`, the `pending_approvals`
  oneshot map, the per-conversation `auto_approved` allowlist, and the
  `Arc<McpManager>`.
- **`commands.rs`**: the command layer replacing the egui loop. DTOs + commands;
  chat orchestration is `run_turn` (tool loop) calling `stream_once`.
- **`mcp/`**: MCP client manager (`rmcp` SDK) — connects stdio servers, discovers
  tools, routes `mcp_<server>_<tool>` calls; merged into the tool set in `run_turn`.
- **`metrics/`**: the ~1.5s metrics poller — GPU (`gpu.rs`: macmon / remote
  `fornax-agent`) + server (`prometheus.rs`: vLLM/llama.cpp/llama-swap `/metrics`
  scrape); emits the `metrics` event.
- **`core/`**: reused UI-agnostic modules — `api.rs` (OpenAI client, SSE
  streaming, `fetch_models`), `message.rs`, `storage.rs` (SQLite + branching tree
  + presets + projects), `config.rs` (TOML), `tools.rs`, `agents.rs` (`~/.agents`
  loader), `markdown.rs`, `slash.rs`.

### Frontend — `src/`
- **`App.tsx`**: top-level state; `runStream(starter)` is the shared streaming
  pipeline for send / regenerate / edit; `applyAppearance` sets the theme/fonts.
- **`lib/tauri.ts`**: the only place that calls `invoke`/`Channel`. **`types.ts`**
  mirrors the Rust DTOs — keep them in sync. `lib/artifacts.ts` extracts + titles
  artifacts.
- **`components/`**: `Sidebar` (chats + projects + move-to-project menu),
  `ChatView`, `MessageItem`, `Markdown`+`CodeBlock` (react-markdown + shiki),
  `StatusView`, `ArtifactPanel`, `RightDock` (Status/Artifacts tabs beside the
  chat), `Dialog` (`useDialog` — in-app confirm/prompt), `SettingsModal`,
  `ApprovalCard`.

## Concurrency Model

- The core `api::stream_chat` pushes `StreamEvent`s over a `tokio::sync::mpsc`
  channel. `stream_once` drains it and forwards translated `ChatEvent`s to the
  frontend over a Tauri `Channel<ChatEvent>`. The frontend's `handleEvent`
  (in `App.tsx`) applies them.
- Async commands run on Tauri's tokio runtime, so `tokio::spawn` works.
- **Never hold a std `Mutex` guard across `.await`.** Lock `AppState`, do the
  synchronous DB/config work, drop the guard, then await. Every command in
  `commands.rs` follows this — match it.
- Tool approvals use a `oneshot` parked in `AppState.pending_approvals` keyed by
  tool_call id; the `resolve_tool` command fulfills it; cancellation denies.

## Gotchas

- **`gen` is a reserved keyword** (edition 2024). The shared generation-params
  struct is `GenParams`, its field is `gp`. Don't reintroduce a `gen` identifier.
- Core modules are crate-root via `#[path]` — keep `crate::message::…` etc.
  working; don't move them under a `core::` namespace.
- API keys live in `config.toml` and are resolved server-side
  (`AppState::api_key_for`) — never send them through the frontend.
- `config.toml` is the source of truth and stays hand-editable; endpoints take
  arbitrary URL/port + optional key. Preserve the OpenRouter host-exactness check
  in `api.rs` (don't substring-match hosts — it's a credential-redirect guard).
- Adding a backend↔frontend boundary: add the `#[tauri::command]`, register it in
  `lib.rs`, add a typed wrapper in `lib/tauri.ts`, and mirror the DTO in
  `types.ts`.
- **Native browser dialogs don't work** in the WKWebView — `window.confirm` /
  `prompt` / `alert` silently no-op. Use `useDialog()` (`components/Dialog.tsx`).
- **HTML5 drag-and-drop** needs `dragDropEnabled: false` on the window in
  `tauri.conf.json` (the native OS drag-drop handler otherwise swallows webview
  DnD; this also disables OS file-drop).
- **Themes** are `:root[data-theme="…"]` CSS-variable blocks in `styles.css`,
  applied to `<html>` by `applyAppearance`. To add one: a CSS block + an entry in
  `SettingsModal`'s `THEMES` and the `config.rs` `THEMES` allowlist. `config.theme`
  is the source of truth (`Config::sanitize` migrates/syncs the legacy `dark_mode`).
- **Projects** are organizational only (`project_id` groups chats in the sidebar);
  nothing in `run_turn`/streaming reads them.

## Delivery standards

For substantial work: `cargo clippy -- -D warnings` clean, `tsc --noEmit` +
`vite build` clean, run the review subagent, and follow the PR ritual before
merging.
