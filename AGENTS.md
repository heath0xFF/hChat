# hChat Agent Instructions

hChat is a **Tauri** app (Rust backend + React/TS frontend) — a local-LLM
workstation: chat client + live inference-metrics dashboard + (planned) artifact
rendering. It is mid-rewrite from the original Rust/egui app on the
`rewrite-tauri` branch; **Phase A (chat parity) is complete**, Phase B (metrics)
and Phase C (artifacts) are next.

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
  `Mutex<Config>`, `Mutex<Vec<ToolDef>>`, the stream `CancellationToken`, and the
  `pending_approvals` oneshot map.
- **`commands.rs`**: the command layer replacing the egui loop. DTOs + commands;
  chat orchestration is `run_turn` (tool loop) calling `stream_once`.
- **`core/`**: reused UI-agnostic modules — `api.rs` (OpenAI client, SSE
  streaming, `fetch_models`), `message.rs`, `storage.rs` (SQLite + branching tree
  + presets), `config.rs` (TOML), `tools.rs`, `markdown.rs`, `slash.rs`.

### Frontend — `src/`
- **`App.tsx`**: top-level state; `runStream(starter)` is the shared streaming
  pipeline for send / regenerate / edit.
- **`lib/tauri.ts`**: the only place that calls `invoke`/`Channel`. **`types.ts`**
  mirrors the Rust DTOs — keep them in sync.
- **`components/`**: `Sidebar`, `ChatView`, `MessageItem`, `Markdown`+`CodeBlock`
  (react-markdown + shiki), `StatusView`, `SettingsModal`, `ApprovalCard`.

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

## Delivery standards

For substantial work: `cargo clippy -- -D warnings` clean, `tsc --noEmit` +
`vite build` clean, run the review subagent, and follow the PR ritual before
merging.
