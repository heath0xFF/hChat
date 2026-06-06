# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is hChat

A local-LLM **workstation** desktop app: chat client + live inference-metrics
dashboard + (planned) artifact rendering. It connects to OpenAI-compatible
endpoints — a remote NVIDIA DGX Spark running vLLM, a MacBook running oMLX (MLX)
and llama.cpp (GGUF), OpenRouter, LM Studio, Ollama, etc. — and surfaces decode
tok/s, TTFT, prefill, requests, VRAM, power, and GPU stats per backend.

> **Status:** mid-rewrite. The app was originally Rust/egui; it is being rebuilt
> as a **Tauri** app (Rust core + React/TS frontend) on the `rewrite-tauri`
> branch. **Phase A is complete** (full chat parity). Phase B (metrics dashboard)
> and Phase C (artifacts sidebar) are next. See the plan and roadmap in the
> auto-memory and `/Users/heath/.claude/plans/`.

## Build & Run

```bash
npm install                 # first time (frontend deps + Tauri CLI)
npm run tauri dev           # run the app (Vite dev server + cargo, hot reload)
npm run tauri build         # production bundle (.app / .dmg / .deb)

# Backend-only checks (faster iteration):
cargo build   --manifest-path src-tauri/Cargo.toml
cargo clippy  --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
# Frontend-only checks:
npx tsc --noEmit            # typecheck
npx vite build              # bundle check
```

The `dist` profile in `src-tauri/Cargo.toml` uses `opt-level = "s"`, LTO, and
symbol stripping for a small binary.

## Architecture

Tauri splits the app into a **Rust backend** (`src-tauri/`) and a **web frontend**
(`src/`), bridged by Tauri commands (request/response) and a per-request
`Channel` (streaming).

### Rust backend — `src-tauri/src/`

- **`main.rs` / `lib.rs`** — entry point + the `tauri::Builder`. `lib.rs` declares
  the core modules at the crate root via `#[path = "core/…"]` (so the ported
  code's `crate::message::…` references keep working) and registers every
  `#[tauri::command]` in `generate_handler!`.
- **`state.rs`** — `AppState`: `Mutex<Storage>`, `Mutex<Config>`,
  `Mutex<Vec<ToolDef>>`, the current stream's `CancellationToken`, and the
  `pending_approvals` map (oneshots keyed by tool_call id). Managed by Tauri.
- **`commands.rs`** — the command layer that *replaces the old egui loop*. DTOs +
  commands for config, models, conversations, streaming chat, tools, branching,
  and presets. The chat orchestration lives in `run_turn` / `stream_once` (see
  Concurrency).
- **`core/`** — the reused, UI-agnostic modules (largely unchanged from the egui
  app): `api.rs` (OpenAI-compatible client, SSE streaming, `fetch_models`),
  `message.rs` (`Message`/`Role`/`ContentPart`/`ToolCall`), `storage.rs` (SQLite
  via rusqlite — conversations, branching tree, presets), `config.rs` (TOML
  config + `Endpoint`), `tools.rs` (tool defs + builtins + shell handler),
  `agents.rs` (the `~/.agents` loader — commands/skills/tools, user +
  project-local), `markdown.rs` (segmenting helpers), `slash.rs` (the original
  slash parser; the live UI reimplements it in `lib/slash.ts`).

### Frontend — `src/` (React + TypeScript + Vite)

- **`App.tsx`** — top-level state + orchestration. Owns messages, settings,
  models, streaming, metrics, presets, and the sibling map. `runStream(starter)`
  is the shared streaming pipeline used by send / regenerate / edit.
- **`lib/tauri.ts`** — typed wrappers around `invoke`/`Channel` (the only place
  that touches the Tauri API). **`types.ts`** mirrors the Rust DTOs.
- **`components/`** — `Sidebar`, `ChatView` (topbar + composer + attachments +
  presets), `MessageItem` (markdown, reasoning, tool chips, branch nav, inline
  edit), `Markdown` + `CodeBlock` (react-markdown + shiki), `StatusView` (the
  live metrics dashboard), `ArtifactPanel` (HTML/SVG/Markdown/code preview),
  `SettingsModal`, `ApprovalCard`. `lib/artifacts.ts` extracts artifacts from
  assistant messages; the backend `metrics` event drives `StatusView`.
- **`styles.css`** — hand-rolled terminal-minimal dark theme (CSS variables).

## Key Design Decisions

- **Streaming**: the core `api::stream_chat` still pushes `StreamEvent`s through a
  `tokio::sync::mpsc` channel. `stream_once` drains that channel and forwards
  translated `ChatEvent`s to the frontend over a Tauri `Channel<ChatEvent>`.
- **Tool loop**: `run_turn` streams a turn → persists the assistant message (with
  `tool_calls`) → executes tools (Auto runs inline; Confirm parks a oneshot in
  `pending_approvals` and emits `tool_approval`, resolved by the `resolve_tool`
  command) → re-streams, up to `MAX_TOOL_ITERATIONS` (8).
- **Branching**: `regenerate` and `edit_message` create sibling branches via the
  existing storage tree (`next_branch_index`, `siblings_of`, `walk_from`). After
  each turn the frontend reloads the canonical active path so message ids stay
  correct. `message_siblings` + `walk_from` power the `◀ N/M ▶` navigator.
- **Locking**: commands lock `AppState` mutexes only for short synchronous DB/
  config work and **drop the guard before any `.await`** — never hold a std
  `Mutex` guard across a suspend point.
- **Secrets**: API keys are resolved server-side from `config.toml`
  (`AppState::api_key_for`) and never round-trip through JS.
- **config.toml is the source of truth** and stays hand-editable. Endpoints are
  fully customizable (arbitrary URL/port + optional per-endpoint key). The
  OpenRouter URL auto-fix + host-exactness check in `api.rs` is preserved.

## Conventions

- Keep `crate::…` paths working: core modules are crate-root via `#[path]`. Don't
  rename them to `core::…`.
- `gen` is a reserved keyword (edition 2024) — the shared generation-params struct
  is named `GenParams` and its field is `gp`.
- Delivery standards for substantial work: `cargo clippy -- -D warnings` clean,
  run the review subagent, and follow the PR ritual.

## Not yet ported from the egui app

Most parity features are back: slash commands (`lib/slash.ts`), find-in-conversation
(Cmd+F), per-conversation drafts (`save_draft` + lifted composer input), the live
token counter (`gpt-tokenizer`), and theme/fonts/UI-scale (`applyAppearance` in
App.tsx). Still missing: the "approve all in this conversation" tool allowlist
(approvals are per-call), a one-click markdown-export button (the `export_conversation`
command exists), and per-message hover timestamps.
