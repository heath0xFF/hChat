# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is hChat

A lightweight, fast desktop chat client for OpenAI API-compatible local models (e.g. Ollama). Built in Rust with egui for minimal footprint and maximum speed. Targets Arch Linux and macOS.

## Build & Run

```bash
cargo build --release        # optimized binary at target/release/hchat
cargo run --release           # build and launch
cargo run                     # debug build (faster compile, slower runtime)
```

Release profile uses `opt-level = "s"`, LTO, and symbol stripping for a small binary.

## Architecture

- **`src/main.rs`** — Entry point. Configures eframe window and launches `ChatApp`.
- **`src/app.rs`** — Core UI logic. `ChatApp` struct holds all state (messages, model list, streaming receiver). Uses egui immediate-mode rendering with top bar (model selector + endpoint), bottom panel (input), and central scrollable message area.
- **`src/api.rs`** — OpenAI-compatible API client. `fetch_models()` hits Ollama's `/api/tags`. `stream_chat()` sends to `/v1/chat/completions` with SSE streaming, pushing `StreamEvent`s through a `tokio::sync::mpsc` channel.
- **`src/message.rs`** — `Message` and `Role` types, serde-compatible with OpenAI chat format.

## Key Design Decisions

- **Async bridging**: A `tokio::runtime::Runtime` lives inside `ChatApp`. API calls are spawned on it; results flow back via `mpsc::UnboundedReceiver` and are drained each frame in `process_events()`.
- **Streaming**: SSE chunks are parsed incrementally from a byte stream buffer, not buffered whole.
- **Cross-platform**: eframe/wgpu handles platform differences (Vulkan/OpenGL on Linux, Metal on macOS). No conditional compilation needed.
- **Default endpoint**: `http://localhost:11434/v1` (Ollama's OpenAI-compatible API).
