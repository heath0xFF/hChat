# hChat

A lightweight desktop chat client for local LLMs. Built in Rust with [egui](https://github.com/emilk/egui) for a minimal, fast UI.

Connects to any OpenAI API-compatible endpoint. Defaults to [Ollama](https://ollama.com) at `localhost:11434`.

## Features

- Streaming token display (SSE)
- Model selector auto-populated from Ollama
- Configurable API endpoint
- Cross-platform (Linux, macOS)

## Requirements

- Rust toolchain (`rustup`)
- A running OpenAI-compatible API server (e.g. `ollama serve`)

## Build & Run

```bash
cargo run --release
```

## Keybindings

| Key | Action |
|---|---|
| Enter | Send message |
| Shift+Enter | New line |
