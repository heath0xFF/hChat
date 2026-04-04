# hChat

A lightweight desktop chat client for local LLMs. Built in Rust with [egui](https://github.com/emilk/egui) for a minimal, fast UI.

Connects to any OpenAI API-compatible endpoint. Defaults to [Ollama](https://ollama.com) at `localhost:11434`.

## Features

- Streaming token display (SSE)
- Model selector auto-populated from Ollama
- Configurable API endpoint
- Cross-platform (Linux, macOS)

## Install

Download the latest release from [GitHub Releases](https://github.com/heath0xFF/hChat/releases).

### macOS

```bash
# Binary
tar xzf hchat-macos-arm64.tar.gz
mv hchat /usr/local/bin/

# Or use the .app bundle
unzip hChat.app.zip -d /Applications
```

### Debian/Ubuntu

```bash
sudo dpkg -i hchat_*.deb
```

### Arch Linux

```bash
# From the repo's pkg/arch directory
makepkg -si
```

## Build from source

Requires the [Rust toolchain](https://rustup.rs).

```bash
cargo run --release
```

## Usage

Make sure an OpenAI-compatible API server is running (e.g. `ollama serve`), then launch `hchat`.

## Keybindings

| Key | Action |
|---|---|
| Enter | Send message |
| Shift+Enter | New line |
