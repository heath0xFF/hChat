# hChat

A lightweight desktop chat client for local LLMs. Built in Rust with [egui](https://github.com/emilk/egui) for a minimal, fast UI.

Connects to any OpenAI API-compatible endpoint. Defaults to [Ollama](https://ollama.com) at `localhost:11434`.

## Features

- Streaming token display with stop/regenerate controls
- Markdown rendering in AI responses (code blocks, bold, lists)
- Model selector auto-populated from Ollama
- Conversation history with SQLite persistence
- Sidebar with conversation list, search, and export
- System prompt, temperature, and max tokens settings
- Multiple saved API endpoints
- Edit and resend previous messages
- Copy button on messages
- Token usage display
- Dark/light theme toggle
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

## Launch

1. Start your local LLM server (e.g. `ollama serve`)
2. Run `hchat` from your terminal, or open hChat from your Applications folder on macOS

hChat connects to `http://localhost:11434/v1` by default. You can change the endpoint in the top bar.

## Keybindings

| Key | Action |
|---|---|
| Enter | Send message |
| Shift+Enter | New line |
| Ctrl+N | New conversation |
