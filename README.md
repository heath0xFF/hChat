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
- Persistent settings via TOML config (`~/.config/hchat/config.toml`)
- Configurable font sizes and UI scale
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

Make sure your local LLM server is running first:

```bash
ollama serve
```

Then launch hChat:

```bash
# Linux: run detached so it doesn't tie up your terminal
hchat &disown

# macOS: open the .app bundle, or run detached
open /Applications/hChat.app
# or
hchat &disown
```

hChat connects to `http://localhost:11434/v1` by default. You can change the endpoint in the top bar.

## Configuration

Settings are stored in `~/.config/hchat/config.toml` and persist across sessions. You can edit the file directly or use the settings panel in the app (gear icon).

```toml
font_size = 14.0
mono_font_size = 13.0
ui_scale = 1.0
dark_mode = true
default_endpoint = "http://localhost:11434/v1"
system_prompt = ""
temperature = 0.7
max_tokens = 2048
use_max_tokens = false
saved_endpoints = ["http://localhost:11434/v1"]
```

All fields are optional. Missing fields use defaults, so existing configs won't break on upgrade.

## Keybindings

| Key | Action |
|---|---|
| Enter | Send message |
| Shift+Enter | New line |
| Ctrl+N | New conversation |
