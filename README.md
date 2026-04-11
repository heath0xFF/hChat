# hChat

A lightweight desktop chat client for local LLMs. Built in Rust with [egui](https://github.com/emilk/egui) for a minimal, fast UI.

Connects to any OpenAI API-compatible endpoint. Defaults to [Ollama](https://ollama.com) at `localhost:11434`.

## Features

- Streaming token display with stop/regenerate controls
- Markdown rendering in AI responses (code blocks, bold, lists)
- Model selector auto-populated from your endpoint
- Conversation history with SQLite persistence
- Sidebar with conversation list, search, and export
- System prompt, temperature, and max tokens settings
- Multiple saved API endpoints with per-endpoint API key support
- Works with OpenRouter, LM Studio, vLLM, and any OpenAI-compatible API
- Edit and resend previous messages
- Copy button on messages
- Token usage display
- Dark/light theme toggle
- Persistent settings via TOML config (`~/.config/hchat/config.toml`)
- Configurable font sizes and UI scale
- Cross-platform (Linux, macOS)

## Install

Download the latest release from [GitHub Releases](https://github.com/heath0xFF/hChat/releases).

### macOS (Homebrew)

```bash
brew tap heath0xFF/tap
brew install hchat
```

To update to the latest release:

```bash
brew update && brew upgrade hchat
```

### macOS (manual)

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

### Using OpenRouter or other remote APIs

1. Click `+` next to the endpoint selector in the top bar
2. Enter the API base URL (e.g. `https://openrouter.ai/api/v1`) and your API key
3. Click Add, then select the new endpoint from the dropdown
4. Models auto-populate from the remote API

Or configure it directly in `config.toml` — see [example.config.toml](example.config.toml) for all options.

## Configuration

Settings are stored in `~/.config/hchat/config.toml` and persist across sessions. You can edit the file directly or use the settings panel in the app (gear icon). See [example.config.toml](example.config.toml) for a fully commented example of all options.

All fields are optional. Missing fields use defaults, so existing configs won't break on upgrade.

### Custom fonts

Set `font_family` and `mono_font_family` to any font installed on your system. hChat looks up fonts by name using your platform's font system (fontconfig on Linux, Core Text on macOS). Leave empty to use egui's built-in fonts. Font changes take effect on save and restart.

### API keys

API keys are stored per-endpoint in `config.toml`. Endpoints that don't need authentication (like local Ollama) simply omit the `api_key` field. Keys are sent as `Authorization: Bearer` headers.

## Keybindings

| Key | Action |
|---|---|
| Enter | Send message |
| Shift+Enter | New line |
| Ctrl+N | New conversation |
