# hChat

A lightweight desktop chat client for local LLMs. Built in Rust with [egui](https://github.com/emilk/egui) for a minimal, fast UI.

Connects to any OpenAI API-compatible endpoint. Defaults to [Ollama](https://ollama.com) at `localhost:11434`.

## Features

- Streaming token display with stop/regenerate controls
- **Branching conversations** — every regenerate or edit creates a sibling instead of overwriting; navigate alternates with `◀ N/M ▶` arrows on any branched message
- **Multimodal content** — drag-and-drop images (png/jpg/webp/gif) into the input as attachments for vision-capable models
- **Drag-and-drop text files** (rs, py, md, json, etc.) inline into the input as fenced code blocks with language inferred from the extension
- **Per-conversation settings** — each chat owns its own model, system prompt, temperature, sampling params, and endpoint, persisted to its row
- **Presets** — save the current chat's settings as a named bundle, apply to other chats or seed new ones
- **Pinned conversations** sort to the top of the sidebar
- **Drafts** — your in-progress input persists per-conversation; switch chats and come back to find it intact
- **Auto-titled chats** — after the first assistant response, hChat generates a short conversation title via a one-shot completion against your current model
- Markdown rendering in AI responses, with **per-code-block copy buttons** and language pills
- **Reasoning model support** — inline `<think>` blocks and provider `reasoning` deltas (qwen3, deepseek-r1, gpt-oss, etc.) render as collapsible sections that auto-collapse when streaming finishes
- **Slash commands** for keyboard-first control: `/model`, `/temp`, `/system`, `/clear`, `/copy`, `/help`
- **Find in conversation** (Ctrl+F) with match highlighting and scroll-to-first-match
- Model selector auto-populated from your endpoint
- Conversation history with SQLite persistence (WAL + busy_timeout). Schema migrations run on launch, so upgrading between versions doesn't require wiping your data
- Sidebar with conversation list, search, rename, pin, and markdown export
- **Live token counter** in the input footer (tiktoken-cached) plus post-response usage and cost display
- System prompt, temperature, max tokens — plus **advanced sampling controls** (`top_p`, `frequency_penalty`, `presence_penalty`, stop sequences)
- Multiple saved API endpoints with per-endpoint API key support
- Works with OpenRouter, LM Studio, vLLM, Ollama, and any OpenAI-compatible API
- Hover timestamps on messages
- Empty-state starter prompts to kick off new chats
- Dark/light theme toggle
- Persistent settings via TOML config (`~/.config/hchat/config.toml`) — corrupted files are backed up rather than silently reset
- Configurable font sizes, custom fonts, and UI scale
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

Two layers:

- **Global defaults** live in `~/.config/hchat/config.toml` — model name, system prompt, sampling params, endpoints, fonts, theme. These seed every new conversation. Edit the file directly or use the settings panel (gear icon). See [example.config.toml](example.config.toml) for a fully commented example.
- **Per-conversation overrides** live in the SQLite database alongside your messages. Tweaking the system prompt or temperature inside a chat only affects *that* chat — your global defaults stay untouched. Save the resulting bundle as a preset to apply elsewhere.

Conversation data lives in `~/Library/Application Support/hchat/hchat.db` (macOS) or `~/.local/share/hchat/hchat.db` (Linux). Schema migrations run automatically on launch, so upgrading between versions doesn't require wiping your data.

All config fields are optional. Missing fields use defaults, so existing configs won't break on upgrade.

### Custom fonts

Set `font_family` and `mono_font_family` to any font installed on your system. hChat looks up fonts by name using your platform's font system (fontconfig on Linux, Core Text on macOS). Leave empty to use egui's built-in fonts. Font changes take effect on save and restart.

### API keys

API keys are stored per-endpoint in `config.toml`. Endpoints that don't need authentication (like local Ollama) simply omit the `api_key` field. Keys are sent as `Authorization: Bearer` headers.

## Slash commands

Type any of these in the input box to control hChat without leaving the keyboard. Aliases in parens.

| Command | Action |
|---|---|
| `/model <name>` (`/m`) | Switch model — argument is substring-matched against your model list |
| `/temp <0..2>` (`/t`) | Set sampling temperature |
| `/system <text>` (`/sys`) | Set the system prompt for the current conversation (empty argument clears it) |
| `/clear` (`/new`) | Start a new conversation |
| `/copy` | Copy the last assistant reply to clipboard |
| `/help` (`/?`, `/h`) | Show the command reference |

Unknown commands surface as a toast rather than being sent to the model, so a typo like `/temprature 0.5` won't reach your provider.

## Keybindings

| Key | Action |
|---|---|
| Enter | Send message |
| Shift+Enter | New line |
| Ctrl/Cmd+N | New conversation |
| Ctrl/Cmd+F | Toggle find-in-conversation |
| Esc | Close find bar |

Shortcuts are gated on whether a text field is currently focused — if you're typing in a settings field or message edit, Ctrl+F/Ctrl+N won't fire until you defocus.
