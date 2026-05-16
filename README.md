# hChat

A lightweight desktop chat client for local LLMs. Built in Rust with [egui](https://github.com/emilk/egui) for a minimal, fast UI.

Connects to any OpenAI API-compatible endpoint. Defaults to [LM Studio](https://lmstudio.ai) at `localhost:1234`.

## Features

- Streaming token display with stop/regenerate controls
- **Native tool calling** — the model can read files, search code, run shell commands, and write back into your working directory. Five default tools ship pre-configured (`read_file`, `list_directory`, `search_files`, `write_file`, `run_shell`); destructive ones prompt for approval. Define your own tools as TOML files in `~/.config/hchat/tools/` (works with any tool-capable OpenAI-compatible model)
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
- Works with LM Studio, Ollama, oMLX, OpenRouter, vLLM, and any OpenAI-compatible API
- Hover timestamps on messages
- Empty-state starter prompts to kick off new chats
- Dark/light theme toggle
- Persistent settings via TOML config (`~/.config/hchat/config.toml`) — corrupted files are backed up rather than silently reset
- Configurable font sizes, custom fonts, and UI scale
- Cross-platform (Linux, macOS)

## Prerequisites

hChat is a client — it needs an OpenAI-compatible chat completions endpoint to talk to. Pick whichever you prefer:

### LM Studio (default)

1. Download from [lmstudio.ai](https://lmstudio.ai).
2. Inside LM Studio, search for and download a model (e.g. `qwen2.5-coder-7b-instruct`).
3. Switch to the **Developer** tab, load the model, and start the local server.
4. LM Studio's server exposes three API dialects on port 1234 — a native LM Studio API under `/api/v1/...`, an OpenAI-compatible API under `/v1/...`, and an Anthropic-compatible `/v1/messages`. hChat speaks the OpenAI-compatible routes, so the default endpoint is `http://localhost:1234/v1` — no extra configuration required.

### Ollama

1. Install from [ollama.com](https://ollama.com) (or your package manager).
2. Pull a model: `ollama pull qwen2.5-coder:7b`.
3. Run `ollama serve` (it auto-starts on macOS).
4. In hChat, click `+` next to the endpoint selector and add `http://localhost:11434/v1` — or set it as `default_endpoint` in `config.toml`.

### oMLX

1. Download from [github.com/heath0xFF/oMLX](https://github.com/heath0xFF/oMLX).
2. oMLX runs from the macOS menu bar — it handles model loading, continuous batching, and SSD caching automatically.
3. By default oMLX exposes an OpenAI-compatible endpoint at `http://localhost:8000/v1`. Add it in hChat via the `+` button, or set it as `default_endpoint` in `config.toml`.

### Remote APIs

OpenRouter, OpenAI, vLLM, Together, or any other OpenAI-compatible host works. You'll need the base URL and an API key — see [Using OpenRouter or other remote APIs](#using-openrouter-or-other-remote-apis) below.

If hChat can reach the endpoint but reports "No models available", you started the server but haven't loaded (LM Studio) or pulled (Ollama) a model yet.

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

With your LLM server already running (see [Prerequisites](#prerequisites)):

```bash
# Linux: run detached so it doesn't tie up your terminal
hchat &disown

# macOS: open the .app bundle, or run detached
open /Applications/hChat.app
# or
hchat &disown
```

hChat connects to `http://localhost:1234/v1` (LM Studio) by default. Switch endpoints from the top bar dropdown, or add new ones via the `+` button.

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

API keys are stored per-endpoint in `config.toml`. Endpoints that don't need authentication (like local LM Studio or Ollama) simply omit the `api_key` field. Keys are sent as `Authorization: Bearer` headers.

## Tools

Tool-capable models (OpenAI gpt-4+, Claude, most modern frontier models) can call functions hChat exposes. Five defaults are seeded into `~/.config/hchat/tools/` on first launch:

| Tool | Safety | Description |
|---|---|---|
| `read_file` | auto | Reads a file (with optional `offset`/`limit` for slicing). Up to 100KB per call. |
| `list_directory` | auto | Lists entries with `d/` (directory) or `f/` (file) prefix. |
| `search_files` | auto | Recursive regex search; skips dotdirs and binary files. |
| `write_file` | confirm | Writes a file (overwrite or append). Creates parent dirs. |
| `run_shell` | confirm | Runs `sh -c` in the conversation's working directory. 5 minute wall-clock cap. |

`safety = "auto"` tools execute silently; `safety = "confirm"` tools pop an approval card with the full args before running. The card has **Approve**, **Approve all in this conv** (per-conversation allowlist for that tool name), and **Reject** buttons.

### Working directory

Each conversation has its own `working_dir` (settings panel). All tool calls resolve relative paths against it. Defaults to your home directory; set it to a project root and the model can reason about that codebase end-to-end.

### Defining your own tools

Drop a `.toml` file into `~/.config/hchat/tools/`. The minimum is `name`, `description`, JSON-schema `parameters`, and a `handler`. Two handler types:

```toml
# Builtin: hardcoded Rust handler. Used by the 5 defaults.
handler = "builtin:read_file"

# Shell: forks an argv with {{name}} substitution from the call's arguments.
handler = { shell = ["git", "log", "--oneline", "-n", "{{count}}"] }
safety = "confirm"  # or "auto" for read-only tools
```

Restart hChat to load new tools. Edits to existing tool files take effect on next launch.

### Iteration cap

The model can chain tool calls — read a file, look at the imports, read those, then propose an edit. hChat caps this at **8 cycles per user turn** to prevent runaway loops. The counter resets on the next user message (or on regenerate / edit).

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
