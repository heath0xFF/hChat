# hChat

A local-LLM **workstation** for your desktop: a fast chat client, a live
inference-metrics dashboard, and (soon) an artifact-rendering sidebar — over any
OpenAI-compatible endpoint.

Built with [Tauri](https://tauri.app) (Rust core + a React/TypeScript frontend).
Talk to a remote **DGX Spark** running vLLM, your MacBook running **oMLX** (MLX)
and **llama.cpp** (GGUF), **OpenRouter**, **LM Studio**, **Ollama**, or anything
else that speaks the OpenAI API — and see decode tok/s, TTFT, prefill, requests,
VRAM, power, and GPU stats for each.

> ### ⚠️ Status: rewrite in progress
> hChat started as a Rust/egui chat client and is being rebuilt as a Tauri
> workstation on the `rewrite-tauri` branch.
> - **Phase A — chat (done):** full-featured streaming chat with history,
>   branching, tools, attachments, and presets.
> - **Phase B — metrics dashboard (core done):** live GPU stats via `macmon`
>   (Apple Silicon, no sudo) and the `hchat-agent` on the Spark, plus vLLM /
>   llama.cpp Prometheus scraping. The Status view shows decode/TTFT/prefill,
>   requests, VRAM, power, temp, and per-GPU rows.
> - **Phase C — artifacts sidebar (upcoming):** Claude-Desktop-style rendering of
>   the markdown/code/HTML your models produce.

## Features (today)

- **Streaming chat** with stop, against any OpenAI-compatible endpoint
- **Multiple backends** — switch endpoints from the top bar; each conversation
  remembers its own model + endpoint. Arbitrary URLs/ports, per-endpoint API keys
- **Native tool calling** — the model can read files, search code, run shell
  commands, and write into a working directory. Five tools ship pre-configured
  (`read_file`, `list_directory`, `search_files`, `write_file`, `run_shell`);
  destructive ones pop an approval card. Define your own as TOML in
  `~/.config/hchat/tools/`. Chains are capped at 8 tool cycles per turn
- **Branching** — regenerate the last reply or edit any message to fork a new
  sibling instead of overwriting; navigate alternates with `◀ N/M ▶`
- **Attachments** — drag-drop / paste images (png/jpg/webp/gif) for vision models;
  drop text files (rs, py, md, json, …) to inline them as fenced code
- **Presets** — save the current model + endpoint + sampling bundle by name and
  apply it elsewhere
- **Reasoning models** — inline `<think>` blocks and provider `reasoning` deltas
  (qwen3, deepseek-r1, gpt-oss, …) render as collapsible sections
- **Markdown rendering** with syntax highlighting (shiki) and per-code-block copy
- **Per-conversation settings** — model, system prompt, temperature, and advanced
  sampling (`top_p`, `frequency_penalty`, `presence_penalty`, stop sequences)
- **Auto-titled chats** from the first exchange; sidebar with search, rename, pin,
  delete
- **SQLite history** (WAL) with on-launch schema migrations — upgrades don't wipe
  data
- **Live metrics dashboard** — per-backend decode/TTFT/prefill, requests, VRAM,
  power, temp, KV-cache, and per-GPU rows, with throughput & TTFT charts.
  Apple-Silicon GPU stats come from `macmon` (no sudo); remote NVIDIA boxes use
  the bundled **`hchat-agent`**; vLLM/llama.cpp expose the rest via Prometheus
- `config.toml` stays the source of truth and hand-editable
- Cross-platform (macOS, Linux)

## Requirements

The rewrite builds from source. You need:

- **[Rust toolchain](https://rustup.rs)** (stable) — `rustc` 1.85+
- **[Node.js](https://nodejs.org) 20+** and npm
- **Tauri system dependencies** for your platform (the native webview + build
  tools). Quick version:
  - **macOS** — Xcode Command Line Tools: `xcode-select --install`
  - **Debian/Ubuntu** —
    ```bash
    sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
      libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
    ```
  - **Arch** —
    ```bash
    sudo pacman -S --needed webkit2gtk-4.1 base-devel curl wget file openssl \
      appmenu-gtk-module libappindicator-gtk3 librsvg
    ```
  - Other distros / full details: [Tauri prerequisites](https://tauri.app/start/prerequisites/)

## Build & run

```bash
git clone https://github.com/heath0xFF/hChat
cd hChat
git checkout rewrite-tauri

npm install          # frontend deps + Tauri CLI
npm run tauri dev    # run the app (hot reload)

npm run tauri build  # produce a release bundle (.app / .dmg / .deb)
```

> Prebuilt **releases** on GitHub are the previous (egui) version. The Tauri
> workstation currently builds from source on the branch above.

## Backends

hChat is a client — point it at a running OpenAI-compatible server. Add endpoints
from the `+`/settings UI or in `config.toml`; switch from the top-bar dropdown.

| Backend | Typical endpoint | Notes |
|---|---|---|
| **oMLX** (MLX, macOS) | `http://localhost:8000/v1` | `omlx serve` from [omlx.ai](https://omlx.ai); port is configurable |
| **llama.cpp** (GGUF) | `http://localhost:8080/v1` | run `llama-server --metrics` for Phase B scraping |
| **vLLM** (e.g. DGX Spark) | `http://<host>:8000/v1` | exposes Prometheus `/metrics` |
| **OpenRouter** (cloud) | `https://openrouter.ai/api/v1` | needs an API key |
| **LM Studio** | `http://localhost:1234/v1` | load a model + start the server |
| **Ollama** | `http://localhost:11434/v1` | `ollama pull <model>` first |

If hChat reaches an endpoint but reports "No models available", the server is up
but no model is loaded/pulled.

## Metrics dashboard

The **Status** view (left rail) shows live inference metrics for the active
conversation's backend. Per-request decode/TTFT/prefill is always measured
client-side; richer stats are opt-in per endpoint in `config.toml`:

```toml
# llama.cpp on this Mac (start it with `llama-server --metrics`)
[[saved_endpoints]]
url = "http://localhost:8080/v1"
runtime = "llamacpp"
prometheus_url = "http://localhost:8080/metrics"   # decode/prefill/requests/KV

# Remote DGX Spark running vLLM, GPU stats via the agent (below)
[[saved_endpoints]]
url = "http://spark:8000/v1"
runtime = "vllm"
prometheus_url = "http://spark:8000/metrics"
gpu = "agent"
agent_url = "http://spark:9099"
```

- **`runtime`** — `vllm` | `omlx` | `llamacpp` | `openai`
- **`prometheus_url`** — scraped for decode/prefill tok/s, TTFT, requests, KV cache
- **`gpu`** — `macmon` (Apple Silicon, no sudo) | `agent` (remote NVIDIA) | `none`.
  Local endpoints on macOS default to `macmon` automatically, so VRAM/power/temp
  show up with no config.

### DGX Spark setup (`hchat-agent`)

`nvidia-smi` can't report VRAM on the GB10 / DGX Spark — CPU and GPU share
unified LPDDR5X, so memory shows as `[Not Supported]`. The bundled
**`hchat-agent`** works around this by reading `nvidia-smi` (power/temp/util)
*and* `/proc/meminfo` (unified VRAM) and serving them as JSON. It's a single
zero-dependency Rust binary (~450 KB).

**1. Make sure vLLM is exposing metrics.** vLLM serves Prometheus metrics at
`/metrics` on its API port by default — no extra flags needed. Confirm:

```bash
curl -s http://localhost:8000/metrics | head    # on the Spark
```

**2. Build the agent on the Spark.** It needs the Rust toolchain
([rustup.rs](https://rustup.rs)) and `nvidia-smi` on `PATH` (already present on a
DGX Spark). Get the source onto the box (`git clone` or `scp -r agent/`), then:

```bash
# on the Spark
git clone https://github.com/heath0xFF/hChat && cd hChat
git checkout rewrite-tauri
cd agent
cargo build --release
sudo install -m755 target/release/hchat-agent /usr/local/bin/hchat-agent
```

> Prefer not to install Rust on the Spark? Cross-compile from another Linux box
> with `rustup target add aarch64-unknown-linux-gnu` (or `…-musl` for a fully
> static binary) and `scp` the result over.

**3. Run it.** For a quick test, foreground:

```bash
hchat-agent --port 9099            # serves GET /gpu (binds 0.0.0.0 by default)
curl -s http://localhost:9099/gpu  # sanity check — JSON with VRAM/power/temp
```

To keep it running across reboots, install a systemd service:

```ini
# /etc/systemd/system/hchat-agent.service
[Unit]
Description=hChat GPU metrics agent
After=network.target

[Service]
ExecStart=/usr/local/bin/hchat-agent --port 9099
Restart=on-failure
User=YOUR_USER

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hchat-agent
```

**4. Reach it from your Mac.**

- **Same network:** use the Spark's hostname/IP — `agent_url = "http://spark:9099"`,
  `prometheus_url = "http://spark:8000/metrics"`. The agent binds `0.0.0.0`; if
  the box has a firewall, allow ports `9099` and `8000`.
- **Not on the same network (or you'd rather not expose ports):** SSH-tunnel and
  point hChat at `localhost`:
  ```bash
  ssh -N -L 9099:localhost:9099 -L 8000:localhost:8000 you@spark
  ```
  then use `agent_url = "http://localhost:9099"` and
  `prometheus_url = "http://localhost:8000/metrics"`.

> The agent serves only GPU stats and has no auth. On an untrusted network, run
> it with `--bind 127.0.0.1` and reach it through the SSH tunnel above.

**5. Add the endpoint in hChat** (see the vLLM example in the config above):
`runtime = "vllm"`, `prometheus_url`, `gpu = "agent"`, `agent_url`. Select it in
the top bar and open **Status** — you'll see decode/TTFT/prefill and requests
from vLLM, and VRAM/power/temp/util per GPU from the agent.

## Configuration

Two layers:

- **Global defaults** — `~/.config/hchat/config.toml`: default endpoint, system
  prompt, sampling params, and saved endpoints (with optional per-endpoint API
  keys). Edit directly or use the Settings modal. See
  [example.config.toml](src-tauri/example.config.toml). Corrupt files are backed
  up rather than silently reset.
- **Per-conversation overrides** — stored in SQLite alongside messages. Changing
  the model/temperature inside a chat affects only that chat. Save a bundle as a
  **preset** to reuse it.

Conversation data lives in `~/Library/Application Support/hchat/hchat.db` (macOS)
or `~/.local/share/hchat/hchat.db` (Linux). API keys are sent as
`Authorization: Bearer`; endpoints that don't need auth omit the key.

## Tools

Tool-capable models can call functions hChat exposes. Five defaults are seeded
into `~/.config/hchat/tools/` on first launch:

| Tool | Safety | Description |
|---|---|---|
| `read_file` | auto | Reads a file (optional `offset`/`limit`), up to 100 KB |
| `list_directory` | auto | Lists entries with `d/`/`f/` prefixes |
| `search_files` | auto | Recursive regex search; skips dotdirs + binaries |
| `write_file` | confirm | Writes a file; creates parent dirs |
| `run_shell` | confirm | Runs a shell command in the working dir; 5-min cap |

`auto` tools run silently; `confirm` tools show an approval card with the full
args. Each conversation has its own `working_dir` that relative paths resolve
against (defaults to home).

Define your own by dropping a `.toml` into `~/.config/hchat/tools/`:

```toml
name = "git_log"
description = "Recent commits"
parameters = { type = "object", properties = { count = { type = "integer" } } }

# Builtin Rust handler (the 5 defaults use this):
# handler = "builtin:read_file"
# Or a shell command with {{name}} substitution:
handler = { shell = ["git", "log", "--oneline", "-n", "{{count}}"] }
safety = "confirm"
```

Restart to load new/edited tools.

## Keybindings

| Key | Action |
|---|---|
| Enter | Send message |
| Shift+Enter | New line |
| Cmd/Ctrl+Enter | Save & resend (while editing a message) |

## Not yet re-ported from the egui app

Slash commands, find-in-conversation, draft persistence, the live token counter,
configurable fonts / UI-scale / theme toggle, and the "approve all in this
conversation" tool allowlist. Several of these have storage/parser primitives
still in the tree and will be re-wired as the rewrite continues.
