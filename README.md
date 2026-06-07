# Fornax

<img width="1862" height="1221" alt="image" src="https://github.com/user-attachments/assets/36f957d0-db6c-4420-8464-863ec201a632" />

A local-LLM **workstation** for your desktop: a fast chat client, a live
inference-metrics dashboard, an artifact-rendering panel, and per-request usage
tracking — over any OpenAI-compatible endpoint.

Point it at a remote **DGX Spark** running vLLM, your MacBook running **oMLX**
(MLX) and **llama.cpp** (GGUF), **OpenRouter**, **LM Studio**, **Ollama**, or
anything else that speaks the OpenAI API — chat with your models and watch decode
tok/s, TTFT, prefill, requests, VRAM, power, and per-GPU stats for each backend.

Built with [Tauri](https://tauri.app): a Rust core (streaming, SQLite history,
tools, metrics) behind a React/TypeScript UI — a small native binary, not an
Electron app.

## Features

- **Streaming chat** against any OpenAI-compatible endpoint, with per-conversation
  model, system prompt, and sampling settings
- **Multiple backends** at once — switch from the top bar; each conversation
  remembers its own. Arbitrary URLs/ports, per-endpoint API keys
- **Live metrics dashboard** — decode/TTFT/prefill, requests, VRAM, power, temp,
  KV-cache, and per-GPU rows, with throughput, TTFT, and GPU util/power charts.
  Apple-Silicon GPU
  stats via [macmon](https://github.com/vladkens/macmon) (no sudo); remote NVIDIA
  boxes via the bundled **`fornax-agent`**; vLLM / llama.cpp / llama-swap via
  Prometheus. View it full-page, or **dock it beside the chat** to watch a backend
  while you work
- **Usage history** — every turn's tokens, cost, TTFT, and tok/s are recorded
  locally; the **Usage** page totals them with cost, TTFT/decode p50·p95, a
  by-model breakdown, and a tokens-by-day chart. Optional retention window
  (`usage_retention_days`); your data only, never uploaded
- **Models browser** — one page listing the models available across every saved
  endpoint; click one to start chatting with it
- **Benchmark** — load-test any endpoint (set concurrency + request count) and get
  aggregate throughput, TTFT/decode p50·p95, and per-request charts
- **Native tool calling** — the model can read files, search code, run shell
  commands, and write into a working directory. Five tools ship pre-configured;
  destructive ones prompt for approval (with "approve all in this conversation").
  Define your own as TOML in `~/.config/fornax/tools/`
- **MCP** — connect to [Model Context Protocol](https://modelcontextprotocol.io)
  servers (stdio or streamable HTTP); their tools join the model's tool set
  automatically. Manage in Settings → MCP or `config.toml`
- **Skills & commands** — drop-in [`~/.agents`](https://www.dot-agents.com/)
  resources: `commands/` become slash commands, `skills/` are model- or
  manually-invoked, shared with your other agents (user- and project-level)
- **Artifacts panel** — renders the HTML (live, sandboxed iframe), SVG, Markdown,
  Mermaid diagrams, and code your models produce, with a preview/source toggle;
  shares a collapsible right-side dock with the live Status panel
- **Branching** — regenerate or edit any message to fork a sibling; navigate
  with `◀ N/M ▶`
- **Attachments** — drag-drop / paste images for vision models; drop text files to
  inline them as fenced code
- **Presets** — save a model + endpoint + sampling bundle and apply it elsewhere
- **Reasoning models** — `<think>` blocks and provider `reasoning` deltas render
  as collapsible sections
- **Markdown** rendering with syntax highlighting (shiki) and per-code-block copy
- **Keyboard-first** — slash commands, find-in-conversation, drafts, a live token
  counter, and fully **rebindable shortcuts**
- **SQLite history** with auto-titles, search, pin, rename, and markdown export;
  group chats into **projects** (drag a chat onto a project; pin projects + chats
  to the top)
- Light/dark themes, configurable fonts and UI scale; `config.toml` stays the
  hand-editable source of truth
- Cross-platform (macOS, Linux)

## How it works

Fornax is a thin, fast client. The Rust core makes OpenAI-compatible requests to
whatever servers you point it at, measures each request as it streams, and polls
each backend's metrics sources to populate the dashboard. Nothing leaves your
machines unless you configure a cloud endpoint.

```mermaid
flowchart LR
  subgraph app["Fornax desktop app (Tauri)"]
    ui["React UI<br/>chat · dashboard · artifacts · usage · models · bench"]
    core["Rust core<br/>streaming · SQLite · tools · metrics poller"]
    ui <--> core
  end

  core -->|OpenAI API| omlx["oMLX&nbsp;(MLX)"]
  core -->|OpenAI API| lcpp["llama.cpp / llama-swap"]
  core -->|OpenAI API| vllm["vLLM @ DGX Spark"]
  core -->|OpenAI API| or["OpenRouter / cloud"]

  core -.->|macmon| mac["Apple Silicon GPU<br/>power · temp · unified VRAM"]
  core -.->|HTTP /gpu| agent["fornax-agent<br/>(runs on the Spark)<br/>nvidia-smi + /proc/meminfo"]
  core -.->|/metrics scrape| prom["Prometheus<br/>vLLM · llama.cpp · llama-swap"]
```

Solid arrows are chat traffic; dotted arrows are metrics. Per-request
decode/TTFT/prefill is always measured client-side; the dotted sources add
server-wide throughput and GPU stats.

## Requirements

- **[Rust toolchain](https://rustup.rs)** (stable) — `rustc` 1.85+
- **[Node.js](https://nodejs.org) 20+** and npm
- **Tauri system dependencies** for your platform (the native webview + build
  tools):
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
git clone https://github.com/heath0xFF/fornax
cd fornax

npm install          # frontend deps + Tauri CLI
npm run tauri dev    # run the app (hot reload)

npm run tauri build  # produce a release bundle (.app / .dmg / .deb)
```

There are no prebuilt binaries yet — build from source with the steps above.
`npm run tauri dev` runs the app in development with hot reload — use this day to
day. `npm run tauri build` compiles an optimized frontend (`vite build`) and a
release Rust binary, then packages a **native installer for your platform** — a
`.app` + `.dmg` on macOS, a `.deb` / `.AppImage` / `.rpm` on Linux — under
`src-tauri/target/release/bundle/`. The standalone binary is at
`src-tauri/target/release/fornax` if you just want to copy that somewhere on your
`PATH`.

## Migrating from hChat

Fornax was previously called **hChat**. On first launch it **automatically
migrates** your existing data — it moves the old `hchat` directories to `fornax`
(renaming the database file too), so your conversations, settings, and custom
tools carry over with nothing to export or import. The new locations are:

- **Conversations** (SQLite): `~/Library/Application Support/fornax/fornax.db`
  (macOS) · `~/.local/share/fornax/fornax.db` (Linux)
- **Config**: `~/.config/fornax/config.toml`
- **Custom tools**: the Fornax config dir — `~/Library/Application Support/fornax/tools/`
  (macOS) · `~/.config/fornax/tools/` (Linux)

The migration only runs when the `fornax` directories don't yet exist, so it
never clobbers a fresh install. If you'd rather keep the old data untouched, back
it up first:

```bash
cp -r ~/.config/hchat ~/hchat-config-backup
cp -r ~/Library/Application\ Support/hchat ~/hchat-data-backup   # macOS
```

On first launch Fornax reads your existing `config.toml` — new fields just take
their defaults — and runs SQLite schema migrations in place, so older history
upgrades without data loss. A corrupt config is backed up rather than overwritten.

If you installed the old Homebrew build, remove it: `brew uninstall --cask hchat`
(or `brew uninstall hchat` for the CLI binary).

## Backends

Fornax is a client — point it at a running OpenAI-compatible server. Add endpoints
in **Settings → Endpoints** (or in `config.toml`); switch from the top-bar
dropdown.

| Backend | Typical endpoint | Notes |
|---|---|---|
| **oMLX** (MLX, macOS) | `http://localhost:8000/v1` | `omlx serve` from [omlx.ai](https://omlx.ai); port is configurable |
| **llama.cpp** (GGUF) | `http://localhost:8080/v1` | run `llama-server --metrics` for dashboard scraping |
| **llama-swap** | `http://host:8080/v1` | model-swapping proxy; exposes its own `/metrics` |
| **vLLM** (e.g. DGX Spark) | `http://host:8000/v1` | exposes Prometheus `/metrics` |
| **OpenRouter** (cloud) | `https://openrouter.ai/api/v1` | needs an API key |
| **LM Studio** | `http://localhost:1234/v1` | load a model + start the server |
| **Ollama** | `http://localhost:11434/v1` | `ollama pull <model>` first |

If Fornax reaches an endpoint but reports "No models available", the server is up
but no model is loaded/pulled.

## Metrics dashboard

The **Status** view shows live metrics for a backend you pick from its own
endpoint dropdown (independent of the active chat). Open it full-page from the
left rail, or hit the **⚡** button in the chat top bar to **dock it on the right**
as a tab next to Artifacts — so you can watch decode tok/s, GPU power, and VRAM
while a model is replying. A **Live | Benchmark** toggle at the top switches to a
load-tester (see [Benchmark](#benchmark) below). Per-request decode/TTFT/prefill
is measured client-side; richer stats are opt-in per endpoint (set them in
**Settings → Endpoints** or `config.toml`):

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

- **`runtime`** — `vllm` | `omlx` | `llamacpp` | `llamaswap` | `openai`
- **`prometheus_url`** — scraped for decode/prefill tok/s, TTFT, requests, KV cache
- **`gpu`** — `macmon` (Apple Silicon, no sudo) | `agent` (remote NVIDIA) | `none`.
  Local endpoints on macOS default to `macmon` automatically, so VRAM/power/temp
  show up with no config

**The two sources are independent.** `prometheus_url` feeds the *inference-server*
metrics — the decode / prefill / TTFT tiles and the **Throughput** and **TTFT**
charts — while `gpu` / `agent_url` feeds the *hardware* metrics: the VRAM tile and
the **GPU util** / **GPU power** charts. Configure either, both, or neither:

| You set… | You get | You don't get |
|---|---|---|
| `gpu = "agent"` only | VRAM, GPU util/power, temp, device rows | Throughput / TTFT / requests |
| `prometheus_url` only | Throughput, TTFT, prefill, requests, KV cache | GPU util/power, VRAM |
| both | everything | — |

So if the GPU charts move but **Throughput/TTFT stay flat, you're missing
`prometheus_url`** (not the agent) — point it at your server's `/metrics`
(`http://host:8080/metrics` for llama-swap, `:8000` for vLLM) and set the matching
`runtime`. Caveat: **llama-swap exposes no TTFT metric**, so its live TTFT chart
only ever shows one point per request even when wired up; decode/prefill stream
live.

### The metrics agent (`fornax-agent`)

`nvidia-smi` can't report VRAM on a GB10 / DGX Spark — CPU and GPU share unified
LPDDR5X, so memory shows as `[Not Supported]`. The bundled **`fornax-agent`** reads
`nvidia-smi` (power/temp/util) *and* `/proc/meminfo` (unified VRAM) and serves them
as JSON. It's a single zero-dependency Rust binary (~450 KB) and is independent of
your inference server — it works the same whether the box runs vLLM, llama.cpp, or
llama-swap.

**1. (Optional) Confirm your server's metrics.**
- **vLLM** serves Prometheus at `/metrics` by default (`http://host:8000/metrics`).
- **llama-swap** has its own `/metrics` (`http://host:8080/metrics`).
- **plain llama.cpp** needs `llama-server --metrics`.

```bash
curl -s http://localhost:8000/metrics | head    # on the box
```

**2. Build the agent on the box** (needs the Rust toolchain and `nvidia-smi`):

```bash
git clone https://github.com/heath0xFF/fornax && cd fornax/agent
cargo build --release
sudo install -m755 target/release/fornax-agent /usr/local/bin/fornax-agent
```

> Prefer not to install Rust on the box? Cross-compile from another Linux machine
> with `rustup target add aarch64-unknown-linux-gnu` (or `…-musl` for a fully
> static binary) and `scp` the result over.

**3. Run it** — foreground to test, or as a service to persist:

```bash
fornax-agent --port 9099            # serves GET /gpu (binds 0.0.0.0 by default)
curl -s http://localhost:9099/gpu  # sanity check — JSON with VRAM/power/temp
```

```ini
# /etc/systemd/system/fornax-agent.service
[Unit]
Description=Fornax GPU metrics agent
After=network.target

[Service]
ExecStart=/usr/local/bin/fornax-agent --port 9099
Restart=on-failure
User=YOUR_USER

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now fornax-agent
```

**4. Reach it from your Mac.**

- **Same network:** use the box's hostname/IP (`agent_url = "http://host:9099"`,
  `prometheus_url = "http://host:8000/metrics"`); if it has a firewall, allow
  ports `9099` and the server's metrics port.
- **Not on the same network:** SSH-tunnel and point Fornax at `localhost`:
  ```bash
  ssh -N -L 9099:localhost:9099 -L 8000:localhost:8000 you@host
  ```
  The agent has no auth; on an untrusted network run it with `--bind 127.0.0.1`
  and reach it through the tunnel.

## Usage history

Every completed turn is recorded to a local SQLite table — endpoint, model,
prompt/completion tokens, cost, TTFT, and decode tok/s (tool-loop continuations
each count as their own turn). The **Usage** page in the left rail aggregates it:

- **Totals** — tokens, requests, prompt vs. completion, success rate, and total
  cost (shown only when a provider reports it, e.g. OpenRouter)
- **Latency** — TTFT and decode p50·p95 across all recorded turns
- **By model** — requests, tokens, cost, average TTFT and tok/s per model + endpoint
- **Tokens by day** — a daily chart of total token throughput

It's your data only and never leaves your machine. **Refresh** re-reads it;
**Clear** wipes the whole history.

By default usage is kept forever. Set a retention window to prune old rows
automatically (on startup and whenever the Usage page loads) via
**Settings → General** or `config.toml`:

```toml
usage_retention_days = 90   # delete usage older than 90 days; 0 = keep forever
```

## Models

The **Models** page lists the models each saved endpoint currently serves —
fetched live from every backend's `/v1/models` in parallel, with a per-endpoint
loading/error/empty state (so you can see at a glance which servers are up and
what they have loaded). Click any model to point the current chat at that endpoint
+ model and drop straight into the conversation.

## Benchmark

The **Benchmark** tab (the `Live | Benchmark` toggle on the Status page) load-tests
an endpoint so you can compare models or tune concurrency. Pick an endpoint +
model, set the **concurrency**, **request count**, and **max tokens**, and hit
**Run** — Fornax fires that many completions (capped to the concurrency in flight)
and reports:

- **Aggregate throughput** — total completion tokens ÷ wall-clock time
- **TTFT** and **decode tok/s** — p50 and p95 across the run
- **OK / errors**, total wall time, and total tokens
- **Per-request charts** for TTFT and decode

Throughput/decode numbers need the server to report token usage (vLLM, llama.cpp,
oMLX, OpenRouter all do). Limits: up to 64 concurrent, 500 requests, 4096 tokens
per request.

## Configuration

Two layers:

- **Global defaults** — `~/.config/fornax/config.toml`: default endpoint, system
  prompt, sampling params, appearance, hotkeys, usage retention, and saved
  endpoints (with optional keys + metrics config). Edit directly or use
  **Settings** (Cmd/Ctrl+,). See
  [example.config.toml](src-tauri/example.config.toml). Corrupt files are backed
  up rather than silently reset.
- **Per-conversation overrides** — stored in SQLite alongside messages. Changing
  the model/temperature inside a chat affects only that chat. Save a bundle as a
  **preset** to reuse it.

Conversation data lives in `~/Library/Application Support/fornax/fornax.db` (macOS)
or `~/.local/share/fornax/fornax.db` (Linux). API keys are sent as
`Authorization: Bearer`; endpoints that don't need auth omit the key.

## Tools

Tool-capable models can call functions Fornax exposes. Five defaults are seeded
on first launch into Fornax's tools dir — `~/Library/Application Support/fornax/tools/`
on macOS, `~/.config/fornax/tools/` on Linux:

| Tool | Safety | Description |
|---|---|---|
| `read_file` | auto | Reads a file (optional `offset`/`limit`), up to 100 KB |
| `list_directory` | auto | Lists entries with `d/`/`f/` prefixes |
| `search_files` | auto | Recursive regex search; skips dotdirs + binaries |
| `write_file` | confirm | Writes a file; creates parent dirs |
| `run_shell` | confirm | Runs a shell command in the working dir; 5-min cap |

`auto` tools run silently; `confirm` tools show an approval card (Approve /
Approve-all-in-this-conversation / Deny). Each conversation has its own
`working_dir` that relative paths resolve against (defaults to home). Tool chains
are capped at 8 cycles per turn.

Define your own by dropping a `.toml` into that tools dir:

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

New and edited tools hot-reload — they take effect on your next message, no
restart needed (same for the `~/.agents` resources below).

## Skills, commands & the `~/.agents` convention

Fornax also discovers portable [dot-agents](https://www.dot-agents.com/)-style
resources, so commands, skills, and tools you've set up for other agents work
here too. It scans, in increasing precedence:

```text
~/.agents/                    # user-level
~/.agents/local/              # machine-specific overrides (gitignored)
<conversation working_dir>/.agents/   # project-local, overrides the above
```

Each directory may contain:

- **`commands/<name>.md`** → a `/name` slash command. The markdown body is a
  prompt template; `$ARGUMENTS` (or `{{args}}`) is replaced with whatever you type
  after the command, then sent as your message. Optional YAML frontmatter
  (`description:`) shows up in `/help`.

  ```markdown
  ---
  description: Summarize text crisply
  ---
  Summarize the following in 3 bullet points:

  $ARGUMENTS
  ```

- **`skills/<name>/SKILL.md`** → a skill: YAML frontmatter (`name`,
  `description`) plus instructions. Skills work **two ways** — the model sees the
  available skills and can pull one in on demand (it calls a built-in `use_skill`
  tool to load the full instructions), *and* you can inject one yourself by typing
  `/name`.

  ```markdown
  ---
  name: code-review
  description: Review a diff for correctness and clarity
  ---
  When reviewing, check for: off-by-one errors, unhandled errors, …
  ```

- **`tools/<name>.toml`** → the same tool format as above, joined into the
  model's tool set for that conversation.

`/help` lists the commands and skills it found. Project-local entries (under the
conversation's working directory) override your user-level ones by name.

## MCP servers

Fornax is an [MCP](https://modelcontextprotocol.io) client: connect to MCP servers
and their tools join the model's tool set (namespaced `mcp_<server>_<tool>`),
called and approved like any other tool. Configure in **Settings → MCP** (with
live connection status + tool counts) or in `config.toml`:

```toml
[[mcp_servers]]
name = "filesystem"
transport = "stdio"           # "stdio" spawns the command below; "http" uses `url`
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/heath/code"]
enabled = true
auto_approve = false          # true = run this server's tools without prompting

[[mcp_servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "ghp_…" }

# Remote streamable-HTTP server:
[[mcp_servers]]
name = "remote"
transport = "http"
url = "https://example.com/mcp"
headers = { Authorization = "Bearer …" }
```

Editing servers in Settings reconnects them on save (no restart); there's also a
**Reconnect all** button.

## Keyboard & commands

Shortcuts are rebindable in **Settings → Keyboard** (`mod` = Cmd on macOS /
Ctrl elsewhere). Defaults:

| Key | Action |
|---|---|
| Enter / Shift+Enter | Send / newline |
| `mod`+N | New chat |
| `mod`+L | Focus the message input |
| `mod`+F | Find in conversation |
| `mod`+J | Toggle the Artifacts tab in the right dock |
| `mod`+, | Open settings |
| `mod`+. | Stop generation |
| `mod`+Enter | Save & resend (while editing a message) |

**Slash commands** (type in the composer): `/model <name>`, `/temp <0..2>`,
`/system <text>`, `/clear`, `/copy`, `/help` (aliases `/m /t /sys /new /?`).
