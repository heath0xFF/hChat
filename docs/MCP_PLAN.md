# MCP support — implementation plan (`feat/mcp`)

Add [Model Context Protocol](https://modelcontextprotocol.io) client support so
Fornax can connect to MCP servers, discover their tools, and expose those tools to
the model alongside Fornax's native and `~/.agents` tools. Resources/prompts are a
later phase.

## Tech

- **[`rmcp`](https://github.com/modelcontextprotocol/rust-sdk)** — the official
  Rust SDK. Client features we need:
  - `client`
  - `transport-child-process` — spawn a local stdio server (`npx …`, a binary, …)
  - `transport-streamable-http-client-reqwest` — remote HTTP/SSE servers
- Pin a specific `rmcp` version and verify the exact API against it (the surface
  has moved across releases). The notes below are design-level.

## Config (`config.toml`)

A new `[[mcp_servers]]` array, parsed in `core/config.rs` (backward-compatible —
absent = no servers):

```toml
[[mcp_servers]]
name = "filesystem"          # used to namespace its tools
transport = "stdio"          # "stdio" | "http"
command = "npx"              # stdio: argv[0]
args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/heath/code"]
env = { FOO = "bar" }        # stdio: optional extra env
enabled = true               # default true
auto_approve = false         # if true, its tools run without the approval card

[[mcp_servers]]
name = "remote"
transport = "http"
url = "https://example.com/mcp"
headers = { Authorization = "Bearer …" }
enabled = true
```

## Architecture

- **New module `mcp` (`src-tauri/src/mcp/`)** with an `McpManager` held in
  `AppState` as `Arc<McpManager>` (async-internal; `rmcp` clients are async +
  Send). It owns one connected client per enabled server.
- **Lifecycle:** connect all enabled servers in the Tauri `setup` hook (like the
  metrics poller). stdio servers are child processes kept alive for the app's
  lifetime; reconnect on failure with backoff. A `reconnect_mcp(name)` command
  for the settings UI. Children are killed on app exit.
- **Discovery:** on connect, `list_tools` per server; cache `(server, tool_name,
  input_schema, description)`. Refresh on reconnect or a manual refresh.
- **Tool naming:** the model sees a sanitized, namespaced name
  `mcp_<server>_<tool>` (must match `^[A-Za-z0-9_-]{1,64}$`). Keep a reverse map
  `sanitized → (server, original_tool)`.
- **Exposure in a turn:** in `commands::run_turn`, after gathering user +
  `~/.agents` tools, append the MCP tools (as OpenAI tool specs built from each
  tool's input schema). One combined `tools` array goes to the model.
- **Execution routing:** when a tool call name starts with `mcp_`, route it
  through `McpManager::call_tool(server, tool, args)` instead of the local
  builtin/shell path. Map the MCP `CallToolResult` content back to a string for
  the tool-result message. Honor `auto_approve`: if false, reuse the existing
  approval flow (Approve / Approve-all / Deny); if true, run directly.
- **Errors:** a disconnected server's tools are simply omitted that turn; a failed
  call returns an error tool-result (don't abort the turn).

## Frontend

- **Settings → MCP** section (new tab in the sectioned settings page): list
  servers with status (connected / N tools / error), enable/disable, add/remove,
  and a per-server `auto_approve` toggle. Backed by `list_mcp_servers` (status +
  tool counts) and `save_config`.
- MCP tool calls already render through the existing tool-call chips + result
  bubbles + approval card (they're just tools), so no chat-view changes needed.

## Commands (Tauri)

- `list_mcp_servers() -> Vec<McpServerStatus>` (name, transport, connected,
  tool_count, last_error)
- `reconnect_mcp(name)` / `reconnect_all_mcp()`
- (config add/remove/toggle goes through the existing `save_config`, then trigger
  a reconnect)

## Phasing

1. **Core (stdio):** config parsing, `McpManager` + child-process transport, tool
   discovery, run_turn merge + execution routing + approval. Ship with one real
   server (e.g. `server-filesystem`) verified end-to-end.
2. **HTTP transport** + headers/auth.
3. **Settings UI** (status, enable, add/remove, auto_approve).
4. **Resources & prompts** (optional) — expose MCP resources as attachable
   context and prompts as `/`-style commands.

## Open decisions

- Default approval for MCP tools: **Confirm** (safer — they can do anything),
  overridable per server via `auto_approve`. Confirm here.
- Connect eagerly at startup vs lazily on first use. Eager is simpler and gives
  immediate status; start there.
- Whether to surface MCP **resources** in chat (Phase 4) or skip.

## Verification

- A stdio server (`@modelcontextprotocol/server-filesystem`) connects, its tools
  appear in a turn, the model calls one, the approval card shows (unless
  `auto_approve`), and the result flows back. Unit test the name
  sanitization/reverse-map. `cargo clippy -D warnings`, `tsc`, `vite` clean.
