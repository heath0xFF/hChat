//! MCP client manager. Connects to configured MCP servers (stdio child
//! processes or streamable-HTTP), discovers their tools, and routes the model's
//! tool calls to the right server. Discovered tools are merged into the model's
//! tool set in `commands::run_turn` under namespaced names (`mcp_<server>_<tool>`).

use crate::config::McpServer;
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

const TOOL_PREFIX: &str = "mcp_";

#[derive(Clone)]
struct ToolInfo {
    server: String,
    original: String,
    sanitized: String,
    description: String,
    input_schema: serde_json::Value,
}

struct Conn {
    client: RunningService<RoleClient, ()>,
    auto_approve: bool,
}

/// Per-server connection status surfaced to the Settings UI.
#[derive(Serialize, Clone)]
pub struct McpStatus {
    pub name: String,
    pub transport: String,
    pub connected: bool,
    pub tool_count: usize,
    pub auto_approve: bool,
    pub error: Option<String>,
}

#[derive(Default)]
pub struct McpManager {
    conns: Mutex<HashMap<String, Conn>>,
    tools: Mutex<Vec<ToolInfo>>,
    status: Mutex<Vec<McpStatus>>,
}

/// Build the model-facing tool name: `mcp_<server>_<tool>`, sanitized to the
/// OpenAI tool-name charset and capped at 64 chars.
fn sanitize(server: &str, tool: &str) -> String {
    let mut s: String = format!("{TOOL_PREFIX}{server}_{tool}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    s.truncate(64);
    s
}

impl McpManager {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn is_mcp_tool(name: &str) -> bool {
        name.starts_with(TOOL_PREFIX)
    }

    /// Connect (or reconnect) every enabled server, replacing current state.
    /// Existing connections are dropped first (stdio children terminate).
    pub async fn connect_all(&self, servers: Vec<McpServer>) {
        self.conns.lock().await.clear();
        self.tools.lock().await.clear();
        self.status.lock().await.clear();

        for s in servers {
            let mut status = McpStatus {
                name: s.name.clone(),
                transport: s.transport.clone(),
                connected: false,
                tool_count: 0,
                auto_approve: s.auto_approve,
                error: None,
            };
            if s.enabled {
                match self.connect_one(&s).await {
                    Ok(count) => {
                        status.connected = true;
                        status.tool_count = count;
                    }
                    Err(e) => status.error = Some(e),
                }
            } else {
                status.error = Some("disabled".to_string());
            }
            self.status.lock().await.push(status);
        }
    }

    async fn connect_one(&self, s: &McpServer) -> Result<usize, String> {
        let client = self.dial(s).await?;
        let tools = client
            .peer()
            .list_all_tools()
            .await
            .map_err(|e| format!("list_tools: {e}"))?;

        let mut infos = Vec::new();
        for t in tools {
            let original = t.name.to_string();
            let schema = serde_json::to_value(&*t.input_schema)
                .unwrap_or_else(|_| serde_json::json!({ "type": "object" }));
            infos.push(ToolInfo {
                server: s.name.clone(),
                sanitized: sanitize(&s.name, &original),
                original,
                description: t.description.map(|d| d.to_string()).unwrap_or_default(),
                input_schema: schema,
            });
        }
        let count = infos.len();
        self.tools.lock().await.extend(infos);
        self.conns.lock().await.insert(
            s.name.clone(),
            Conn {
                client,
                auto_approve: s.auto_approve,
            },
        );
        Ok(count)
    }

    async fn dial(&self, s: &McpServer) -> Result<RunningService<RoleClient, ()>, String> {
        match s.transport.as_str() {
            "stdio" => {
                let cmd = s
                    .command
                    .clone()
                    .ok_or("stdio server missing `command`")?;
                let mut command = tokio::process::Command::new(cmd);
                command.args(&s.args);
                for (k, v) in &s.env {
                    command.env(k, v);
                }
                let transport =
                    TokioChildProcess::new(command).map_err(|e| format!("spawn: {e}"))?;
                ().serve(transport).await.map_err(|e| format!("connect: {e}"))
            }
            "http" => Err("http transport is not enabled yet — use transport = \"stdio\"".to_string()),
            other => Err(format!("unknown transport: {other}")),
        }
    }

    /// OpenAI tool specs for every connected MCP tool.
    pub async fn tool_specs(&self) -> Vec<serde_json::Value> {
        self.tools
            .lock()
            .await
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.sanitized,
                        "description": format!("[{}] {}", t.server, t.description),
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect()
    }

    /// Whether the server behind this tool is set to auto-approve.
    pub async fn auto_approve(&self, sanitized: &str) -> bool {
        let server = self
            .tools
            .lock()
            .await
            .iter()
            .find(|t| t.sanitized == sanitized)
            .map(|t| t.server.clone());
        match server {
            Some(server) => self
                .conns
                .lock()
                .await
                .get(&server)
                .map(|c| c.auto_approve)
                .unwrap_or(false),
            None => false,
        }
    }

    /// Invoke an MCP tool by its sanitized name. Returns `(text, is_error)`.
    pub async fn call(&self, sanitized: &str, args: serde_json::Value) -> (String, bool) {
        let (server, original) = {
            let tools = self.tools.lock().await;
            match tools.iter().find(|t| t.sanitized == sanitized) {
                Some(t) => (t.server.clone(), t.original.clone()),
                None => return (format!("unknown MCP tool: {sanitized}"), true),
            }
        };
        let arguments = match args {
            serde_json::Value::Object(m) => Some(m),
            _ => None,
        };
        let conns = self.conns.lock().await;
        let Some(conn) = conns.get(&server) else {
            return (format!("MCP server '{server}' not connected"), true);
        };
        let mut params = CallToolRequestParams::new(original);
        if let Some(map) = arguments {
            params = params.with_arguments(map);
        }
        match conn.client.peer().call_tool(params).await {
            Ok(res) => {
                let text = res
                    .content
                    .iter()
                    .filter_map(|c| c.as_text().map(|t| t.text.clone()))
                    .collect::<Vec<_>>()
                    .join("\n");
                let text = if text.is_empty() {
                    "(no text content)".to_string()
                } else {
                    text
                };
                (text, res.is_error.unwrap_or(false))
            }
            Err(e) => (format!("MCP call failed: {e}"), true),
        }
    }

    pub async fn status(&self) -> Vec<McpStatus> {
        self.status.lock().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::McpServer;
    use std::collections::HashMap;

    #[test]
    fn sanitize_namespaces_and_caps() {
        let s = sanitize("file-sys", "read.file");
        assert!(s.starts_with("mcp_file-sys_read_file"));
        assert!(s.len() <= 64);
    }

    // Spawns `npx @modelcontextprotocol/server-everything` over stdio and checks
    // the full connect → list_tools → call_tool path. Run with:
    //   cargo test --lib mcp -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "spawns npx; network on first run"]
    async fn connects_stdio_lists_and_calls() {
        let mgr = McpManager::new();
        let server = McpServer {
            name: "everything".into(),
            transport: "stdio".into(),
            command: Some("npx".into()),
            args: vec!["-y".into(), "@modelcontextprotocol/server-everything".into()],
            env: HashMap::new(),
            url: None,
            headers: HashMap::new(),
            enabled: true,
            auto_approve: true,
        };
        mgr.connect_all(vec![server]).await;
        let status = mgr.status().await;
        eprintln!(
            "status: connected={} tools={} err={:?}",
            status[0].connected, status[0].tool_count, status[0].error
        );
        assert!(status[0].connected, "connect failed: {:?}", status[0].error);

        let specs = mgr.tool_specs().await;
        eprintln!("discovered {} tools", specs.len());
        assert!(!specs.is_empty());

        // server-everything exposes an `echo` tool.
        let (text, is_err) = mgr
            .call(
                "mcp_everything_echo",
                serde_json::json!({ "message": "hi from hChat" }),
            )
            .await;
        eprintln!("echo -> is_err={is_err} text={text}");
        assert!(!is_err, "echo errored: {text}");
        assert!(text.contains("hi from hChat"));
    }
}
