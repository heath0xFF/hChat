use crate::config::Config;
use crate::mcp::McpManager;
use crate::storage::Storage;
use crate::tools;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// A parked tool-approval request: the oneshot to fulfill plus the context
/// needed to honor an "approve all in this conversation" decision.
pub struct PendingApproval {
    pub tx: oneshot::Sender<bool>,
    pub conversation_id: i64,
    pub tool: String,
}

/// All long-lived app state, managed by Tauri and shared across commands.
///
/// `Storage` wraps a rusqlite `Connection` (Send but not Sync), so it lives
/// behind a `Mutex`. Every command locks briefly for synchronous DB work and
/// drops the guard before any `.await` — never hold a lock across a suspend
/// point.
pub struct AppState {
    pub storage: Mutex<Storage>,
    pub config: Mutex<Config>,
    /// Cancellation token for the in-flight chat stream, if any. Replaced on
    /// each `send_message`; `cancel_stream` fires it.
    pub cancel: Mutex<Option<CancellationToken>>,
    /// Tool calls awaiting user approval, keyed by tool_call id. The streaming
    /// task parks a oneshot here and emits a `tool_approval` event; the
    /// `resolve_tool` command fulfills it with the user's decision.
    pub pending_approvals: Mutex<HashMap<String, PendingApproval>>,
    /// `(conversation_id, tool_name)` pairs the user chose to auto-approve for
    /// the rest of the session via "approve all in this conversation".
    pub auto_approved: Mutex<HashSet<(i64, String)>>,
    /// MCP client connections + discovered tools. Connected asynchronously after
    /// startup (see `lib.rs` setup).
    pub mcp: Arc<McpManager>,
}

impl AppState {
    pub fn new() -> Self {
        let config = Config::load();
        let storage = Storage::new();
        // Seed the default tools on first launch; tools are then (re)loaded from
        // disk on every turn, so edits take effect without a restart.
        tools::seed_defaults_if_empty(&tools::user_tools_dir());
        Self {
            storage: Mutex::new(storage),
            config: Mutex::new(config),
            cancel: Mutex::new(None),
            pending_approvals: Mutex::new(HashMap::new()),
            auto_approved: Mutex::new(HashSet::new()),
            mcp: McpManager::new(),
        }
    }

    /// Resolve the API key configured for `endpoint` (exact URL match against
    /// `saved_endpoints`). Returns `None` when the endpoint isn't saved or has
    /// no key — local servers (oMLX, llama.cpp) typically need none.
    pub fn api_key_for(&self, endpoint: &str) -> Option<String> {
        let cfg = self.config.lock().unwrap();
        cfg.saved_endpoints
            .iter()
            .find(|e| e.url == endpoint)
            .and_then(|e| e.api_key.clone())
    }
}
