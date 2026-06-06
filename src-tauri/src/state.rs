use crate::config::Config;
use crate::storage::Storage;
use crate::tools::{self, ToolDef};
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// All long-lived app state, managed by Tauri and shared across commands.
///
/// `Storage` wraps a rusqlite `Connection` (Send but not Sync), so it lives
/// behind a `Mutex`. Every command locks briefly for synchronous DB work and
/// drops the guard before any `.await` — never hold a lock across a suspend
/// point.
pub struct AppState {
    pub storage: Mutex<Storage>,
    pub config: Mutex<Config>,
    pub tools: Mutex<Vec<ToolDef>>,
    /// Cancellation token for the in-flight chat stream, if any. Replaced on
    /// each `send_message`; `cancel_stream` fires it.
    pub cancel: Mutex<Option<CancellationToken>>,
    /// Tool calls awaiting user approval, keyed by tool_call id. The streaming
    /// task parks a oneshot here and emits a `tool_approval` event; the
    /// `resolve_tool` command fulfills it with the user's decision.
    pub pending_approvals: Mutex<HashMap<String, oneshot::Sender<bool>>>,
}

impl AppState {
    pub fn new() -> Self {
        let config = Config::load();
        let storage = Storage::new();
        let tools_dir = tools::user_tools_dir();
        tools::seed_defaults_if_empty(&tools_dir);
        let loaded = tools::load_from_dir(&tools_dir);
        Self {
            storage: Mutex::new(storage),
            config: Mutex::new(config),
            tools: Mutex::new(loaded),
            cancel: Mutex::new(None),
            pending_approvals: Mutex::new(HashMap::new()),
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
