use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Unix epoch milliseconds. Session-local — never sent to the API and not
    /// persisted to SQLite (Phase 2 will migrate the schema). Set when a
    /// message is added in the current session; messages loaded from disk
    /// have `None` and the UI hides the hover timestamp for them.
    #[serde(skip_serializing, default)]
    pub created_at: Option<i64>,
}

impl Message {
    pub fn new(role: Role, content: String) -> Self {
        Self {
            role,
            content,
            created_at: Some(now_ms()),
        }
    }
}

pub fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
