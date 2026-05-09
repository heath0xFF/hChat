use crate::message::{ContentPart, Message, Role};
use rusqlite::{Connection, params};
use std::path::PathBuf;
use std::time::Duration;

/// Returned by `list_conversations`: enough to render the sidebar (pinned-first
/// sort, optional pin-marker glyph) without a second query per row.
pub struct Conversation {
    pub id: i64,
    pub title: String,
    pub pinned: bool,
}

/// Per-conversation settings snapshot. All fields are `Option` because a
/// conversation may have been created before any setting was customized
/// (column NULL → use the global default).
#[derive(Clone, Debug, Default)]
pub struct ConversationSettings {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub use_max_tokens: bool,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Vec<String>,
    pub endpoint: Option<String>,
}

pub struct Storage {
    conn: Connection,
}

impl Storage {
    pub fn new() -> Self {
        let path = Self::db_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
            }
        }
        let conn = Connection::open(&path).expect("Failed to open database");
        // Enable foreign keys and WAL mode for better concurrency
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .ok();
        // Wait up to 5s for a lock instead of immediately returning SQLITE_BUSY.
        // Without this, two hChat windows writing concurrently race and the
        // loser silently drops its write (we use `.ok()` on most calls).
        let _ = conn.busy_timeout(Duration::from_secs(5));
        let storage = Self { conn };
        storage.init_tables();
        storage
    }

    fn db_path() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("hchat")
            .join("hchat.db")
    }

    fn init_tables(&self) {
        // schema_version is the migration cursor. Each migration runs at most
        // once; a fresh DB jumps straight to the latest version after the
        // initial CREATE block.
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    model TEXT,
                    system_prompt TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    use_max_tokens INTEGER NOT NULL DEFAULT 0,
                    top_p REAL,
                    frequency_penalty REAL,
                    presence_penalty REAL,
                    stop_sequences TEXT,
                    endpoint TEXT,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    draft TEXT,
                    auto_titled INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    parent_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
                    branch_index INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    model TEXT,
                    system_prompt TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    use_max_tokens INTEGER NOT NULL DEFAULT 0,
                    top_p REAL,
                    frequency_penalty REAL,
                    presence_penalty REAL,
                    stop_sequences TEXT,
                    endpoint TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conv_position ON messages(conversation_id, position);
                CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_pinned_updated ON conversations(pinned DESC, updated_at DESC);
                INSERT OR IGNORE INTO schema_version (version) VALUES (2);",
            )
            .expect("Failed to create tables");
    }

    /// Current schema version, or 0 if the table is empty/missing.
    #[allow(dead_code)]
    fn schema_version(&self) -> i64 {
        self.conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0)
    }

    pub fn create_conversation(&self, title: &str) -> Result<i64, String> {
        self.conn
            .execute(
                "INSERT INTO conversations (title) VALUES (?1)",
                params![title],
            )
            .map_err(|e| format!("Failed to create conversation: {e}"))?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn update_conversation_title(&self, id: i64, title: &str) {
        self.conn
            .execute(
                "UPDATE conversations SET title = ?1, updated_at = datetime('now') WHERE id = ?2",
                params![title, id],
            )
            .ok();
    }

    pub fn list_conversations(&self) -> Vec<Conversation> {
        let mut stmt = match self.conn.prepare(
            "SELECT id, title, pinned FROM conversations ORDER BY pinned DESC, updated_at DESC",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        match stmt.query_map([], |row| {
            Ok(Conversation {
                id: row.get(0)?,
                title: row.get(1)?,
                pinned: row.get::<_, i64>(2)? != 0,
            })
        }) {
            Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn delete_conversation(&self, id: i64) {
        // Use a transaction so both deletes succeed or both fail
        if let Ok(tx) = self.conn.unchecked_transaction() {
            let r1 = tx.execute(
                "DELETE FROM messages WHERE conversation_id = ?1",
                params![id],
            );
            let r2 = tx.execute("DELETE FROM conversations WHERE id = ?1", params![id]);

            if r1.is_ok() && r2.is_ok() {
                tx.commit().ok();
            }
            // tx drops and rolls back on failure
        }
    }

    pub fn save_messages(
        &self,
        conversation_id: i64,
        messages: &[Message],
    ) -> Result<(), String> {
        // Use a transaction so delete + re-insert is atomic
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| format!("Failed to start transaction: {e}"))?;

        tx.execute(
            "DELETE FROM messages WHERE conversation_id = ?1",
            params![conversation_id],
        )
        .map_err(|e| format!("Failed to clear messages: {e}"))?;

        {
            let mut stmt = tx
                .prepare("INSERT INTO messages (conversation_id, role, content, position, created_at) VALUES (?1, ?2, ?3, ?4, ?5)")
                .map_err(|e| format!("Failed to prepare insert: {e}"))?;

            for (i, msg) in messages.iter().enumerate() {
                let role_str = match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };
                let content_json = serde_json::to_string(&msg.content)
                    .map_err(|e| format!("Failed to encode content: {e}"))?;
                stmt.execute(params![
                    conversation_id,
                    role_str,
                    content_json,
                    i,
                    msg.created_at
                ])
                .map_err(|e| format!("Failed to insert message: {e}"))?;
            }
        }

        tx.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?1",
            params![conversation_id],
        )
        .map_err(|e| format!("Failed to update timestamp: {e}"))?;

        // Don't swallow commit failures: a failed commit means the user thinks
        // their conversation persisted when SQLite rejected it (disk full,
        // SQLITE_BUSY past timeout, etc.). The caller surfaces this via the
        // app's error banner.
        tx.commit().map_err(|e| format!("Failed to commit: {e}"))
    }

    pub fn load_messages(&self, conversation_id: i64) -> Vec<Message> {
        let mut stmt = match self.conn.prepare(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ?1 ORDER BY position",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        stmt.query_map(params![conversation_id], |row| {
            let role_str: String = row.get(0)?;
            let role = match role_str.as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => Role::Assistant,
            };
            let content_raw: String = row.get(1)?;
            let created_at: Option<i64> = row.get(2).ok();
            // Parse the stored JSON into ContentParts. Fall back to wrapping
            // raw text if the row was somehow written as plain text (defensive
            // — shouldn't happen post-v2, but cheap to handle).
            let content: Vec<ContentPart> = serde_json::from_str(&content_raw).unwrap_or_else(|_| {
                vec![ContentPart::Text {
                    text: content_raw,
                }]
            });
            Ok(Message {
                role,
                content,
                created_at,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    pub fn search(&self, query: &str) -> Vec<(i64, String, String)> {
        // Two-phase search: SQL LIKE prunes candidate rows by raw content
        // (which is JSON-encoded), then we parse each match in Rust and only
        // keep rows whose *text* parts contain the query. Without the second
        // phase, searching for "text" or "image_url" would match every
        // multimodal row because of the JSON keys.
        let q_lower = query.to_ascii_lowercase();
        let escaped = query
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("%{escaped}%");
        let mut stmt = match self.conn.prepare(
            "SELECT c.id, c.title, m.content FROM messages m
             JOIN conversations c ON c.id = m.conversation_id
             WHERE m.content LIKE ?1 ESCAPE '\\'
             ORDER BY c.updated_at DESC
             LIMIT 200",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let candidates: Vec<(i64, String, String)> = stmt
            .query_map(params![pattern], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

        candidates
            .into_iter()
            .filter_map(|(id, title, raw)| {
                let parts: Vec<ContentPart> =
                    serde_json::from_str(&raw).unwrap_or_else(|_| {
                        vec![ContentPart::Text { text: raw.clone() }]
                    });
                let mut text = String::new();
                for p in &parts {
                    if let ContentPart::Text { text: t } = p {
                        text.push_str(t);
                        text.push('\n');
                    }
                }
                if text.to_ascii_lowercase().contains(&q_lower) {
                    Some((id, title, text))
                } else {
                    None
                }
            })
            .take(50)
            .collect()
    }

    pub fn export_markdown(&self, conversation_id: i64) -> String {
        let messages = self.load_messages(conversation_id);
        let mut out = String::new();

        let title: String = self
            .conn
            .query_row(
                "SELECT title FROM conversations WHERE id = ?1",
                params![conversation_id],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| "Conversation".to_string());

        out.push_str(&format!("# {title}\n\n"));

        for msg in &messages {
            let role = match msg.role {
                Role::User => "**You**",
                Role::Assistant => "**AI**",
                Role::System => "**System**",
            };
            let body = msg.text_str();
            let image_count = msg.images().count();
            let suffix = if image_count > 0 {
                format!("\n\n_({image_count} image attachment(s))_")
            } else {
                String::new()
            };
            out.push_str(&format!("{role}:\n\n{body}{suffix}\n\n---\n\n"));
        }

        out
    }

    // ---- per-conversation settings ----

    pub fn load_conversation_settings(&self, id: i64) -> ConversationSettings {
        self.conn
            .query_row(
                "SELECT model, system_prompt, temperature, max_tokens, use_max_tokens,
                        top_p, frequency_penalty, presence_penalty, stop_sequences, endpoint
                 FROM conversations WHERE id = ?1",
                params![id],
                |row| {
                    let stop_json: Option<String> = row.get(8).ok();
                    let stop_sequences = stop_json
                        .as_deref()
                        .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                        .unwrap_or_default();
                    Ok(ConversationSettings {
                        model: row.get(0).ok(),
                        system_prompt: row.get(1).ok(),
                        temperature: row.get::<_, Option<f64>>(2).ok().flatten().map(|v| v as f32),
                        max_tokens: row.get::<_, Option<i64>>(3).ok().flatten().map(|v| v as u32),
                        use_max_tokens: row.get::<_, i64>(4).unwrap_or(0) != 0,
                        top_p: row.get::<_, Option<f64>>(5).ok().flatten().map(|v| v as f32),
                        frequency_penalty: row
                            .get::<_, Option<f64>>(6)
                            .ok()
                            .flatten()
                            .map(|v| v as f32),
                        presence_penalty: row
                            .get::<_, Option<f64>>(7)
                            .ok()
                            .flatten()
                            .map(|v| v as f32),
                        stop_sequences,
                        endpoint: row.get(9).ok(),
                    })
                },
            )
            .unwrap_or_default()
    }

    pub fn save_conversation_settings(&self, id: i64, s: &ConversationSettings) {
        let stop_json = if s.stop_sequences.is_empty() {
            None
        } else {
            serde_json::to_string(&s.stop_sequences).ok()
        };
        self.conn
            .execute(
                "UPDATE conversations
                 SET model = ?1, system_prompt = ?2, temperature = ?3, max_tokens = ?4,
                     use_max_tokens = ?5, top_p = ?6, frequency_penalty = ?7,
                     presence_penalty = ?8, stop_sequences = ?9, endpoint = ?10,
                     updated_at = datetime('now')
                 WHERE id = ?11",
                params![
                    s.model,
                    s.system_prompt,
                    s.temperature.map(|v| v as f64),
                    s.max_tokens.map(|v| v as i64),
                    s.use_max_tokens as i64,
                    s.top_p.map(|v| v as f64),
                    s.frequency_penalty.map(|v| v as f64),
                    s.presence_penalty.map(|v| v as f64),
                    stop_json,
                    s.endpoint,
                    id,
                ],
            )
            .ok();
    }

    pub fn set_pinned(&self, id: i64, pinned: bool) {
        self.conn
            .execute(
                "UPDATE conversations SET pinned = ?1 WHERE id = ?2",
                params![pinned as i64, id],
            )
            .ok();
    }

    pub fn save_draft(&self, id: i64, draft: &str) {
        self.conn
            .execute(
                "UPDATE conversations SET draft = ?1 WHERE id = ?2",
                params![draft, id],
            )
            .ok();
    }

    /// Returns true if the conversation has not yet been auto-titled. Used
    /// to gate the one-shot auto-title call so manual renames stick.
    pub fn needs_auto_title(&self, id: i64) -> bool {
        self.conn
            .query_row(
                "SELECT auto_titled FROM conversations WHERE id = ?1",
                params![id],
                |row| row.get::<_, i64>(0),
            )
            .map(|v| v == 0)
            .unwrap_or(false)
    }

    pub fn mark_auto_titled(&self, id: i64) {
        self.conn
            .execute(
                "UPDATE conversations SET auto_titled = 1 WHERE id = ?1",
                params![id],
            )
            .ok();
    }

    pub fn load_draft(&self, id: i64) -> Option<String> {
        self.conn
            .query_row(
                "SELECT draft FROM conversations WHERE id = ?1",
                params![id],
                |row| row.get::<_, Option<String>>(0),
            )
            .ok()
            .flatten()
            .filter(|s| !s.is_empty())
    }

    // ---- presets ----

    pub fn list_presets(&self) -> Vec<Preset> {
        let mut stmt = match self.conn.prepare(
            "SELECT id, name, model, system_prompt, temperature, max_tokens, use_max_tokens,
                    top_p, frequency_penalty, presence_penalty, stop_sequences, endpoint
             FROM presets ORDER BY name COLLATE NOCASE",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        stmt.query_map([], |row| {
            let stop_json: Option<String> = row.get(10).ok();
            let stop_sequences = stop_json
                .as_deref()
                .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                .unwrap_or_default();
            Ok(Preset {
                id: row.get(0)?,
                name: row.get(1)?,
                settings: ConversationSettings {
                    model: row.get(2).ok(),
                    system_prompt: row.get(3).ok(),
                    temperature: row.get::<_, Option<f64>>(4).ok().flatten().map(|v| v as f32),
                    max_tokens: row.get::<_, Option<i64>>(5).ok().flatten().map(|v| v as u32),
                    use_max_tokens: row.get::<_, i64>(6).unwrap_or(0) != 0,
                    top_p: row.get::<_, Option<f64>>(7).ok().flatten().map(|v| v as f32),
                    frequency_penalty: row
                        .get::<_, Option<f64>>(8)
                        .ok()
                        .flatten()
                        .map(|v| v as f32),
                    presence_penalty: row
                        .get::<_, Option<f64>>(9)
                        .ok()
                        .flatten()
                        .map(|v| v as f32),
                    stop_sequences,
                    endpoint: row.get(11).ok(),
                },
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    pub fn create_preset(&self, name: &str, s: &ConversationSettings) -> Result<i64, String> {
        let stop_json = if s.stop_sequences.is_empty() {
            None
        } else {
            serde_json::to_string(&s.stop_sequences).ok()
        };
        self.conn
            .execute(
                "INSERT INTO presets
                 (name, model, system_prompt, temperature, max_tokens, use_max_tokens,
                  top_p, frequency_penalty, presence_penalty, stop_sequences, endpoint)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    name,
                    s.model,
                    s.system_prompt,
                    s.temperature.map(|v| v as f64),
                    s.max_tokens.map(|v| v as i64),
                    s.use_max_tokens as i64,
                    s.top_p.map(|v| v as f64),
                    s.frequency_penalty.map(|v| v as f64),
                    s.presence_penalty.map(|v| v as f64),
                    stop_json,
                    s.endpoint,
                ],
            )
            .map_err(|e| format!("Failed to create preset: {e}"))?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn delete_preset(&self, id: i64) {
        self.conn
            .execute("DELETE FROM presets WHERE id = ?1", params![id])
            .ok();
    }
}

#[derive(Clone)]
pub struct Preset {
    pub id: i64,
    pub name: String,
    pub settings: ConversationSettings,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentPart, ImageUrl, Message, Role};

    fn mem_storage() -> Storage {
        let conn = Connection::open_in_memory().expect("open :memory:");
        conn.execute_batch("PRAGMA foreign_keys=ON;").ok();
        let storage = Storage { conn };
        storage.init_tables();
        storage
    }

    #[test]
    fn schema_is_at_v2_after_init() {
        let s = mem_storage();
        assert_eq!(s.schema_version(), 2);
    }

    #[test]
    fn round_trips_text_messages() {
        let s = mem_storage();
        let id = s.create_conversation("hello").unwrap();
        let msgs = vec![
            Message::text(Role::User, "hi".to_string()),
            Message::text(Role::Assistant, "hello!".to_string()),
        ];
        s.save_messages(id, &msgs).unwrap();
        let loaded = s.load_messages(id);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text_str(), "hi");
        assert_eq!(loaded[1].text_str(), "hello!");
    }

    #[test]
    fn round_trips_messages_with_images() {
        let s = mem_storage();
        let id = s.create_conversation("vision").unwrap();
        let parts = vec![
            ContentPart::Text {
                text: "what's in this".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,AAAA".to_string(),
                    detail: None,
                },
            },
        ];
        let msgs = vec![Message::from_parts(Role::User, parts.clone())];
        s.save_messages(id, &msgs).unwrap();
        let loaded = s.load_messages(id);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, parts);
    }

    #[test]
    fn search_post_filters_json_keys() {
        // Without the post-filter, searching for "text" or "image_url" would
        // match every multimodal row because of the JSON keys themselves.
        let s = mem_storage();
        let id = s.create_conversation("vision conv").unwrap();
        s.save_messages(
            id,
            &[Message::from_parts(
                Role::User,
                vec![
                    ContentPart::Text {
                        text: "hello world".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,AAAA".to_string(),
                            detail: None,
                        },
                    },
                ],
            )],
        )
        .unwrap();
        // "image_url" is a JSON key but not in any text part — must NOT match.
        assert!(s.search("image_url").is_empty());
        // The actual user text MUST match.
        assert_eq!(s.search("hello world").len(), 1);
    }

    #[test]
    fn round_trips_per_conversation_settings() {
        let s = mem_storage();
        let id = s.create_conversation("settings test").unwrap();
        let original = ConversationSettings {
            model: Some("gpt-4o".to_string()),
            system_prompt: Some("be terse".to_string()),
            temperature: Some(0.42),
            max_tokens: Some(1234),
            use_max_tokens: true,
            top_p: Some(0.9),
            frequency_penalty: None,
            presence_penalty: Some(0.5),
            stop_sequences: vec!["END".to_string(), "STOP".to_string()],
            endpoint: Some("https://openrouter.ai/api/v1".to_string()),
        };
        s.save_conversation_settings(id, &original);
        let loaded = s.load_conversation_settings(id);
        assert_eq!(loaded.model, original.model);
        assert_eq!(loaded.system_prompt, original.system_prompt);
        assert_eq!(loaded.temperature, original.temperature);
        assert_eq!(loaded.max_tokens, original.max_tokens);
        assert_eq!(loaded.use_max_tokens, original.use_max_tokens);
        assert_eq!(loaded.top_p, original.top_p);
        assert_eq!(loaded.frequency_penalty, original.frequency_penalty);
        assert_eq!(loaded.presence_penalty, original.presence_penalty);
        assert_eq!(loaded.stop_sequences, original.stop_sequences);
        assert_eq!(loaded.endpoint, original.endpoint);
    }

    #[test]
    fn pinned_conversations_sort_to_top() {
        let s = mem_storage();
        let a = s.create_conversation("first").unwrap();
        let b = s.create_conversation("second").unwrap();
        let c = s.create_conversation("third").unwrap();
        s.set_pinned(a, true);
        let listed = s.list_conversations();
        assert_eq!(listed.len(), 3);
        // Pinned conversation is always first.
        assert_eq!(listed[0].id, a);
        assert!(listed[0].pinned);
        // The remaining two are the unpinned ones — order between them
        // depends on updated_at resolution (datetime('now') is per-second), so
        // don't assert order, just membership.
        let rest: Vec<i64> = listed[1..].iter().map(|c| c.id).collect();
        assert!(rest.contains(&b));
        assert!(rest.contains(&c));
        assert!(!listed[1].pinned);
        assert!(!listed[2].pinned);
    }

    #[test]
    fn drafts_round_trip_and_clear() {
        let s = mem_storage();
        let id = s.create_conversation("drafty").unwrap();
        s.save_draft(id, "in progress");
        assert_eq!(s.load_draft(id).as_deref(), Some("in progress"));
        s.save_draft(id, "");
        // Empty drafts read back as None — we treat "" as "no draft".
        assert_eq!(s.load_draft(id), None);
    }

    #[test]
    fn auto_title_flag_is_one_shot() {
        let s = mem_storage();
        let id = s.create_conversation("t").unwrap();
        assert!(s.needs_auto_title(id));
        s.mark_auto_titled(id);
        assert!(!s.needs_auto_title(id));
    }

    #[test]
    fn presets_crud() {
        let s = mem_storage();
        assert!(s.list_presets().is_empty());
        let settings = ConversationSettings {
            model: Some("haiku".to_string()),
            system_prompt: Some("be concise".to_string()),
            temperature: Some(0.7),
            ..ConversationSettings::default()
        };
        let pid = s.create_preset("Concise", &settings).unwrap();
        let listed = s.list_presets();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Concise");
        assert_eq!(listed[0].settings.model.as_deref(), Some("haiku"));
        s.delete_preset(pid);
        assert!(s.list_presets().is_empty());
    }
}
