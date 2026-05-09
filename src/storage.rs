use crate::message::{Message, Role};
use rusqlite::{Connection, params};
use std::path::PathBuf;
use std::time::Duration;

pub struct Conversation {
    pub id: i64,
    pub title: String,
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
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );",
            )
            .expect("Failed to create tables");
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
        let mut stmt = match self
            .conn
            .prepare("SELECT id, title FROM conversations ORDER BY updated_at DESC")
        {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        match stmt.query_map([], |row| {
            Ok(Conversation {
                id: row.get(0)?,
                title: row.get(1)?,
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
                .prepare("INSERT INTO messages (conversation_id, role, content, position) VALUES (?1, ?2, ?3, ?4)")
                .map_err(|e| format!("Failed to prepare insert: {e}"))?;

            for (i, msg) in messages.iter().enumerate() {
                let role_str = match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };
                stmt.execute(params![conversation_id, role_str, msg.content, i])
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
            "SELECT role, content FROM messages WHERE conversation_id = ?1 ORDER BY position",
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
            Ok(Message {
                role,
                content: row.get(1)?,
                created_at: None,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    pub fn search(&self, query: &str) -> Vec<(i64, String, String)> {
        // Escape LIKE special characters
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
             LIMIT 50",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        stmt.query_map(params![pattern], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
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
            out.push_str(&format!("{role}:\n\n{}\n\n---\n\n", msg.content));
        }

        out
    }
}
