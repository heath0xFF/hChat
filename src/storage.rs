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
        // Step 1: schema_version + the *baseline 0.6.0 shape* of every
        // table. Newer columns are ALTERed in below. This split matters
        // because `CREATE TABLE IF NOT EXISTS conversations (... pinned ...)`
        // is a no-op when the table already exists from a prior version
        // *without* `pinned` — so the old code crashed on first launch
        // for any user upgrading from 0.6.x without wiping the DB.
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
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );",
            )
            .expect("Failed to create baseline tables");

        // Step 2: ensure all v2 columns exist, ALTERing them in if missing.
        // Idempotent — re-runs cheaply on every launch.
        self.ensure_v2_columns();

        // Step 3: indexes + version stamp. Safe now that columns exist.
        self.conn
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_messages_conv_position ON messages(conversation_id, position);
                 CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id);
                 CREATE INDEX IF NOT EXISTS idx_conversations_pinned_updated ON conversations(pinned DESC, updated_at DESC);
                 INSERT OR IGNORE INTO schema_version (version) VALUES (2);",
            )
            .expect("Failed to create indexes");
    }

    /// Bring every table up to the v2 column set by ALTERing in any column
    /// that's missing. The list of (table, column, type) is the source of
    /// truth — adding a future column is one line here, no schema-version
    /// math required.
    ///
    /// Errors are swallowed per-column on purpose: ALTER TABLE failures on
    /// a single column shouldn't take down launch (the user can still read
    /// existing data), and the most common failure ("column already exists"
    /// when our `pragma_table_info` race lost) is benign.
    fn ensure_v2_columns(&self) {
        const CONV_COLS: &[(&str, &str)] = &[
            ("model", "TEXT"),
            ("system_prompt", "TEXT"),
            ("temperature", "REAL"),
            ("max_tokens", "INTEGER"),
            ("use_max_tokens", "INTEGER NOT NULL DEFAULT 0"),
            ("top_p", "REAL"),
            ("frequency_penalty", "REAL"),
            ("presence_penalty", "REAL"),
            ("stop_sequences", "TEXT"),
            ("endpoint", "TEXT"),
            ("pinned", "INTEGER NOT NULL DEFAULT 0"),
            ("draft", "TEXT"),
            ("auto_titled", "INTEGER NOT NULL DEFAULT 0"),
        ];
        const MSG_COLS: &[(&str, &str)] = &[
            // SQLite allows ALTER TABLE ADD COLUMN with a self-referencing
            // FK as long as no NOT NULL is involved. parent_id is nullable
            // (root rows have NULL) so this is safe.
            ("parent_id", "INTEGER REFERENCES messages(id) ON DELETE CASCADE"),
            ("branch_index", "INTEGER NOT NULL DEFAULT 0"),
            ("created_at", "INTEGER"),
        ];
        const PRESET_COLS: &[(&str, &str)] = &[
            ("model", "TEXT"),
            ("system_prompt", "TEXT"),
            ("temperature", "REAL"),
            ("max_tokens", "INTEGER"),
            ("use_max_tokens", "INTEGER NOT NULL DEFAULT 0"),
            ("top_p", "REAL"),
            ("frequency_penalty", "REAL"),
            ("presence_penalty", "REAL"),
            ("stop_sequences", "TEXT"),
            ("endpoint", "TEXT"),
        ];

        for (col, ty) in CONV_COLS {
            self.add_column_if_missing("conversations", col, ty);
        }
        for (col, ty) in MSG_COLS {
            self.add_column_if_missing("messages", col, ty);
        }
        for (col, ty) in PRESET_COLS {
            self.add_column_if_missing("presets", col, ty);
        }
    }

    fn add_column_if_missing(&self, table: &str, column: &str, type_def: &str) {
        let exists: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info(?1) WHERE name = ?2",
                params![table, column],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if exists == 0 {
            // Table/column names come from compile-time string literals
            // above, not user input — string-formatting them into the SQL
            // is safe here.
            let sql = format!("ALTER TABLE {table} ADD COLUMN {column} {type_def}");
            if let Err(e) = self.conn.execute(&sql, []) {
                eprintln!("Migration warning: failed to add {table}.{column}: {e}");
            }
        }
    }

    /// Current schema version, or 0 if the table is empty/missing.
    #[allow(dead_code)] // used by tests; future migrations will branch on this
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

    /// Incremental save: messages with `id = None` are INSERTed (and have
    /// their assigned rowid written back into the struct via the caller's
    /// `&mut`); messages with `id = Some` are UPDATEd in place. This
    /// preserves branch history — the previous delete-and-reinsert pattern
    /// would have wiped sibling branches every save.
    ///
    /// **Auto-linkage**: a new message (id=None) with no explicit parent_id
    /// links to the immediately preceding message in the slice. This makes
    /// normal turn appends ("user push then assistant push") just work
    /// without callers threading ids around. Sibling creation (regenerate,
    /// edit) sets `parent_id` explicitly *before* calling save, so the
    /// auto-link doesn't apply there.
    ///
    /// **Caller contract**: when relying on auto-linkage, pass a slice that
    /// includes the conversation's *current active tail* — not just the
    /// new messages. Both `save_current` (in app.rs) and the test helpers
    /// pass `&mut self.messages`, which is the active path; this is the
    /// supported shape. Passing only the new tail will fall back to a
    /// "highest-position row in the conv" tail query that may parent off
    /// an unrelated sibling branch if the user navigated away from the
    /// most-recently-inserted branch.
    pub fn save_messages(
        &self,
        conversation_id: i64,
        messages: &mut [Message],
    ) -> Result<(), String> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| format!("Failed to start transaction: {e}"))?;

        // Find the highest existing position so new inserts append rather
        // than collide with existing rows. Position is a tiebreaker for the
        // legacy load path and a stable insertion order.
        let mut next_position: i64 = tx
            .query_row(
                "SELECT COALESCE(MAX(position), -1) FROM messages WHERE conversation_id = ?1",
                params![conversation_id],
                |row| row.get(0),
            )
            .unwrap_or(-1)
            + 1;

        let mut prev_id: Option<i64> = None;
        for msg in messages.iter_mut() {
            let role_str = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            let content_json = serde_json::to_string(&msg.content)
                .map_err(|e| format!("Failed to encode content: {e}"))?;
            match msg.id {
                Some(id) => {
                    // Streaming append on the assistant message updates
                    // content frequently; parent_id and branch_index never
                    // change after insert, so don't touch them.
                    tx.execute(
                        "UPDATE messages SET content = ?1, created_at = ?2 WHERE id = ?3",
                        params![content_json, msg.created_at, id],
                    )
                    .map_err(|e| format!("Failed to update message: {e}"))?;
                    prev_id = Some(id);
                }
                None => {
                    // Auto-link: if the caller didn't set parent_id, fall
                    // back to the previous message in the slice (or the
                    // conversation's existing tail when this is the first
                    // entry to be inserted in a turn).
                    if msg.parent_id.is_none() {
                        msg.parent_id = prev_id.or_else(|| {
                            tx.query_row(
                                "SELECT id FROM messages
                                 WHERE conversation_id = ?1
                                 ORDER BY position DESC LIMIT 1",
                                params![conversation_id],
                                |row| row.get::<_, i64>(0),
                            )
                            .ok()
                        });
                    }
                    tx.execute(
                        "INSERT INTO messages
                         (conversation_id, role, content, position, parent_id, branch_index, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            conversation_id,
                            role_str,
                            content_json,
                            next_position,
                            msg.parent_id,
                            msg.branch_index,
                            msg.created_at,
                        ],
                    )
                    .map_err(|e| format!("Failed to insert message: {e}"))?;
                    let new_id = tx.last_insert_rowid();
                    msg.id = Some(new_id);
                    prev_id = Some(new_id);
                    next_position += 1;
                }
            }
        }

        tx.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?1",
            params![conversation_id],
        )
        .map_err(|e| format!("Failed to update timestamp: {e}"))?;

        tx.commit().map_err(|e| format!("Failed to commit: {e}"))
    }

    /// Compute the next branch_index for a new sibling under `parent_id` in
    /// the given conversation. Returns 0 if the parent has no children yet.
    pub fn next_branch_index(&self, conversation_id: i64, parent_id: Option<i64>) -> i64 {
        // SQL NULL doesn't compare equal with `=`, so use IS for the root
        // branch case (parent_id IS NULL).
        let max: i64 = match parent_id {
            Some(pid) => self
                .conn
                .query_row(
                    "SELECT COALESCE(MAX(branch_index), -1) FROM messages
                     WHERE conversation_id = ?1 AND parent_id = ?2",
                    params![conversation_id, pid],
                    |row| row.get(0),
                )
                .unwrap_or(-1),
            None => self
                .conn
                .query_row(
                    "SELECT COALESCE(MAX(branch_index), -1) FROM messages
                     WHERE conversation_id = ?1 AND parent_id IS NULL",
                    params![conversation_id],
                    |row| row.get(0),
                )
                .unwrap_or(-1),
        };
        max + 1
    }

    /// Load the active path for a conversation. Two modes:
    ///
    /// - **Legacy (0.7.0)**: every row has `parent_id IS NULL`. Return all
    ///   rows in `position` order, exactly as before.
    /// - **Branched**: walk from the newest root (parent_id IS NULL with
    ///   the most recent `created_at`) and at each step pick the child with
    ///   the most recent `created_at`. Stops at a leaf.
    pub fn load_messages(&self, conversation_id: i64) -> Vec<Message> {
        // Pull every row once, then walk in memory. The largest hChat
        // conversation is going to have hundreds of messages, not millions.
        let rows = match self.fetch_all_message_rows(conversation_id) {
            Some(r) => r,
            None => return Vec::new(),
        };
        if rows.is_empty() {
            return Vec::new();
        }

        // Branching is signalled by the presence of any non-NULL parent_id.
        // A clean legacy conv (or a newly-created conv with one root and no
        // children yet) takes the position-ordered fast path.
        let any_parent_set = rows.iter().any(|r| r.parent_id.is_some());
        if !any_parent_set {
            let mut sorted = rows;
            sorted.sort_by_key(|r| r.position);
            return sorted.into_iter().map(MessageRow::into_message).collect();
        }

        // Branched walk: newest root, then newest child at each step.
        use std::collections::HashMap;
        let mut by_parent: HashMap<Option<i64>, Vec<MessageRow>> = HashMap::new();
        for r in rows {
            by_parent.entry(r.parent_id).or_default().push(r);
        }
        // Sort each bucket by created_at DESC so `.first()` gives the newest.
        for bucket in by_parent.values_mut() {
            // Newest sibling wins; branch_index DESC is the deterministic
            // tiebreaker when two siblings share a created_at (or when both
            // are NULL on legacy rows).
            bucket.sort_by_key(|r| (std::cmp::Reverse(r.created_at), std::cmp::Reverse(r.branch_index)));
        }

        let mut out: Vec<Message> = Vec::new();
        let mut current_parent: Option<i64> = None;
        while let Some(bucket) = by_parent.get(&current_parent) {
            let Some(picked) = bucket.first() else {
                break;
            };
            let id = picked.id;
            // Clone here because the bucket may be referenced again at a
            // different parent depth (rare, but cheap relative to a chat
            // round-trip).
            out.push(picked.clone().into_message());
            current_parent = Some(id);
        }
        out
    }

    fn fetch_all_message_rows(&self, conversation_id: i64) -> Option<Vec<MessageRow>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, role, content, created_at, parent_id, branch_index, position
                 FROM messages WHERE conversation_id = ?1",
            )
            .ok()?;
        let mapped = stmt
            .query_map(params![conversation_id], MessageRow::from_row)
            .ok()?;
        Some(mapped.filter_map(|r| r.ok()).collect())
    }

    /// One-time backfill: set `parent_id` of each row to the id of the
    /// previous row (by position) within the same conversation. No-op if
    /// any row already has a non-NULL parent_id (idempotent).
    ///
    /// Called by edit/regenerate before they create the first sibling in a
    /// legacy 0.7.0 conversation, so the new branch slots into a coherent
    /// parent chain.
    pub fn backfill_parent_ids(&self, conversation_id: i64) -> Result<(), String> {
        let already: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM messages
                 WHERE conversation_id = ?1 AND parent_id IS NOT NULL",
                params![conversation_id],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if already > 0 {
            return Ok(());
        }
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| format!("Failed to start backfill tx: {e}"))?;
        let pairs: Vec<(i64, i64)> = {
            let mut stmt = tx
                .prepare(
                    "SELECT id, position FROM messages
                     WHERE conversation_id = ?1 ORDER BY position",
                )
                .map_err(|e| format!("Failed to prepare backfill query: {e}"))?;
            stmt.query_map(params![conversation_id], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|e| format!("Failed to query backfill rows: {e}"))?
            .filter_map(|r| r.ok())
            .collect()
        };
        // Walk in position order: each row's parent is the previous row's id.
        let mut prev_id: Option<i64> = None;
        for (id, _pos) in pairs {
            if let Some(p) = prev_id {
                tx.execute(
                    "UPDATE messages SET parent_id = ?1 WHERE id = ?2",
                    params![p, id],
                )
                .map_err(|e| format!("Failed backfill update: {e}"))?;
            }
            prev_id = Some(id);
        }
        tx.commit()
            .map_err(|e| format!("Failed to commit backfill: {e}"))
    }

    /// All messages with the same parent_id as `message_id` (including the
    /// message itself), sorted by branch_index ascending. Used to render the
    /// `◀ N/M ▶` sibling navigator.
    pub fn siblings_of(&self, message_id: i64) -> Vec<MessageHeader> {
        // Look up parent_id and conversation_id of the target.
        let row: Option<(Option<i64>, i64)> = self
            .conn
            .query_row(
                "SELECT parent_id, conversation_id FROM messages WHERE id = ?1",
                params![message_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();
        let Some((parent_id, conv_id)) = row else {
            return Vec::new();
        };
        let mut stmt = match parent_id {
            Some(_) => self.conn.prepare(
                "SELECT id, branch_index, created_at FROM messages
                 WHERE conversation_id = ?1 AND parent_id = ?2
                 ORDER BY branch_index",
            ),
            None => self.conn.prepare(
                "SELECT id, branch_index, created_at FROM messages
                 WHERE conversation_id = ?1 AND parent_id IS NULL
                 ORDER BY branch_index",
            ),
        };
        let Ok(stmt) = &mut stmt else {
            return Vec::new();
        };
        // Collect inside each arm so the two closures don't need to unify
        // — `query_map` is generic over the closure type and the arms would
        // otherwise produce incompatible MappedRows types.
        let map_row = |row: &rusqlite::Row<'_>| {
            Ok::<MessageHeader, rusqlite::Error>(MessageHeader {
                id: row.get(0)?,
                branch_index: row.get(1)?,
                created_at: row.get(2).ok(),
            })
        };
        match parent_id {
            Some(pid) => stmt
                .query_map(params![conv_id, pid], map_row)
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default(),
            None => stmt
                .query_map(params![conv_id], map_row)
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default(),
        }
    }

    /// Walk the active path starting from a specific message — used when
    /// the user clicks a sibling-navigation arrow and we need to rebuild
    /// the suffix of `self.messages` from that branch point down. The
    /// returned vec includes `start_id` as the first element, then walks
    /// children picking newest at each fork.
    pub fn walk_from(&self, start_id: i64) -> Vec<Message> {
        // Lift the conversation id once, then reuse the per-conv fetch.
        let conv_id: Option<i64> = self
            .conn
            .query_row(
                "SELECT conversation_id FROM messages WHERE id = ?1",
                params![start_id],
                |row| row.get(0),
            )
            .ok();
        let Some(conv_id) = conv_id else {
            return Vec::new();
        };
        let rows = match self.fetch_all_message_rows(conv_id) {
            Some(r) => r,
            None => return Vec::new(),
        };
        let start = match rows.iter().find(|r| r.id == start_id) {
            Some(r) => r.clone(),
            None => return Vec::new(),
        };

        use std::collections::HashMap;
        let mut by_parent: HashMap<Option<i64>, Vec<MessageRow>> = HashMap::new();
        for r in rows {
            by_parent.entry(r.parent_id).or_default().push(r);
        }
        for bucket in by_parent.values_mut() {
            // Newest sibling wins; branch_index DESC is the deterministic
            // tiebreaker when two siblings share a created_at (or when both
            // are NULL on legacy rows).
            bucket.sort_by_key(|r| (std::cmp::Reverse(r.created_at), std::cmp::Reverse(r.branch_index)));
        }

        let mut out = vec![start.clone().into_message()];
        let mut current_parent: Option<i64> = Some(start.id);
        while let Some(bucket) = by_parent.get(&current_parent) {
            let Some(picked) = bucket.first() else {
                break;
            };
            let id = picked.id;
            out.push(picked.clone().into_message());
            current_parent = Some(id);
        }
        out
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

/// Lightweight sibling-row description for the navigation UI. Doesn't carry
/// content — just enough to pick the right id and show "N/M".
pub struct MessageHeader {
    pub id: i64,
    /// Persisted branch_index. Not currently read by the UI (the N/M
    /// display uses array position) but exposed for tests / future code.
    #[allow(dead_code)]
    pub branch_index: i64,
    #[allow(dead_code)]
    pub created_at: Option<i64>,
}

/// Raw row from the messages table — we rebuild `Message` from this with
/// the JSON content parse separated out so the load/walk paths don't
/// duplicate parsing logic.
#[derive(Clone)]
struct MessageRow {
    id: i64,
    role: Role,
    content: Vec<ContentPart>,
    created_at: Option<i64>,
    parent_id: Option<i64>,
    branch_index: i64,
    position: i64,
}

impl MessageRow {
    fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        let role_str: String = row.get(1)?;
        let role = match role_str.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => Role::Assistant,
        };
        let content_raw: String = row.get(2)?;
        let content: Vec<ContentPart> = serde_json::from_str(&content_raw)
            .unwrap_or_else(|_| vec![ContentPart::Text { text: content_raw }]);
        Ok(MessageRow {
            id: row.get(0)?,
            role,
            content,
            created_at: row.get(3).ok(),
            parent_id: row.get(4).ok(),
            // Schema-guaranteed NOT NULL columns — propagate errors rather
            // than silently default to 0 if the row shape ever drifts.
            branch_index: row.get(5)?,
            position: row.get(6)?,
        })
    }

    fn into_message(self) -> Message {
        Message {
            role: self.role,
            content: self.content,
            created_at: self.created_at,
            id: Some(self.id),
            parent_id: self.parent_id,
            branch_index: self.branch_index,
        }
    }
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

    /// Build a Storage with the *original 0.6.0 schema* — no
    /// schema_version table, no v2 columns. Used to prove that
    /// init_tables migrates cleanly when launched against an old DB.
    fn legacy_0_6_storage_with_data() -> Storage {
        let conn = Connection::open_in_memory().expect("open :memory:");
        conn.execute_batch(
            "CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                position INTEGER NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );",
        )
        .expect("create legacy schema");
        // Insert one row in the legacy plain-text content shape.
        conn.execute(
            "INSERT INTO conversations (title) VALUES ('legacy chat')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, position)
             VALUES (1, 'user', 'hello from 0.6', 0)",
            [],
        )
        .unwrap();
        Storage { conn }
    }

    #[test]
    fn schema_is_at_v2_after_init() {
        let s = mem_storage();
        assert_eq!(s.schema_version(), 2);
    }

    #[test]
    fn init_tables_migrates_legacy_0_6_db_in_place() {
        // This is the regression test for the "no such column: parent_id"
        // crash that hit any 0.6.x user upgrading without wiping their DB.
        let s = legacy_0_6_storage_with_data();
        // init_tables must NOT panic even though the existing tables lack
        // every v2 column (parent_id, pinned, draft, auto_titled, etc).
        s.init_tables();

        // Schema version is now 2.
        assert_eq!(s.schema_version(), 2);

        // The pre-existing conversation + message survived the migration.
        let convs = s.list_conversations();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].title, "legacy chat");
        assert!(!convs[0].pinned); // newly-added column defaults to 0
        let msgs = s.load_messages(convs[0].id);
        assert_eq!(msgs.len(), 1);
        // Plain-text content (not JSON) round-trips via the load fallback.
        assert_eq!(msgs[0].text_str(), "hello from 0.6");
        // New v2 columns default sensibly on legacy rows.
        assert_eq!(msgs[0].parent_id, None);
        assert_eq!(msgs[0].branch_index, 0);

        // Subsequent saves write into the migrated schema without error.
        let mut new_msgs = vec![Message::text(Role::Assistant, "hi from 0.8".into())];
        s.save_messages(convs[0].id, &mut new_msgs)
            .expect("save into migrated DB");
        assert!(new_msgs[0].id.is_some());
    }

    #[test]
    fn init_tables_is_idempotent() {
        // Running init_tables twice on an already-migrated DB must be a
        // no-op (no duplicate-column errors, schema_version stays at 2).
        let s = legacy_0_6_storage_with_data();
        s.init_tables();
        s.init_tables();
        assert_eq!(s.schema_version(), 2);
    }

    #[test]
    fn round_trips_text_messages() {
        let s = mem_storage();
        let id = s.create_conversation("hello").unwrap();
        let mut msgs = vec![
            Message::text(Role::User, "hi".to_string()),
            Message::text(Role::Assistant, "hello!".to_string()),
        ];
        s.save_messages(id, &mut msgs).unwrap();
        let loaded = s.load_messages(id);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text_str(), "hi");
        assert_eq!(loaded[1].text_str(), "hello!");
        // Auto-link: the second message's parent is the first.
        assert_eq!(loaded[1].parent_id, loaded[0].id);
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
        let mut msgs = vec![Message::from_parts(Role::User, parts.clone())];
        s.save_messages(id, &mut msgs).unwrap();
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
            &mut [Message::from_parts(
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

    // ----- Phase 4: branching -----

    /// Helper: bump created_at by a known delta so newest-wins picks are
    /// deterministic. SQLite times via `datetime('now')` only resolve to
    /// the second; we set our own ms timestamps to side-step that.
    fn ts(base: i64, offset: i64) -> Option<i64> {
        Some(base + offset)
    }

    #[test]
    fn save_messages_assigns_ids_and_writes_them_back() {
        let s = mem_storage();
        let conv = s.create_conversation("ids").unwrap();
        let mut msgs = vec![
            Message::text(Role::User, "u".into()),
            Message::text(Role::Assistant, "a".into()),
        ];
        s.save_messages(conv, &mut msgs).unwrap();
        assert!(msgs[0].id.is_some());
        assert!(msgs[1].id.is_some());
        assert_eq!(msgs[1].parent_id, msgs[0].id);
    }

    #[test]
    fn save_messages_updates_existing_in_place() {
        let s = mem_storage();
        let conv = s.create_conversation("update").unwrap();
        let mut msgs = vec![Message::text(Role::User, "before".into())];
        s.save_messages(conv, &mut msgs).unwrap();
        let id = msgs[0].id.unwrap();
        // Mutate content and save again — id stays, content changes.
        msgs[0].content = vec![ContentPart::Text {
            text: "after".into(),
        }];
        s.save_messages(conv, &mut msgs).unwrap();
        assert_eq!(msgs[0].id, Some(id));
        let loaded = s.load_messages(conv);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].text_str(), "after");
    }

    #[test]
    fn legacy_load_falls_back_to_position_order() {
        // A 0.7.0 conversation: every row has parent_id=NULL. load_messages
        // must return them in position (== insertion) order.
        let s = mem_storage();
        let conv = s.create_conversation("legacy").unwrap();
        let mut msgs = vec![
            Message::text(Role::User, "u1".into()),
            Message::text(Role::Assistant, "a1".into()),
            Message::text(Role::User, "u2".into()),
            Message::text(Role::Assistant, "a2".into()),
        ];
        s.save_messages(conv, &mut msgs).unwrap();
        // Manually NULL out parent_id to simulate a legacy row layout
        // (auto-link populated it during save above).
        s.conn
            .execute(
                "UPDATE messages SET parent_id = NULL WHERE conversation_id = ?1",
                params![conv],
            )
            .unwrap();
        let loaded = s.load_messages(conv);
        assert_eq!(
            loaded.iter().map(|m| m.text_str()).collect::<Vec<_>>(),
            vec!["u1", "a1", "u2", "a2"]
        );
    }

    #[test]
    fn backfill_parent_ids_chains_legacy_rows() {
        let s = mem_storage();
        let conv = s.create_conversation("backfill").unwrap();
        let mut msgs = vec![
            Message::text(Role::User, "u1".into()),
            Message::text(Role::Assistant, "a1".into()),
            Message::text(Role::User, "u2".into()),
        ];
        s.save_messages(conv, &mut msgs).unwrap();
        s.conn
            .execute(
                "UPDATE messages SET parent_id = NULL WHERE conversation_id = ?1",
                params![conv],
            )
            .unwrap();

        s.backfill_parent_ids(conv).unwrap();
        let loaded = s.load_messages(conv);
        assert_eq!(loaded[0].parent_id, None);
        assert_eq!(loaded[1].parent_id, loaded[0].id);
        assert_eq!(loaded[2].parent_id, loaded[1].id);
    }

    #[test]
    fn backfill_parent_ids_is_idempotent() {
        let s = mem_storage();
        let conv = s.create_conversation("idempotent").unwrap();
        let mut msgs = vec![
            Message::text(Role::User, "u".into()),
            Message::text(Role::Assistant, "a".into()),
        ];
        s.save_messages(conv, &mut msgs).unwrap();
        let before = s.load_messages(conv);
        // Run backfill twice — should be a no-op the second time and not
        // disturb existing parent_ids.
        s.backfill_parent_ids(conv).unwrap();
        s.backfill_parent_ids(conv).unwrap();
        let after = s.load_messages(conv);
        assert_eq!(
            after.iter().map(|m| m.parent_id).collect::<Vec<_>>(),
            before.iter().map(|m| m.parent_id).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn branched_load_picks_newest_sibling_at_each_fork() {
        // Build a tree:
        //   user (id=1)
        //     ├── assistant_old (branch 0, ts t)
        //     └── assistant_new (branch 1, ts t+10)  ← active
        //           └── (no children)
        let s = mem_storage();
        let conv = s.create_conversation("branched").unwrap();
        let base = 1_700_000_000_000i64;
        let mut user = Message::text(Role::User, "what".into());
        user.created_at = ts(base, 0);
        let mut a_old = Message::text(Role::Assistant, "old reply".into());
        a_old.created_at = ts(base, 1);
        let mut msgs = vec![user, a_old];
        s.save_messages(conv, &mut msgs).unwrap();
        let user_id = msgs[0].id.unwrap();

        // Inject a sibling assistant directly (skipping the regenerate UI
        // path we test elsewhere).
        let mut a_new = Message::text(Role::Assistant, "new reply".into());
        a_new.parent_id = Some(user_id);
        a_new.branch_index = 1;
        a_new.created_at = ts(base, 10);
        s.save_messages(conv, &mut [a_new]).unwrap();

        let loaded = s.load_messages(conv);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text_str(), "what");
        // Newest sibling wins at the branch point.
        assert_eq!(loaded[1].text_str(), "new reply");
        assert_eq!(loaded[1].branch_index, 1);
    }

    #[test]
    fn siblings_of_returns_all_branches_at_a_point() {
        let s = mem_storage();
        let conv = s.create_conversation("sibs").unwrap();
        let mut msgs = vec![
            Message::text(Role::User, "ask".into()),
            Message::text(Role::Assistant, "one".into()),
        ];
        s.save_messages(conv, &mut msgs).unwrap();
        let user_id = msgs[0].id.unwrap();
        let first_a_id = msgs[1].id.unwrap();

        let mut sib = Message::text(Role::Assistant, "two".into());
        sib.parent_id = Some(user_id);
        sib.branch_index = 1;
        s.save_messages(conv, &mut [sib]).unwrap();

        let sibs = s.siblings_of(first_a_id);
        assert_eq!(sibs.len(), 2);
        assert_eq!(sibs[0].branch_index, 0);
        assert_eq!(sibs[1].branch_index, 1);
    }

    #[test]
    fn next_branch_index_grows_per_parent() {
        let s = mem_storage();
        let conv = s.create_conversation("nbi").unwrap();
        let mut msgs = vec![Message::text(Role::User, "u".into())];
        s.save_messages(conv, &mut msgs).unwrap();
        let user_id = msgs[0].id.unwrap();
        // First assistant under this parent: branch 0.
        assert_eq!(s.next_branch_index(conv, Some(user_id)), 0);
        let mut a = Message::text(Role::Assistant, "a".into());
        a.parent_id = Some(user_id);
        a.branch_index = 0;
        s.save_messages(conv, &mut [a]).unwrap();
        // Now next is 1.
        assert_eq!(s.next_branch_index(conv, Some(user_id)), 1);
    }

    #[test]
    fn walk_from_returns_subtree_active_path() {
        // user -> assistant_a -> user2_a -> assistant_a2
        //                    \-> user2_b -> assistant_b2 (newer, wins on default load)
        let s = mem_storage();
        let conv = s.create_conversation("walk").unwrap();
        let base = 1_700_000_000_000i64;
        let mut u = Message::text(Role::User, "u".into());
        u.created_at = ts(base, 0);
        let mut a = Message::text(Role::Assistant, "a".into());
        a.created_at = ts(base, 1);
        let mut msgs = vec![u, a];
        s.save_messages(conv, &mut msgs).unwrap();
        let a_id = msgs[1].id.unwrap();

        let mut u2a = Message::text(Role::User, "u2-old".into());
        u2a.parent_id = Some(a_id);
        u2a.branch_index = 0;
        u2a.created_at = ts(base, 2);
        s.save_messages(conv, &mut [u2a.clone()]).unwrap();

        let mut u2b = Message::text(Role::User, "u2-new".into());
        u2b.parent_id = Some(a_id);
        u2b.branch_index = 1;
        u2b.created_at = ts(base, 10);
        s.save_messages(conv, &mut [u2b]).unwrap();

        // Walking from a_id picks the newer u2-new branch.
        let path = s.walk_from(a_id);
        let texts: Vec<String> = path.iter().map(|m| m.text_str()).collect();
        assert_eq!(texts, vec!["a", "u2-new"]);
    }
}
