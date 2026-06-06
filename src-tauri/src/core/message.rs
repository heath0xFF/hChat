use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// A tool result, sent back to the model after a tool call. Carries
    /// `tool_call_id` referencing the assistant's call.
    Tool,
}

/// One tool invocation as the model emitted it. OpenAI ships these on
/// assistant messages as a separate `tool_calls` field (not as content
/// parts). Multiple tools can be called in one assistant turn.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    /// Always `"function"` in the current OpenAI shape — kept as a field
    /// for forward-compatibility if more types appear.
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments string. OpenAI sends it as a string (not a
    /// nested object) — we keep it that way so we can pass it through the
    /// tools' parameter validation later.
    pub arguments: String,
}

/// A single piece of message content. Mirrors OpenAI's chat completions
/// `content` array shape so we can serialize directly to the wire format.
/// Pure-text messages are `[Text { text }]`; multimodal messages mix
/// `Text` and `ImageUrl` parts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
    /// Tool calls the assistant emitted in this turn. Sent on the wire as a
    /// peer of `content`, not nested inside it. `None` for non-assistant
    /// messages and for assistant turns that didn't call any tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For Role::Tool messages: the id of the tool_call this is answering.
    /// Required by the API on tool messages, ignored on others.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Unix epoch milliseconds. Session-local — never sent to the API and not
    /// persisted as its own column for messages loaded before phase 2 (those
    /// have `None`). Set when a message is added in the current session.
    #[serde(skip_serializing, default)]
    pub created_at: Option<i64>,
    /// SQLite rowid. `None` for messages that haven't been written yet — the
    /// next `save_messages` will INSERT and capture the assigned id back into
    /// the struct. Persisted messages always have `Some`.
    #[serde(skip, default)]
    pub id: Option<i64>,
    /// Parent message in the branching tree. `None` for the conversation root
    /// (the very first message) and for any message in a legacy 0.7.0
    /// conversation that hasn't been backfilled yet.
    #[serde(skip, default)]
    pub parent_id: Option<i64>,
    /// Sibling index under `parent_id`. 0 for the original/first reply at a
    /// branch point; 1, 2, ... for subsequent regenerations or edits. The
    /// active path picks the sibling with the most recent `created_at` at
    /// each parent.
    #[serde(skip, default)]
    pub branch_index: i64,
}

impl Message {
    /// Construct a text-only message — the most common shape.
    pub fn text(role: Role, body: String) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text { text: body }],
            tool_calls: None,
            tool_call_id: None,
            created_at: Some(now_ms()),
            id: None,
            parent_id: None,
            branch_index: 0,
        }
    }

    /// Construct from raw parts (used when loading from storage / building
    /// multimodal messages with attachments).
    pub fn from_parts(role: Role, parts: Vec<ContentPart>) -> Self {
        Self {
            role,
            content: parts,
            tool_calls: None,
            tool_call_id: None,
            created_at: Some(now_ms()),
            id: None,
            parent_id: None,
            branch_index: 0,
        }
    }

    /// Construct a tool-result message (`role: tool`) responding to a
    /// specific tool_call by id.
    pub fn tool_result(tool_call_id: String, body: String) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentPart::Text { text: body }],
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
            created_at: Some(now_ms()),
            id: None,
            parent_id: None,
            branch_index: 0,
        }
    }

    /// Concatenate all `Text` parts into a single string. Image parts are
    /// skipped. This is the right call for any UI/logic that previously
    /// treated `content` as a `String` (rendering, search, char counts).
    pub fn text_str(&self) -> String {
        let mut out = String::new();
        for part in &self.content {
            if let ContentPart::Text { text } = part {
                out.push_str(text);
            }
        }
        out
    }

    /// Append text to the trailing `Text` part if present, otherwise push a
    /// new `Text` part. Used by streaming token append and reasoning-tag
    /// insertion in the assistant message.
    pub fn append_text(&mut self, s: &str) {
        if let Some(ContentPart::Text { text }) = self.content.last_mut() {
            text.push_str(s);
        } else {
            self.content.push(ContentPart::Text {
                text: s.to_string(),
            });
        }
    }

    /// True if the message has no parts, or all `Text` parts are empty (no
    /// image parts either). Mirrors the old `content.is_empty()` check.
    pub fn is_empty_content(&self) -> bool {
        self.content.iter().all(|p| match p {
            ContentPart::Text { text } => text.is_empty(),
            ContentPart::ImageUrl { .. } => false,
        })
    }

    /// Iterator over image attachments. Used by send-time validation and
    /// (eventually) thumbnail rendering.
    #[allow(dead_code)]
    pub fn images(&self) -> impl Iterator<Item = &ImageUrl> {
        self.content.iter().filter_map(|p| match p {
            ContentPart::ImageUrl { image_url } => Some(image_url),
            _ => None,
        })
    }
}

pub fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_serializes_as_parts_array() {
        let m = Message::text(Role::User, "hello".to_string());
        let json = serde_json::to_value(&m).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "hello");
    }

    #[test]
    fn image_part_matches_openai_wire_shape() {
        let m = Message::from_parts(
            Role::User,
            vec![
                ContentPart::Text {
                    text: "describe this".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,AAAA".to_string(),
                        detail: Some("low".to_string()),
                    },
                },
            ],
        );
        let json = serde_json::to_value(&m).unwrap();
        assert_eq!(json["content"][1]["type"], "image_url");
        assert_eq!(
            json["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
        assert_eq!(json["content"][1]["image_url"]["detail"], "low");
    }

    #[test]
    fn image_url_omits_detail_when_none() {
        let m = Message::from_parts(
            Role::User,
            vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "x".to_string(),
                    detail: None,
                },
            }],
        );
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("detail"), "detail should be omitted: {json}");
    }

    #[test]
    fn round_trip_preserves_parts() {
        let original = Message::from_parts(
            Role::Assistant,
            vec![
                ContentPart::Text {
                    text: "before".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "u".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "after".to_string(),
                },
            ],
        );
        let json = serde_json::to_string(&original).unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(original.content, parsed.content);
    }

    #[test]
    fn text_str_concatenates_text_parts_only() {
        let m = Message::from_parts(
            Role::User,
            vec![
                ContentPart::Text {
                    text: "foo ".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "x".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "bar".to_string(),
                },
            ],
        );
        assert_eq!(m.text_str(), "foo bar");
    }

    #[test]
    fn append_text_extends_trailing_text_part() {
        let mut m = Message::text(Role::Assistant, "hello".to_string());
        m.append_text(" world");
        assert_eq!(m.content.len(), 1);
        assert_eq!(m.text_str(), "hello world");
    }

    #[test]
    fn append_text_creates_part_if_last_is_image() {
        let mut m = Message::from_parts(
            Role::User,
            vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "x".to_string(),
                    detail: None,
                },
            }],
        );
        m.append_text("caption");
        assert_eq!(m.content.len(), 2);
        assert_eq!(m.text_str(), "caption");
    }

    #[test]
    fn append_text_creates_part_when_empty() {
        let mut m = Message::from_parts(Role::Assistant, Vec::new());
        m.append_text("first");
        assert_eq!(m.content.len(), 1);
        assert_eq!(m.text_str(), "first");
    }

    #[test]
    fn is_empty_content_handles_text_and_images() {
        assert!(Message::from_parts(Role::User, Vec::new()).is_empty_content());
        assert!(Message::text(Role::User, String::new()).is_empty_content());
        assert!(!Message::text(Role::User, "x".to_string()).is_empty_content());
        let with_img = Message::from_parts(
            Role::User,
            vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "x".to_string(),
                    detail: None,
                },
            }],
        );
        assert!(!with_img.is_empty_content());
    }
}
