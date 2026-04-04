use crate::message::Message;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Deserialize)]
pub struct OllamaModel {
    pub name: String,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Option<Vec<ChunkChoice>>,
    usage: Option<Usage>,
}

#[derive(Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: Option<Delta>,
}

#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
}

pub enum StreamEvent {
    Token(String),
    Done,
    Error(String),
    UsageInfo(Usage),
}

const MAX_ERROR_BODY: usize = 4096;
const MAX_STREAM_BUFFER: usize = 1024 * 1024; // 1MB

pub async fn fetch_models(base_url: &str) -> Result<Vec<String>, String> {
    let client = Client::new();
    let url = base_url.trim_end_matches('/');
    let api_base = url.trim_end_matches("/v1");
    let resp = client
        .get(format!("{api_base}/api/tags"))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch models: {e}"))?;

    let tags: OllamaTagsResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse models: {e}"))?;

    Ok(tags.models.into_iter().map(|m| m.name).collect())
}

pub fn stream_chat(
    base_url: String,
    model: String,
    messages: Vec<Message>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tx: mpsc::UnboundedSender<StreamEvent>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let client = Client::new();
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        let req = ChatRequest {
            model,
            messages,
            stream: true,
            temperature,
            max_tokens,
        };

        let resp = match client.post(&url).json(&req).send().await {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(format!("Request failed: {e}")));
                return;
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            // Limit error body size to prevent OOM
            let body = resp
                .bytes()
                .await
                .map(|b| {
                    let truncated = &b[..b.len().min(MAX_ERROR_BODY)];
                    String::from_utf8_lossy(truncated).into_owned()
                })
                .unwrap_or_default();
            let _ = tx.send(StreamEvent::Error(format!("{status}: {body}")));
            return;
        }

        let mut stream = resp.bytes_stream();
        // Use a byte buffer to handle UTF-8 sequences split across chunks
        let mut raw_buf: Vec<u8> = Vec::new();

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    let _ = tx.send(StreamEvent::Done);
                    return;
                }
                chunk = stream.next() => {
                    let Some(chunk) = chunk else {
                        let _ = tx.send(StreamEvent::Done);
                        return;
                    };

                    let bytes = match chunk {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = tx.send(StreamEvent::Error(format!("Stream error: {e}")));
                            return;
                        }
                    };

                    raw_buf.extend_from_slice(&bytes);

                    if raw_buf.len() > MAX_STREAM_BUFFER {
                        let _ = tx.send(StreamEvent::Error("Stream buffer overflow".to_string()));
                        return;
                    }

                    // Find the last valid UTF-8 boundary to avoid splitting multi-byte chars
                    let valid_up_to = match std::str::from_utf8(&raw_buf) {
                        Ok(_) => raw_buf.len(),
                        Err(e) => e.valid_up_to(),
                    };

                    if valid_up_to == 0 {
                        continue;
                    }

                    // Safe: we just validated this range is valid UTF-8
                    let text = std::str::from_utf8(&raw_buf[..valid_up_to]).unwrap();
                    let mut remaining_start = 0;

                    for line_end in memchr_newlines(text) {
                        let line = text[remaining_start..line_end].trim();
                        remaining_start = line_end + 1;

                        if line.is_empty() || line.starts_with(':') {
                            continue;
                        }

                        let data = if let Some(d) = line.strip_prefix("data: ") {
                            d.trim()
                        } else {
                            continue;
                        };

                        if data == "[DONE]" {
                            let _ = tx.send(StreamEvent::Done);
                            return;
                        }

                        if let Ok(chunk) = serde_json::from_str::<ChatChunk>(data) {
                            if let Some(choices) = chunk.choices {
                                for choice in choices {
                                    if let Some(delta) = choice.delta {
                                        if let Some(content) = delta.content {
                                            let _ = tx.send(StreamEvent::Token(content));
                                        }
                                    }
                                }
                            }
                            if let Some(usage) = chunk.usage {
                                let _ = tx.send(StreamEvent::UsageInfo(usage));
                            }
                        }
                    }

                    // Keep unprocessed bytes (incomplete line + any trailing invalid UTF-8)
                    let consumed = remaining_start.min(valid_up_to);
                    raw_buf = raw_buf[consumed..].to_vec();
                }
            }
        }
    });
}

/// Find all newline positions in a string slice
fn memchr_newlines(s: &str) -> Vec<usize> {
    s.bytes()
        .enumerate()
        .filter_map(|(i, b)| if b == b'\n' { Some(i) } else { None })
        .collect()
}
