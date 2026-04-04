use crate::message::Message;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

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
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Option<Vec<ChunkChoice>>,
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
}

pub async fn fetch_models(base_url: &str) -> Result<Vec<String>, String> {
    let client = Client::new();
    // Ollama's native endpoint for listing models
    let url = base_url.trim_end_matches('/');
    // Try /api/tags (Ollama native), strip /v1 suffix if present
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
    tx: mpsc::UnboundedSender<StreamEvent>,
) {
    tokio::spawn(async move {
        let client = Client::new();
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        let req = ChatRequest {
            model,
            messages,
            stream: true,
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
            let body = resp.text().await.unwrap_or_default();
            let _ = tx.send(StreamEvent::Error(format!("{status}: {body}")));
            return;
        }

        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(format!("Stream error: {e}")));
                    return;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // Process SSE lines
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

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
                }
            }
        }

        let _ = tx.send(StreamEvent::Done);
    });
}
