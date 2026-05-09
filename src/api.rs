use crate::message::Message;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Returns true iff `url` parses and its host is exactly `expected_host`. Used
/// to gate provider-specific behavior (URL rewrite, vendor headers) so a user
/// can't accidentally route their bearer token to OpenRouter by typing
/// `https://my-proxy.example/openrouter.ai/v1` — substring matching of the
/// host name was a credential-redirection footgun.
fn host_is(url: &str, expected_host: &str) -> bool {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_ascii_lowercase()))
        .as_deref()
        == Some(expected_host)
}

#[derive(Debug, Deserialize)]
pub struct OllamaModel {
    pub name: String,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModel {
    id: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModel>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "stop")]
    stop_sequences: Option<Vec<String>>,
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
    /// OpenRouter returns `cost` inside the `usage` object (credits charged)
    pub cost: Option<f64>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: Option<Delta>,
}

#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
    /// Some providers (DeepSeek, OpenRouter forwarding reasoning models) emit a
    /// separate `reasoning` field for chain-of-thought. We surface it as a
    /// `Reasoning` event so the UI can show it collapsed.
    reasoning: Option<String>,
}

pub enum StreamEvent {
    Token(String),
    /// Reasoning/thinking text from a separate provider field. Distinct from
    /// `Token` so the UI can render it differently. For inline `<think>` tags
    /// inside content, the UI parses the assembled text itself.
    Reasoning(String),
    Done,
    Error(String),
    UsageInfo(Usage),
    /// `url` is the endpoint the fetch was started against. The receiver
    /// validates it still matches the currently-selected endpoint before
    /// applying — without that check, switching endpoints while a fetch is
    /// in flight could populate the new endpoint with the old endpoint's
    /// model list.
    ModelsLoaded {
        url: String,
        models: Vec<String>,
    },
}

const MAX_ERROR_BODY: usize = 4096;
const MAX_STREAM_BUFFER: usize = 1024 * 1024; // 1MB

pub async fn fetch_models(base_url: &str, api_key: Option<&str>) -> Result<Vec<String>, String> {
    let client = Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

    let mut url = base_url.trim_end_matches('/').to_string();

    // Auto-fix OpenRouter URLs (users often provide just the domain or the
    // completions endpoint). Only rewrite when the URL's host is literally
    // openrouter.ai — substring matching would let `evil.com/openrouter.ai`
    // hijack the rewrite and leak the bearer token.
    let is_openrouter = host_is(&url, "openrouter.ai");
    if is_openrouter {
        url = "https://openrouter.ai/api/v1".to_string();
    } else if url.ends_with("/chat/completions") {
        url = url.trim_end_matches("/chat/completions").to_string();
    }

    // Try OpenAI-standard /v1/models first (works for OpenRouter, LM Studio, vLLM, etc.)
    let models_url = if url.ends_with("/v1") {
        format!("{url}/models")
    } else {
        format!("{url}/v1/models")
    };

    let mut request = client.get(&models_url);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    if is_openrouter {
        request = request
            .header("HTTP-Referer", "https://github.com/hhheath/hChat")
            .header("X-Title", "hChat");
    }

    if let Ok(resp) = request.send().await {
        if resp.status().is_success() {
            if let Ok(openai_resp) = resp.json::<OpenAIModelsResponse>().await {
                let models: Vec<String> = openai_resp.data.into_iter().map(|m| m.id).collect();
                if !models.is_empty() {
                    return Ok(models);
                }
            }
        }
    }

    // Fall back to Ollama /api/tags
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

pub struct ChatParams {
    pub base_url: String,
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub api_key: Option<String>,
}

pub fn stream_chat(
    params: ChatParams,
    tx: mpsc::UnboundedSender<StreamEvent>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let ChatParams {
            base_url,
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            stop_sequences,
            api_key,
        } = params;
        let client = match Client::builder().connect_timeout(CONNECT_TIMEOUT).build() {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(format!(
                    "Failed to create HTTP client: {e}"
                )));
                return;
            }
        };
        let mut base_url = base_url.trim_end_matches('/').to_string();

        let is_openrouter = host_is(&base_url, "openrouter.ai");
        if is_openrouter {
            base_url = "https://openrouter.ai/api/v1".to_string();
        } else if base_url.ends_with("/chat/completions") {
            base_url = base_url.trim_end_matches("/chat/completions").to_string();
        }

        let url = format!("{}/chat/completions", base_url);

        let req = ChatRequest {
            model,
            messages,
            stream: true,
            temperature,
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            stop_sequences,
        };

        let mut request = client.post(&url).json(&req);
        if let Some(key) = &api_key {
            request = request.bearer_auth(key);
        }
        if is_openrouter {
            request = request
                .header("HTTP-Referer", "https://github.com/hhheath/hChat")
                .header("X-Title", "hChat");
        }

        let resp = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(format!("Request failed: {e}")));
                return;
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .bytes()
                .await
                .map(|b| {
                    let truncated = &b[..b.len().min(MAX_ERROR_BODY)];
                    String::from_utf8_lossy(truncated).into_owned()
                })
                .unwrap_or_default();
            // Log full error for debugging but only show sanitized message in UI
            eprintln!("API error {status}: {body}");
            let message = match status.as_u16() {
                401 => format!("{status}: Authentication failed — check your API key"),
                403 => format!("{status}: Access denied"),
                404 => format!("{status}: Endpoint not found — check your URL"),
                429 => format!("{status}: Rate limited — try again later"),
                500..=599 => format!("{status}: Server error"),
                _ => format!("{status}: Request failed"),
            };
            let _ = tx.send(StreamEvent::Error(message));
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

                    if raw_buf.len() + bytes.len() > MAX_STREAM_BUFFER {
                        let _ = tx.send(StreamEvent::Error("Stream buffer overflow".to_string()));
                        return;
                    }

                    raw_buf.extend_from_slice(&bytes);

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

                    for line_end in newline_positions(text) {
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
                                        // Skip empty strings: some providers
                                        // (Ollama qwen3 thinking, OpenAI keep-
                                        // alives) emit `content: ""` alongside
                                        // a reasoning delta. Treating that as
                                        // "content arrived" prematurely closes
                                        // the <think> block and the next
                                        // reasoning delta re-opens a new one,
                                        // producing dozens of tiny blocks.
                                        if let Some(reasoning) = delta.reasoning
                                            && !reasoning.is_empty()
                                        {
                                            let _ = tx.send(StreamEvent::Reasoning(reasoning));
                                        }
                                        if let Some(content) = delta.content
                                            && !content.is_empty()
                                        {
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
fn newline_positions(s: &str) -> impl Iterator<Item = usize> + '_ {
    s.bytes()
        .enumerate()
        .filter_map(|(i, b)| if b == b'\n' { Some(i) } else { None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_openrouter_fetch() {
        let models = fetch_models("https://openrouter.ai/api/v1", None).await;
        println!("Result: {:?}", models);
    }

    #[test]
    fn host_is_exact_match() {
        assert!(host_is("https://openrouter.ai/api/v1", "openrouter.ai"));
        assert!(host_is("https://openrouter.ai", "openrouter.ai"));
        assert!(host_is("http://openrouter.ai/foo", "openrouter.ai"));
        // Case-insensitive host
        assert!(host_is("https://OpenRouter.AI/api", "openrouter.ai"));
    }

    #[test]
    fn host_is_rejects_substring_attacks() {
        // Path-component "openrouter.ai" — the credential redirect bug.
        assert!(!host_is(
            "https://my-proxy.example/openrouter.ai/v1",
            "openrouter.ai"
        ));
        // Subdomain prefix
        assert!(!host_is(
            "https://openrouter.ai.evil.example/api",
            "openrouter.ai"
        ));
        // Different host
        assert!(!host_is("https://openai.com/v1", "openrouter.ai"));
    }

    #[test]
    fn host_is_rejects_unparseable() {
        assert!(!host_is("openrouter.ai/api/v1", "openrouter.ai")); // no scheme
        assert!(!host_is("not a url", "openrouter.ai"));
        assert!(!host_is("", "openrouter.ai"));
    }
}
