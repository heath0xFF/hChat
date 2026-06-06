use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;

#[cfg(unix)]
fn set_dir_permissions(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
}

const MAX_CONFIG_SIZE: u64 = 1_048_576; // 1 MB

/// Which inference server is behind an endpoint. Drives how the metrics
/// dashboard interprets it (Prometheus dialect, GPU source defaults).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    Vllm,
    Omlx,
    #[serde(rename = "llamacpp")]
    LlamaCpp,
    /// llama-swap proxy in front of llama.cpp (its own `llama_swap_*` metrics).
    #[serde(rename = "llamaswap")]
    LlamaSwap,
    /// Generic OpenAI-compatible (incl. cloud like OpenRouter). No GPU metrics.
    #[default]
    Openai,
}

/// Where to pull GPU stats (VRAM/power/temp/util) for this endpoint's host.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuKind {
    /// No GPU metrics (cloud, or not configured).
    #[default]
    None,
    /// Local Apple Silicon GPU via the macmon sampler (no sudo).
    Macmon,
    /// A remote `hchat-agent` exposing nvidia-smi + /proc/meminfo over HTTP.
    Agent,
}

fn runtime_is_default(r: &Runtime) -> bool {
    *r == Runtime::Openai
}
fn gpu_is_default(g: &GpuKind) -> bool {
    *g == GpuKind::None
}

#[derive(Serialize, Clone, Debug)]
pub struct Endpoint {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Inference runtime behind this endpoint (for metrics interpretation).
    #[serde(default, skip_serializing_if = "runtime_is_default")]
    pub runtime: Runtime,
    /// Prometheus `/metrics` URL to scrape (vLLM, or llama.cpp with `--metrics`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prometheus_url: Option<String>,
    /// GPU metrics source for this endpoint's host.
    #[serde(default, skip_serializing_if = "gpu_is_default")]
    pub gpu: GpuKind,
    /// `hchat-agent` base URL when `gpu = agent` (e.g. http://spark:9099).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_url: Option<String>,
}

impl<'de> Deserialize<'de> for Endpoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum EndpointRepr {
            Simple(String),
            Full {
                url: String,
                #[serde(default)]
                api_key: Option<String>,
                #[serde(default)]
                runtime: Runtime,
                #[serde(default)]
                prometheus_url: Option<String>,
                #[serde(default)]
                gpu: GpuKind,
                #[serde(default)]
                agent_url: Option<String>,
            },
        }

        match EndpointRepr::deserialize(deserializer)? {
            EndpointRepr::Simple(url) => Ok(Endpoint::new(url)),
            EndpointRepr::Full {
                url,
                api_key,
                runtime,
                prometheus_url,
                gpu,
                agent_url,
            } => Ok(Endpoint {
                url,
                api_key,
                runtime,
                prometheus_url,
                gpu,
                agent_url,
            }),
        }
    }
}

impl Endpoint {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            runtime: Runtime::default(),
            prometheus_url: None,
            gpu: GpuKind::default(),
            agent_url: None,
        }
    }
}

/// User-rebindable keyboard shortcuts. Combos use a simple `mod+key` syntax
/// where `mod` is Cmd on macOS / Ctrl elsewhere (e.g. `mod+n`, `mod+shift+f`,
/// `escape`). Interpreted on the frontend.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Hotkeys {
    pub new_chat: String,
    pub focus_input: String,
    pub find: String,
    pub settings: String,
    pub toggle_artifacts: String,
    pub stop: String,
}

impl Default for Hotkeys {
    fn default() -> Self {
        Self {
            new_chat: "mod+n".to_string(),
            focus_input: "mod+l".to_string(),
            find: "mod+f".to_string(),
            settings: "mod+,".to_string(),
            toggle_artifacts: "mod+j".to_string(),
            stop: "mod+.".to_string(),
        }
    }
}

fn default_stdio() -> String {
    "stdio".to_string()
}
fn default_true() -> bool {
    true
}

/// An MCP server hChat connects to. `transport = "stdio"` spawns `command`
/// (with `args`/`env`); `transport = "http"` connects to `url` (with `headers`).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct McpServer {
    pub name: String,
    #[serde(default = "default_stdio")]
    pub transport: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Run this server's tools without the per-call approval card.
    #[serde(default)]
    pub auto_approve: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Config {
    pub font_size: f32,
    pub mono_font_size: f32,
    pub ui_scale: f32,
    pub dark_mode: bool,
    pub default_endpoint: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub use_max_tokens: bool,
    pub saved_endpoints: Vec<Endpoint>,
    pub font_family: String,
    pub mono_font_family: String,
    /// Nucleus sampling. `None` (or `< 0`) means "don't send", which is what most
    /// users want — leaving it to the model default.
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    /// Stop sequences. Empty vec → don't send.
    pub stop_sequences: Vec<String>,
    /// Rebindable keyboard shortcuts.
    pub hotkeys: Hotkeys,
    /// MCP servers to connect to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServer>,
    /// Retention window for the recorded usage history, in days. `0` (the
    /// default) keeps everything forever; any positive value prunes usage rows
    /// older than that many days (on startup and whenever the Usage page loads).
    #[serde(default)]
    pub usage_retention_days: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            font_size: 14.0,
            mono_font_size: 13.0,
            ui_scale: 1.0,
            dark_mode: true,
            default_endpoint: "http://localhost:1234/v1".to_string(),
            system_prompt: String::new(),
            temperature: 0.7,
            max_tokens: 2048,
            use_max_tokens: false,
            saved_endpoints: vec![Endpoint::new("http://localhost:1234/v1")],
            font_family: String::new(),
            mono_font_family: String::new(),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: Vec::new(),
            hotkeys: Hotkeys::default(),
            mcp_servers: Vec::new(),
            usage_retention_days: 0,
        }
    }
}

impl Config {
    pub fn path() -> Result<PathBuf, String> {
        dirs::home_dir()
            .map(|h| h.join(".config/hchat/config.toml"))
            .ok_or_else(|| "Could not determine home directory".to_string())
    }

    pub fn load() -> Self {
        match Self::try_load() {
            Ok(config) => config,
            Err(e) => {
                eprintln!("Warning: {e}, using default config");
                Self::default()
            }
        }
    }

    /// Load config from disk, returning an error string on failure instead of
    /// silently falling back to defaults.
    pub fn try_load() -> Result<Self, String> {
        let path = Self::path()?;

        match fs::metadata(&path) {
            Ok(meta) => {
                if meta.len() > MAX_CONFIG_SIZE {
                    return Err(format!(
                        "Config file too large ({} bytes, max {MAX_CONFIG_SIZE})",
                        meta.len()
                    ));
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                let example_config = include_str!("../../example.config.toml");
                if let Some(parent) = path.parent() {
                    let _ = fs::create_dir_all(parent);
                    #[cfg(unix)]
                    set_dir_permissions(parent);
                }
                if let Err(e) = fs::write(&path, example_config) {
                    eprintln!("Warning: could not write default config: {e}");
                }
                return Ok(Self::default());
            }
            Err(e) => {
                return Err(format!("Could not read config file: {e}"));
            }
        }

        let contents =
            fs::read_to_string(&path).map_err(|e| format!("Could not read config file: {e}"))?;
        let mut config: Config = match toml::from_str(&contents) {
            Ok(c) => c,
            Err(e) => {
                // Don't silently overwrite the user's settings with defaults —
                // back up the broken file so they can recover, then surface
                // the parse error. The fallback in `load()` will pick up
                // defaults but the original is preserved on disk.
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let backup = path.with_extension(format!("toml.broken-{stamp}"));
                if let Err(rename_err) = fs::rename(&path, &backup) {
                    eprintln!(
                        "Warning: could not back up corrupt config to {}: {rename_err}",
                        backup.display()
                    );
                } else {
                    eprintln!(
                        "Note: corrupt config backed up to {}",
                        backup.display()
                    );
                }
                return Err(format!("Could not parse config: {e}"));
            }
        };
        config.sanitize();
        Ok(config)
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Could not create config directory: {e}"))?;
            #[cfg(unix)]
            set_dir_permissions(parent);
        }
        let contents =
            toml::to_string_pretty(self).map_err(|e| format!("Could not serialize config: {e}"))?;
        let tmp = path.with_extension("toml.tmp");
        fs::write(&tmp, &contents).map_err(|e| format!("Could not write config file: {e}"))?;
        fs::rename(&tmp, &path).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            format!("Could not finalize config file: {e}")
        })?;
        Ok(())
    }

    pub fn sanitize(&mut self) {
        if !self.font_size.is_finite() || !(8.0..=24.0).contains(&self.font_size) {
            self.font_size = 14.0;
        }
        if !self.mono_font_size.is_finite() || !(8.0..=24.0).contains(&self.mono_font_size) {
            self.mono_font_size = 13.0;
        }
        if !self.ui_scale.is_finite() || !(0.75..=2.0).contains(&self.ui_scale) {
            self.ui_scale = 1.0;
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            self.temperature = 0.7;
        }
        if !(64..=16384).contains(&self.max_tokens) {
            self.max_tokens = 2048;
        }
        if let Some(v) = self.top_p {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                self.top_p = None;
            }
        }
        if let Some(v) = self.frequency_penalty {
            if !v.is_finite() || !(-2.0..=2.0).contains(&v) {
                self.frequency_penalty = None;
            }
        }
        if let Some(v) = self.presence_penalty {
            if !v.is_finite() || !(-2.0..=2.0).contains(&v) {
                self.presence_penalty = None;
            }
        }
        // Drop empty stop sequences and cap at 4 (OpenAI hard limit).
        self.stop_sequences
            .retain(|s| !s.is_empty() && s.len() <= 256);
        self.stop_sequences.truncate(4);
        if self.saved_endpoints.is_empty() {
            self.saved_endpoints = vec![Endpoint::new(self.default_endpoint.clone())];
        }
        for ep in &mut self.saved_endpoints {
            ep.url = ep.url.trim().to_string();
        }
        self.saved_endpoints.retain(|ep| !ep.url.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_config_parses_and_sanitizes() {
        let raw = include_str!("../../example.config.toml");
        let mut cfg: Config = toml::from_str(raw).expect("example.config.toml must be valid TOML");
        cfg.sanitize();
        // Hotkeys table is present with the documented defaults.
        assert_eq!(cfg.hotkeys.settings, "mod+,");
        assert_eq!(cfg.hotkeys.new_chat, "mod+n");
        assert!(!cfg.saved_endpoints.is_empty());
    }
}
