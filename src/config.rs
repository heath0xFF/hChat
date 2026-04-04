use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::PathBuf;

const MAX_CONFIG_SIZE: u64 = 1_048_576; // 1 MB

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
    pub saved_endpoints: Vec<String>,
    pub font_family: String,
    pub mono_font_family: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            font_size: 14.0,
            mono_font_size: 13.0,
            ui_scale: 1.0,
            dark_mode: true,
            default_endpoint: "http://localhost:11434/v1".to_string(),
            system_prompt: String::new(),
            temperature: 0.7,
            max_tokens: 2048,
            use_max_tokens: false,
            saved_endpoints: vec!["http://localhost:11434/v1".to_string()],
            font_family: String::new(),
            mono_font_family: String::new(),
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
        let path = match Self::path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: {e}, using default config");
                return Self::default();
            }
        };

        match fs::metadata(&path) {
            Ok(meta) => {
                if meta.len() > MAX_CONFIG_SIZE {
                    eprintln!(
                        "Warning: config file too large ({} bytes), using defaults",
                        meta.len()
                    );
                    return Self::default();
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                let config = Self::default();
                if let Err(e) = config.save() {
                    eprintln!("Warning: could not save default config: {e}");
                }
                return config;
            }
            Err(e) => {
                eprintln!("Warning: could not read config file: {e}");
                return Self::default();
            }
        }

        match fs::read_to_string(&path) {
            Ok(contents) => {
                let mut config: Config = toml::from_str(&contents).unwrap_or_else(|e| {
                    eprintln!("Warning: could not parse config: {e}, using defaults");
                    Self::default()
                });
                config.sanitize();
                config
            }
            Err(e) => {
                eprintln!("Warning: could not read config file: {e}");
                Self::default()
            }
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Could not create config directory: {e}"))?;
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
        if self.saved_endpoints.is_empty() {
            self.saved_endpoints = vec![self.default_endpoint.clone()];
        }
    }
}
