//! Tool definitions and the loader that walks `~/.config/hchat/tools/`.
//!
//! A tool is described by a TOML file with at minimum `name`, `description`,
//! `parameters` (JSON Schema, passed through to the API verbatim), and
//! `handler`. The handler is either `"builtin:<name>"` to invoke hardcoded
//! Rust (the 5 default tools ship this way) or `{ shell = [...argv] }` for
//! user-defined wrappers around shell commands.

use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the function's parameters. Stored as a free-form
    /// `serde_json::Value` because we don't validate it ourselves — the
    /// LLM provider does. Passes through to the API's `tools[].function.
    /// parameters` field unchanged.
    pub parameters: serde_json::Value,
    pub handler: Handler,
    #[serde(default)]
    pub safety: Safety,
}

/// Two handler shapes:
/// - `handler = "builtin:read_file"` → `Handler::Builtin("read_file")`
///   (the prefix is stripped on parse). Dispatched to native Rust.
/// - `handler = { shell = ["git", "log", "-n", "{{count}}"] }` →
///   `Handler::Shell(...)`. Forked + execed at run time with `{{var}}`
///   placeholders substituted from the call's arguments.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Handler {
    Builtin(BuiltinRef),
    Shell { shell: Vec<String> },
}

#[derive(Debug, Clone)]
pub struct BuiltinRef(pub String);

impl<'de> Deserialize<'de> for BuiltinRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        match s.strip_prefix("builtin:") {
            Some(name) if !name.is_empty() => Ok(BuiltinRef(name.to_string())),
            _ => Err(serde::de::Error::custom(format!(
                "expected handler = \"builtin:<name>\", got '{s}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Safety {
    /// Execute without prompting. The default — fits read-only tools.
    #[default]
    Auto,
    /// Show an approval card before each invocation. Use for write_file,
    /// run_shell, and any user-defined tool that mutates state.
    Confirm,
}

/// User config dir for tool TOMLs. `dirs::config_dir()` per platform plus
/// the `hchat/tools/` subpath.
pub fn user_tools_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hchat")
        .join("tools")
}

/// Load every `*.toml` file in `dir` as a ToolDef. Files that fail to parse
/// are reported on stderr and skipped — one bad tool shouldn't disable all
/// the others. Returns an empty vec if the directory doesn't exist.
pub fn load_from_dir(dir: &Path) -> Vec<ToolDef> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        match parse_file(&path) {
            Ok(def) => out.push(def),
            Err(e) => eprintln!("Tool load warning: {}: {e}", path.display()),
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn parse_file(path: &Path) -> Result<ToolDef, String> {
    let body = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    toml::from_str::<ToolDef>(&body).map_err(|e| e.to_string())
}

/// On first launch (or whenever the user's tools dir is empty), seed the
/// directory with the 5 default tool TOMLs that ship in the binary. Users
/// can edit, delete, or replace them — we don't overwrite on subsequent
/// launches.
pub fn seed_defaults_if_empty(dir: &Path) {
    if dir.exists()
        && std::fs::read_dir(dir)
            .map(|mut it| it.next().is_some())
            .unwrap_or(false)
    {
        return;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    for (name, body) in DEFAULT_TOOLS {
        let path = dir.join(name);
        let _ = std::fs::write(&path, body);
    }
}

/// Convert a ToolDef into the OpenAI `tools[]` JSON shape:
/// `{"type":"function","function":{"name","description","parameters"}}`.
/// Done at send time so we don't store the wire shape — `ToolDef` is the
/// in-process truth.
pub fn to_api_shape(defs: &[ToolDef]) -> Vec<serde_json::Value> {
    defs.iter()
        .map(|d| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.parameters,
                },
            })
        })
        .collect()
}

// ---- builtin handlers ----

/// Caps and tunables for builtin tools. Centralized so tests can refer to
/// the same constants the production code does.
pub mod limits {
    pub const READ_FILE_MAX_BYTES: usize = 100 * 1024;
    pub const SEARCH_MAX_MATCHES: usize = 200;
    pub const SHELL_OUTPUT_MAX_BYTES: usize = 50 * 1024;
    /// Per-call wall-clock cap on shell tool execution. Sized for typical
    /// dev workflows (`cargo build`, `cargo test`, `npm install`) which
    /// routinely exceed a minute. A runaway model can still hit this and
    /// stop, but legitimate work isn't cut off prematurely.
    pub const SHELL_TIMEOUT_SECS: u64 = 300;
}

/// Resolve a path argument: absolute paths pass through, relative paths
/// join onto `working_dir`. Symlinks are NOT canonicalized — we want the
/// user-facing path in error messages, and the tool surface intentionally
/// trusts the user to pick what they're willing to expose.
fn resolve_path(arg: &str, working_dir: &Path) -> PathBuf {
    let p = Path::new(arg);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        working_dir.join(p)
    }
}

fn arg_str<'a>(args: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing or non-string argument: {key}"))
}

/// Run a shell-handler tool: substitute `{{name}}` placeholders in the argv
/// template with values from the call's arguments object, then exec it
/// with `working_dir` as cwd. Captures stdout + stderr + exit status.
pub fn run_shell_tool(
    template: &[String],
    args: &serde_json::Value,
    working_dir: &Path,
) -> Result<String, String> {
    use std::process::{Command, Stdio};
    if template.is_empty() {
        return Err("shell tool template is empty".to_string());
    }
    let resolved: Vec<String> = template
        .iter()
        .map(|piece| substitute_placeholders(piece, args))
        .collect();
    let mut cmd = Command::new(&resolved[0]);
    cmd.args(&resolved[1..])
        .current_dir(working_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(limits::SHELL_TIMEOUT_SECS);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    return Err(format!("timed out after {}s", limits::SHELL_TIMEOUT_SECS));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => return Err(format!("wait failed: {e}")),
        }
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("collect failed: {e}"))?;
    let mut combined = String::new();
    if !out.stdout.is_empty() {
        combined.push_str("--- stdout ---\n");
        combined.push_str(&String::from_utf8_lossy(&out.stdout));
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }
    if !out.stderr.is_empty() {
        combined.push_str("--- stderr ---\n");
        combined.push_str(&String::from_utf8_lossy(&out.stderr));
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }
    combined.push_str(&format!(
        "--- exit: {} ---",
        out.status.code().unwrap_or(-1)
    ));
    Ok(truncate_with_marker(&combined, limits::SHELL_OUTPUT_MAX_BYTES))
}

/// Replace `{{name}}` occurrences in `s` with the string value of the
/// matching key in `args` (an object). Missing keys substitute as empty
/// string. Non-string values are stringified via `Display` for primitives,
/// `to_string()` for objects/arrays.
fn substitute_placeholders(s: &str, args: &serde_json::Value) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'{' && bytes[i + 1] == b'{' {
            // Find matching '}}' — scan ahead.
            if let Some(end) = find_close(s, i + 2) {
                let key = &s[i + 2..end];
                let trimmed = key.trim();
                let value = match args.get(trimmed) {
                    Some(serde_json::Value::String(v)) => v.clone(),
                    Some(serde_json::Value::Null) | None => String::new(),
                    Some(other) => other.to_string(),
                };
                out.push_str(&value);
                i = end + 2;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn find_close(s: &str, from: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = from;
    while i + 1 < bytes.len() {
        if bytes[i] == b'}' && bytes[i + 1] == b'}' {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Dispatch a builtin tool by name. Unknown names return an error rather
/// than panicking — a stale TOML referring to a removed builtin shouldn't
/// crash the app.
pub fn run_builtin(
    name: &str,
    args: &serde_json::Value,
    working_dir: &Path,
) -> Result<String, String> {
    match name {
        "read_file" => builtin_read_file(args, working_dir),
        "list_directory" => builtin_list_directory(args, working_dir),
        "search_files" => builtin_search_files(args, working_dir),
        "write_file" => builtin_write_file(args, working_dir),
        "run_shell" => builtin_run_shell(args, working_dir),
        other => Err(format!("unknown builtin tool: {other}")),
    }
}

fn builtin_read_file(args: &serde_json::Value, wd: &Path) -> Result<String, String> {
    let path = resolve_path(arg_str(args, "path")?, wd);
    let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
    let limit = args.get("limit").and_then(|v| v.as_u64());
    let body = std::fs::read_to_string(&path)
        .map_err(|e| format!("read failed: {}: {e}", path.display()))?;

    if offset == 0 && limit.is_none() {
        return Ok(truncate_with_marker(&body, limits::READ_FILE_MAX_BYTES));
    }

    // Line slicing path. `offset` is 1-indexed per the TOML doc.
    let start = offset.saturating_sub(1) as usize;
    let take = limit.map(|n| n as usize).unwrap_or(usize::MAX);
    let mut out = String::new();
    for line in body.lines().skip(start).take(take) {
        out.push_str(line);
        out.push('\n');
        if out.len() >= limits::READ_FILE_MAX_BYTES {
            break;
        }
    }
    Ok(truncate_with_marker(&out, limits::READ_FILE_MAX_BYTES))
}

fn builtin_list_directory(args: &serde_json::Value, wd: &Path) -> Result<String, String> {
    let path = resolve_path(arg_str(args, "path")?, wd);
    let entries = std::fs::read_dir(&path)
        .map_err(|e| format!("list failed: {}: {e}", path.display()))?;
    let mut rows: Vec<(bool, String)> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        rows.push((is_dir, name));
    }
    // Directories first, then files; both alphabetical.
    rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut out = String::new();
    for (is_dir, name) in rows {
        out.push_str(if is_dir { "d/ " } else { "f/ " });
        out.push_str(&name);
        out.push('\n');
    }
    if out.is_empty() {
        out.push_str("(empty directory)\n");
    }
    Ok(out)
}

fn builtin_search_files(args: &serde_json::Value, wd: &Path) -> Result<String, String> {
    let pattern = arg_str(args, "pattern")?;
    let root = resolve_path(arg_str(args, "path")?, wd);
    let re = regex::Regex::new(pattern).map_err(|e| format!("invalid regex: {e}"))?;
    let mut matches: Vec<String> = Vec::new();
    walk_search(&root, &re, &mut matches);
    if matches.is_empty() {
        return Ok("(no matches)".to_string());
    }
    Ok(matches.join("\n"))
}

/// Recursively walk `dir`, reading text files and emitting `path:lineno: line`
/// for each regex match. Skips hidden directories (`.git`, `.cache`, etc.)
/// and binary files (heuristic: contains a NUL byte in the first 1KB).
/// Stops early once `SEARCH_MAX_MATCHES` is reached.
fn walk_search(dir: &Path, re: &regex::Regex, out: &mut Vec<String>) {
    if out.len() >= limits::SEARCH_MAX_MATCHES {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        if out.len() >= limits::SEARCH_MAX_MATCHES {
            return;
        }
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            walk_search(&path, re, out);
        } else if ft.is_file() {
            search_one_file(&path, re, out);
        }
    }
}

fn search_one_file(path: &Path, re: &regex::Regex, out: &mut Vec<String>) {
    let body = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return,
    };
    // Binary heuristic: a NUL byte in the head means we treat as binary.
    if body.iter().take(1024).any(|&b| b == 0) {
        return;
    }
    let text = match std::str::from_utf8(&body) {
        Ok(s) => s,
        Err(_) => return,
    };
    for (lineno, line) in text.lines().enumerate() {
        if out.len() >= limits::SEARCH_MAX_MATCHES {
            return;
        }
        if re.is_match(line) {
            out.push(format!("{}:{}: {}", path.display(), lineno + 1, line));
        }
    }
}

fn builtin_write_file(args: &serde_json::Value, wd: &Path) -> Result<String, String> {
    let path = resolve_path(arg_str(args, "path")?, wd);
    let content = arg_str(args, "content")?;
    let append = args.get("append").and_then(|v| v.as_bool()).unwrap_or(false);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir failed: {}: {e}", parent.display()))?;
    }
    if append {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("open failed: {}: {e}", path.display()))?;
        f.write_all(content.as_bytes())
            .map_err(|e| format!("write failed: {}: {e}", path.display()))?;
    } else {
        std::fs::write(&path, content)
            .map_err(|e| format!("write failed: {}: {e}", path.display()))?;
    }
    Ok(format!(
        "wrote {} bytes to {}",
        content.len(),
        path.display()
    ))
}

fn builtin_run_shell(args: &serde_json::Value, wd: &Path) -> Result<String, String> {
    let command = arg_str(args, "command")?;
    use std::process::{Command, Stdio};
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(wd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn failed: {e}"))?;

    // Wait with a timeout so a runaway command can't hang the app forever.
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(limits::SHELL_TIMEOUT_SECS);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    return Err(format!(
                        "timed out after {}s",
                        limits::SHELL_TIMEOUT_SECS
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => return Err(format!("wait failed: {e}")),
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("collect failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let exit = output.status.code().unwrap_or(-1);
    let mut combined = String::new();
    if !stdout.is_empty() {
        combined.push_str("--- stdout ---\n");
        combined.push_str(&stdout);
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }
    if !stderr.is_empty() {
        combined.push_str("--- stderr ---\n");
        combined.push_str(&stderr);
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }
    combined.push_str(&format!("--- exit: {exit} ---"));
    Ok(truncate_with_marker(&combined, limits::SHELL_OUTPUT_MAX_BYTES))
}

/// Truncate a string to `max_bytes` and append a marker so the model knows
/// the result was clipped. Char-aware so we don't slice through a multi-
/// byte UTF-8 boundary.
fn truncate_with_marker(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n... [truncated, {} more bytes]", &s[..end], s.len() - end)
}

/// Embedded TOML for the 5 default tools. `(filename, contents)` pairs.
const DEFAULT_TOOLS: &[(&str, &str)] = &[
    (
        "read_file.toml",
        include_str!("default_tools/read_file.toml"),
    ),
    (
        "list_directory.toml",
        include_str!("default_tools/list_directory.toml"),
    ),
    (
        "search_files.toml",
        include_str!("default_tools/search_files.toml"),
    ),
    (
        "write_file.toml",
        include_str!("default_tools/write_file.toml"),
    ),
    (
        "run_shell.toml",
        include_str!("default_tools/run_shell.toml"),
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_builtin_handler() {
        let body = r#"
name = "read_file"
description = "Read a file"
handler = "builtin:read_file"

[parameters]
type = "object"
"#;
        let def: ToolDef = toml::from_str(body).unwrap();
        assert_eq!(def.name, "read_file");
        match def.handler {
            Handler::Builtin(BuiltinRef(name)) => assert_eq!(name, "read_file"),
            _ => panic!("expected Builtin"),
        }
        assert_eq!(def.safety, Safety::Auto);
    }

    #[test]
    fn parses_shell_handler() {
        let body = r#"
name = "git_status"
description = "Show git status"
handler = { shell = ["git", "status", "--short"] }
safety = "confirm"

[parameters]
type = "object"
"#;
        let def: ToolDef = toml::from_str(body).unwrap();
        assert_eq!(def.name, "git_status");
        match def.handler {
            Handler::Shell { shell } => assert_eq!(shell, vec!["git", "status", "--short"]),
            _ => panic!("expected Shell"),
        }
        assert_eq!(def.safety, Safety::Confirm);
    }

    #[test]
    fn rejects_handler_missing_builtin_prefix() {
        let body = r#"
name = "x"
description = "x"
handler = "read_file"

[parameters]
type = "object"
"#;
        assert!(toml::from_str::<ToolDef>(body).is_err());
    }

    #[test]
    fn all_default_tomls_parse() {
        // Sanity check — the 5 embedded TOMLs must be valid ToolDefs.
        for (name, body) in DEFAULT_TOOLS {
            toml::from_str::<ToolDef>(body)
                .unwrap_or_else(|e| panic!("default tool {name} failed to parse: {e}"));
        }
    }

    #[test]
    fn write_tool_defaults_to_confirm() {
        // Safety regression test: write_file and run_shell MUST default to
        // "confirm" — losing this would silently auto-execute destructive
        // calls.
        for (filename, body) in DEFAULT_TOOLS {
            let def: ToolDef = toml::from_str(body).unwrap();
            match def.name.as_str() {
                "write_file" | "run_shell" => assert_eq!(
                    def.safety,
                    Safety::Confirm,
                    "{filename} must default to safety=confirm"
                ),
                _ => {}
            }
        }
    }

    #[test]
    fn read_only_tools_default_to_auto() {
        for (filename, body) in DEFAULT_TOOLS {
            let def: ToolDef = toml::from_str(body).unwrap();
            match def.name.as_str() {
                "read_file" | "list_directory" | "search_files" => assert_eq!(
                    def.safety,
                    Safety::Auto,
                    "{filename} should default to safety=auto"
                ),
                _ => {}
            }
        }
    }

    // ----- builtin handlers -----

    /// Lightweight temp-dir helper — avoids the `tempdir` crate dep just
    /// for tests. PID + nanos suffix so parallel test runs don't collide.
    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nonce: u128 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("hchat-tools-test-{pid}-{nonce}-{name}"));
        std::fs::create_dir_all(&p).expect("mkdir tmp");
        p
    }

    #[test]
    fn read_file_returns_contents() {
        let dir = tmp_path("read");
        let f = dir.join("hello.txt");
        std::fs::write(&f, "hello world\nsecond line").unwrap();
        let args = serde_json::json!({"path": "hello.txt"});
        let out = run_builtin("read_file", &args, &dir).unwrap();
        assert!(out.contains("hello world"));
    }

    #[test]
    fn read_file_resolves_absolute_paths() {
        let dir = tmp_path("read_abs");
        let f = dir.join("note.md");
        std::fs::write(&f, "# header").unwrap();
        // Pass absolute path; working_dir is irrelevant.
        let args = serde_json::json!({"path": f.to_string_lossy()});
        let elsewhere = std::env::temp_dir();
        let out = run_builtin("read_file", &args, &elsewhere).unwrap();
        assert!(out.contains("# header"));
    }

    #[test]
    fn read_file_offset_and_limit() {
        let dir = tmp_path("read_lines");
        let f = dir.join("multi.txt");
        std::fs::write(&f, "1\n2\n3\n4\n5\n").unwrap();
        let args = serde_json::json!({"path": "multi.txt", "offset": 2, "limit": 2});
        let out = run_builtin("read_file", &args, &dir).unwrap();
        assert!(out.contains("2\n3\n"), "got: {out:?}");
        assert!(!out.contains("4"));
    }

    #[test]
    fn read_file_missing_path_arg_errors() {
        let args = serde_json::json!({});
        assert!(run_builtin("read_file", &args, &std::env::temp_dir()).is_err());
    }

    #[test]
    fn list_directory_marks_files_and_dirs() {
        let dir = tmp_path("list");
        std::fs::write(dir.join("a.txt"), "x").unwrap();
        std::fs::create_dir(dir.join("subdir")).unwrap();
        let out =
            run_builtin("list_directory", &serde_json::json!({"path": "."}), &dir).unwrap();
        assert!(out.contains("d/ subdir"), "got: {out}");
        assert!(out.contains("f/ a.txt"), "got: {out}");
    }

    #[test]
    fn search_files_finds_matches_and_skips_dotdirs() {
        let dir = tmp_path("search");
        std::fs::write(dir.join("foo.txt"), "needle here\nother line").unwrap();
        let dotdir = dir.join(".hidden");
        std::fs::create_dir(&dotdir).unwrap();
        std::fs::write(dotdir.join("bar.txt"), "needle in hidden").unwrap();
        let out = run_builtin(
            "search_files",
            &serde_json::json!({"pattern": "needle", "path": "."}),
            &dir,
        )
        .unwrap();
        assert!(out.contains("foo.txt"), "got: {out}");
        assert!(!out.contains(".hidden"), "should skip dotdirs: {out}");
    }

    #[test]
    fn write_file_creates_parent_dirs() {
        let dir = tmp_path("write");
        let args =
            serde_json::json!({"path": "nested/sub/out.txt", "content": "body"});
        let r = run_builtin("write_file", &args, &dir);
        assert!(r.is_ok(), "{r:?}");
        let written = std::fs::read_to_string(dir.join("nested/sub/out.txt")).unwrap();
        assert_eq!(written, "body");
    }

    #[test]
    fn write_file_append_mode() {
        let dir = tmp_path("write_append");
        let args1 = serde_json::json!({"path": "a.txt", "content": "first"});
        run_builtin("write_file", &args1, &dir).unwrap();
        let args2 = serde_json::json!({"path": "a.txt", "content": "+second", "append": true});
        run_builtin("write_file", &args2, &dir).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "first+second");
    }

    #[test]
    fn run_shell_captures_stdout_and_exit() {
        let dir = tmp_path("shell");
        let out = run_builtin(
            "run_shell",
            &serde_json::json!({"command": "echo hi && exit 0"}),
            &dir,
        )
        .unwrap();
        assert!(out.contains("hi"), "got: {out}");
        assert!(out.contains("exit: 0"), "got: {out}");
    }

    #[test]
    fn run_shell_uses_working_dir_as_cwd() {
        let dir = tmp_path("shell_cwd");
        std::fs::write(dir.join("marker"), "found").unwrap();
        let out = run_builtin(
            "run_shell",
            &serde_json::json!({"command": "ls"}),
            &dir,
        )
        .unwrap();
        assert!(out.contains("marker"), "got: {out}");
    }

    #[test]
    fn unknown_builtin_returns_error_not_panic() {
        assert!(run_builtin("nonexistent", &serde_json::json!({}), &std::env::temp_dir()).is_err());
    }

    #[test]
    fn substitute_placeholders_string_value() {
        let args = serde_json::json!({"name": "alice", "id": 42});
        assert_eq!(
            substitute_placeholders("hello {{name}} #{{id}}", &args),
            "hello alice #42"
        );
    }

    #[test]
    fn substitute_placeholders_missing_key_yields_empty() {
        let args = serde_json::json!({});
        assert_eq!(substitute_placeholders("--{{x}}--", &args), "----");
    }

    #[test]
    fn to_api_shape_emits_function_envelope() {
        let body = r#"
name = "noop"
description = "does nothing"
handler = "builtin:read_file"

[parameters]
type = "object"
"#;
        let def: ToolDef = toml::from_str(body).unwrap();
        let shape = to_api_shape(&[def]);
        assert_eq!(shape.len(), 1);
        assert_eq!(shape[0]["type"], "function");
        assert_eq!(shape[0]["function"]["name"], "noop");
        assert_eq!(shape[0]["function"]["description"], "does nothing");
        assert!(shape[0]["function"]["parameters"].is_object());
    }

}
