//! `~/.agents` convention loader — discovers slash commands, skills, and tools
//! from a shared, portable directory layout (see dot-agents.com), plus a
//! project-local `.agents/` in the conversation's working directory.
//!
//! Layout (any subset may exist):
//! ```text
//! ~/.agents/
//!   commands/<name>.md            # → a /command; body is a prompt template
//!   skills/<name>/SKILL.md        # frontmatter (name, description) + instructions
//!   tools/<name>.toml             # same format as Fornax's own tools
//!   local/…                       # machine-specific overrides (same shape)
//! <working_dir>/.agents/…         # project-local, overrides user-level
//! ```
//! Precedence (later wins on name collision): `~/.agents` < `~/.agents/local`
//! < `<working_dir>/.agents`.

use crate::tools::{self, ToolDef};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const MAX_FILE_BYTES: u64 = 512 * 1024;

#[derive(Clone, serde::Serialize)]
pub struct AgentCommand {
    pub name: String,
    pub description: String,
    /// Prompt template; `$ARGUMENTS` / `{{args}}` are substituted at use time.
    pub body: String,
}

#[derive(Clone, serde::Serialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Full instructions (the SKILL.md body after frontmatter).
    pub body: String,
}

#[derive(Default)]
pub struct AgentBundle {
    pub commands: Vec<AgentCommand>,
    pub skills: Vec<Skill>,
    pub tools: Vec<ToolDef>,
}

/// Roots in increasing precedence order.
fn roots(working_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(home) = dirs::home_dir() {
        out.push(home.join(".agents"));
        out.push(home.join(".agents").join("local"));
    }
    if let Some(wd) = working_dir {
        out.push(wd.join(".agents"));
    }
    out
}

/// Load commands + skills + tools from all roots, with later roots overriding
/// earlier ones by name.
pub fn load(working_dir: Option<&Path>) -> AgentBundle {
    // BTreeMap keeps output deterministic; insertion order across roots gives
    // precedence (later roots overwrite).
    let mut commands: BTreeMap<String, AgentCommand> = BTreeMap::new();
    let mut skills: BTreeMap<String, Skill> = BTreeMap::new();
    let mut tool_map: BTreeMap<String, ToolDef> = BTreeMap::new();

    for root in roots(working_dir) {
        for c in load_commands(&root.join("commands")) {
            commands.insert(c.name.clone(), c);
        }
        for s in load_skills(&root.join("skills")) {
            skills.insert(s.name.clone(), s);
        }
        for t in tools::load_from_dir(&root.join("tools")) {
            tool_map.insert(t.name.clone(), t);
        }
    }

    AgentBundle {
        commands: commands.into_values().collect(),
        skills: skills.into_values().collect(),
        tools: tool_map.into_values().collect(),
    }
}

fn read_capped(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > MAX_FILE_BYTES {
        eprintln!("agents: skipping oversized {} ({} bytes)", path.display(), meta.len());
        return None;
    }
    std::fs::read_to_string(path).ok()
}

fn load_commands(dir: &Path) -> Vec<AgentCommand> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(text) = read_capped(&path) else {
            continue;
        };
        let (fm, body) = split_frontmatter(&text);
        out.push(AgentCommand {
            name: stem.to_string(),
            description: fm_value(fm, "description").unwrap_or_default(),
            body: body.trim().to_string(),
        });
    }
    out
}

fn load_skills(dir: &Path) -> Vec<Skill> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let skill_md = path.join("SKILL.md");
        let Some(text) = read_capped(&skill_md) else {
            continue;
        };
        let (fm, body) = split_frontmatter(&text);
        let dir_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let name = fm_value(fm, "name").unwrap_or(dir_name);
        if name.is_empty() {
            continue;
        }
        out.push(Skill {
            name,
            description: fm_value(fm, "description").unwrap_or_default(),
            body: body.trim().to_string(),
        });
    }
    out
}

/// Split a leading `---\n…\n---\n` YAML frontmatter block from the body.
/// Returns `(frontmatter, body)`; frontmatter is empty when absent.
fn split_frontmatter(text: &str) -> (&str, &str) {
    let t = text.strip_prefix('\u{feff}').unwrap_or(text);
    let Some(rest) = t.strip_prefix("---\n").or_else(|| t.strip_prefix("---\r\n")) else {
        return ("", t);
    };
    // Find the closing `---` on its own line.
    for marker in ["\n---\n", "\n---\r\n", "\n---"] {
        if let Some(end) = rest.find(marker) {
            let fm = &rest[..end];
            let body = &rest[end + marker.len()..];
            return (fm, body);
        }
    }
    ("", t)
}

/// Pull a `key: value` out of a simple frontmatter block (no nested YAML).
fn fm_value(frontmatter: &str, key: &str) -> Option<String> {
    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start();
            if let Some(v) = rest.strip_prefix(':') {
                let v = v.trim().trim_matches('"').trim_matches('\'');
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_split_and_value() {
        let text = "---\nname: data-viz\ndescription: Make charts\n---\nDo the thing.\n";
        let (fm, body) = split_frontmatter(text);
        assert_eq!(fm_value(fm, "name").as_deref(), Some("data-viz"));
        assert_eq!(fm_value(fm, "description").as_deref(), Some("Make charts"));
        assert_eq!(body.trim(), "Do the thing.");
    }

    #[test]
    fn no_frontmatter_is_all_body() {
        let (fm, body) = split_frontmatter("just a prompt\n");
        assert_eq!(fm, "");
        assert_eq!(body.trim(), "just a prompt");
    }

    #[test]
    fn loads_project_local_commands_skills_tools() {
        let base = std::env::temp_dir().join(format!("fornax-agents-{}", std::process::id()));
        let a = base.join(".agents");
        std::fs::create_dir_all(a.join("commands")).unwrap();
        std::fs::create_dir_all(a.join("skills").join("code-review")).unwrap();
        std::fs::create_dir_all(a.join("tools")).unwrap();
        std::fs::write(
            a.join("commands").join("summarize.md"),
            "---\ndescription: Summarize\n---\nSummarize: $ARGUMENTS",
        )
        .unwrap();
        std::fs::write(
            a.join("skills").join("code-review").join("SKILL.md"),
            "---\nname: code-review\ndescription: Review code\n---\nReview carefully.",
        )
        .unwrap();
        std::fs::write(
            a.join("tools").join("echo.toml"),
            "name = \"agent_echo\"\ndescription = \"echo\"\nparameters = { type = \"object\" }\nhandler = { shell = [\"echo\", \"hi\"] }\n",
        )
        .unwrap();

        let bundle = load(Some(&base));
        let _ = std::fs::remove_dir_all(&base);

        assert!(
            bundle
                .commands
                .iter()
                .any(|c| c.name == "summarize" && c.body.contains("$ARGUMENTS"))
        );
        assert!(
            bundle
                .skills
                .iter()
                .any(|s| s.name == "code-review" && s.description == "Review code")
        );
        assert!(bundle.tools.iter().any(|t| t.name == "agent_echo"));
    }
}
