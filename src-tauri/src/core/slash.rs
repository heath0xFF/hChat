/// Slash-command parser for the chat input.
///
/// Anything starting with `/` and a known verb is intercepted as a command;
/// everything else passes through as a regular message. Unknown verbs are
/// reported back as an error so users notice typos instead of silently sending
/// `/temprature 0.5` to the model.
#[derive(Debug, PartialEq)]
pub enum Command {
    /// Switch active model. Argument is matched as substring against the model
    /// list; the first match wins.
    Model(String),
    /// Set sampling temperature. Clamped to [0.0, 2.0] by the caller.
    Temperature(f32),
    /// Replace the system prompt for the current conversation.
    System(String),
    /// Clear current conversation (start fresh).
    Clear,
    /// Copy the last assistant message to clipboard.
    Copy,
    /// Show the help banner.
    Help,
}

#[derive(Debug, PartialEq)]
pub enum ParseResult {
    /// Recognised command, ready to execute.
    Command(Command),
    /// Looked like a command (`/foo`) but the verb is unknown.
    Unknown(String),
    /// Recognised verb but the argument was missing or malformed.
    BadArgs { verb: String, reason: String },
    /// Not a command at all — send as a normal message.
    NotACommand,
}

pub fn parse(input: &str) -> ParseResult {
    let trimmed = input.trim_start();
    if !trimmed.starts_with('/') {
        return ParseResult::NotACommand;
    }
    // A bare `/` or whitespace-only after `/` is just a literal message.
    let body = trimmed[1..].trim_end();
    if body.is_empty() {
        return ParseResult::NotACommand;
    }

    let (verb, rest) = match body.split_once(char::is_whitespace) {
        Some((v, r)) => (v, r.trim()),
        None => (body, ""),
    };
    let verb_lower = verb.to_ascii_lowercase();

    match verb_lower.as_str() {
        "model" | "m" => {
            if rest.is_empty() {
                ParseResult::BadArgs {
                    verb: "model".into(),
                    reason: "usage: /model <name-or-substring>".into(),
                }
            } else {
                ParseResult::Command(Command::Model(rest.to_string()))
            }
        }
        "temp" | "temperature" | "t" => match rest.parse::<f32>() {
            Ok(v) if v.is_finite() => ParseResult::Command(Command::Temperature(v)),
            _ => ParseResult::BadArgs {
                verb: "temp".into(),
                reason: "usage: /temp <0.0..=2.0>".into(),
            },
        },
        "system" | "sys" => ParseResult::Command(Command::System(rest.to_string())),
        "clear" | "new" => ParseResult::Command(Command::Clear),
        "copy" => ParseResult::Command(Command::Copy),
        "help" | "?" | "h" => ParseResult::Command(Command::Help),
        _ => ParseResult::Unknown(verb.to_string()),
    }
}

pub fn help_text() -> &'static str {
    "Slash commands:\n\
     • /model <name>  — switch model (substring match)\n\
     • /temp <0..2>   — set temperature\n\
     • /system <text> — set system prompt\n\
     • /clear         — start a new conversation\n\
     • /copy          — copy last reply to clipboard\n\
     • /help          — show this message"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_message_is_not_a_command() {
        assert_eq!(parse("hello"), ParseResult::NotACommand);
        assert_eq!(parse("  hello /world"), ParseResult::NotACommand);
        assert_eq!(parse(""), ParseResult::NotACommand);
        assert_eq!(parse("/"), ParseResult::NotACommand);
        assert_eq!(parse("/   "), ParseResult::NotACommand);
    }

    #[test]
    fn model_command() {
        assert_eq!(
            parse("/model gpt-4o"),
            ParseResult::Command(Command::Model("gpt-4o".into()))
        );
        assert_eq!(
            parse("  /model   foo bar  "),
            ParseResult::Command(Command::Model("foo bar".into()))
        );
        assert!(matches!(parse("/model"), ParseResult::BadArgs { .. }));
    }

    #[test]
    fn temp_command() {
        assert_eq!(
            parse("/temp 0.7"),
            ParseResult::Command(Command::Temperature(0.7))
        );
        assert_eq!(
            parse("/t 1.2"),
            ParseResult::Command(Command::Temperature(1.2))
        );
        assert!(matches!(parse("/temp abc"), ParseResult::BadArgs { .. }));
        assert!(matches!(parse("/temp"), ParseResult::BadArgs { .. }));
    }

    #[test]
    fn system_command_allows_empty() {
        // /system with no args means clear the system prompt
        assert_eq!(
            parse("/system"),
            ParseResult::Command(Command::System(String::new()))
        );
        assert_eq!(
            parse("/system  you are helpful  "),
            ParseResult::Command(Command::System("you are helpful".into()))
        );
    }

    #[test]
    fn unknown_verb() {
        assert_eq!(
            parse("/foobar"),
            ParseResult::Unknown("foobar".into())
        );
    }

    #[test]
    fn case_insensitive_verbs() {
        assert!(matches!(
            parse("/MODEL gpt-4o"),
            ParseResult::Command(Command::Model(_))
        ));
        assert!(matches!(
            parse("/Help"),
            ParseResult::Command(Command::Help)
        ));
    }
}
