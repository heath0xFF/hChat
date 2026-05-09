/// Best-effort lookup of context window size for a model id.
///
/// Returns `None` for unknown models so the UI can show an em-dash instead of a
/// fake budget. Matches are substring-based, case-insensitive — the same model
/// surfaces under many names (`gpt-4o`, `openai/gpt-4o`, `gpt-4o-2024-08-06`).
pub fn context_window(model_id: &str) -> Option<u32> {
    let m = model_id.to_ascii_lowercase();

    // Order matters: more specific patterns first.
    const TABLE: &[(&str, u32)] = &[
        // OpenAI
        ("gpt-4.1", 1_047_576),
        ("gpt-4o-mini", 128_000),
        ("gpt-4o", 128_000),
        ("gpt-4-turbo", 128_000),
        ("gpt-4-32k", 32_768),
        ("gpt-4", 8_192),
        ("gpt-3.5-turbo-16k", 16_385),
        ("gpt-3.5", 16_385),
        ("o1-mini", 128_000),
        ("o1-preview", 128_000),
        ("o1", 200_000),
        ("o3-mini", 200_000),
        ("o3", 200_000),
        ("o4-mini", 200_000),
        // Anthropic
        ("claude-3-5-sonnet", 200_000),
        ("claude-3-5-haiku", 200_000),
        ("claude-3-opus", 200_000),
        ("claude-3-sonnet", 200_000),
        ("claude-3-haiku", 200_000),
        ("claude-sonnet-4", 200_000),
        ("claude-opus-4", 200_000),
        ("claude-haiku-4", 200_000),
        ("claude-", 200_000),
        // Google
        ("gemini-2.5-pro", 2_097_152),
        ("gemini-2.5-flash", 1_048_576),
        ("gemini-1.5-pro", 2_097_152),
        ("gemini-1.5-flash", 1_048_576),
        ("gemini-2.0", 1_048_576),
        ("gemini-", 32_768),
        // Meta / Llama
        ("llama-3.3", 128_000),
        ("llama-3.2", 128_000),
        ("llama-3.1", 128_000),
        ("llama3.3", 128_000),
        ("llama3.2", 128_000),
        ("llama3.1", 128_000),
        ("llama3", 8_192),
        ("llama2", 4_096),
        // Mistral
        ("mistral-large", 128_000),
        ("mistral-nemo", 128_000),
        ("mistral-small", 32_768),
        ("mistral-7b", 32_768),
        ("mistral", 32_768),
        ("mixtral-8x22b", 65_536),
        ("mixtral", 32_768),
        // Qwen
        ("qwen2.5", 32_768),
        ("qwen2", 32_768),
        ("qwen-", 32_768),
        ("qwq", 32_768),
        // DeepSeek
        ("deepseek-r1", 65_536),
        ("deepseek-v3", 65_536),
        ("deepseek-coder", 16_384),
        ("deepseek", 32_768),
        // Misc local
        ("phi-4", 16_384),
        ("phi-3", 128_000),
        ("phi3", 128_000),
        ("gemma-2", 8_192),
        ("gemma2", 8_192),
        ("gemma-3", 131_072),
        ("gemma3", 131_072),
        ("gemma", 8_192),
        ("codellama", 16_384),
        ("yi-", 32_768),
        ("command-r", 128_000),
        ("nemotron", 128_000),
        ("granite", 8_192),
    ];

    for (needle, ctx) in TABLE {
        if m.contains(needle) {
            return Some(*ctx);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_resolve() {
        assert_eq!(context_window("gpt-4o"), Some(128_000));
        assert_eq!(context_window("openai/gpt-4o-2024-08-06"), Some(128_000));
        assert_eq!(context_window("gpt-4o-mini"), Some(128_000));
        assert_eq!(context_window("claude-3-5-sonnet-20241022"), Some(200_000));
        assert_eq!(context_window("anthropic/claude-sonnet-4-5"), Some(200_000));
        assert_eq!(context_window("gemini-1.5-pro"), Some(2_097_152));
        assert_eq!(context_window("llama3.1:70b"), Some(128_000));
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(context_window("totally-made-up-model"), None);
        assert_eq!(context_window(""), None);
    }

    #[test]
    fn specificity_wins() {
        // gpt-4o-mini must resolve before gpt-4 generic — both are 128k here
        // but the assertion is that ordering is respected
        assert_eq!(context_window("gpt-4o-mini-2024-07-18"), Some(128_000));
        assert_eq!(context_window("gpt-4-32k-0613"), Some(32_768));
    }
}
