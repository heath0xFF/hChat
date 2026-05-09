/// Splits a chat message into segments so we can render fenced code blocks and
/// reasoning blocks ourselves while still letting `egui_commonmark` handle the
/// surrounding prose.
///
/// Why pre-split instead of forking the renderer: `egui_commonmark` 0.23
/// doesn't expose a code-block hook, and fenced code blocks are 99% of the
/// custom rendering we want anyway (per-block copy, language pill, reasoning
/// collapse).
#[derive(Debug, PartialEq)]
pub enum Segment<'a> {
    /// Plain markdown prose to feed back into the CommonMark renderer.
    Markdown(&'a str),
    /// Fenced code block. `lang` is the info string after the opening fence
    /// (often a language name; sometimes a filename or empty).
    Code { lang: &'a str, body: &'a str },
    /// `<think>...</think>` block from reasoning models. Body is the inner
    /// text; the UI renders it collapsed.
    Reasoning { body: &'a str, closed: bool },
}

/// Walks `text` and yields segments. Byte-level the segment slices cover the
/// whole input, but markdown semantics do not survive splitting: a numbered
/// list interrupted by a fenced block restarts at 1 in the next prose
/// segment, paragraphs straddling a `<think>` block get split, and reference
/// link definitions in one segment can't resolve in another. For chat output
/// these cases are rare; if they bite, the renderer caller should join
/// adjacent prose segments before passing to the markdown viewer.
pub fn segments(text: &str) -> Vec<Segment<'_>> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut prose_start = 0;

    while i < bytes.len() {
        // Reasoning block — only check at line start (or start of input).
        if at_line_start(text, i) {
            if let Some(end) = match_open(text, i, "<think>") {
                flush_prose(text, prose_start, i, &mut out);
                let body_start = end;
                let (body_end, closed) = find_close(text, body_start, "</think>");
                out.push(Segment::Reasoning {
                    body: &text[body_start..body_end],
                    closed,
                });
                i = if closed {
                    body_end + "</think>".len()
                } else {
                    text.len()
                };
                prose_start = i;
                continue;
            }
            if let Some(end) = match_open(text, i, "<thinking>") {
                flush_prose(text, prose_start, i, &mut out);
                let body_start = end;
                let (body_end, closed) = find_close(text, body_start, "</thinking>");
                out.push(Segment::Reasoning {
                    body: &text[body_start..body_end],
                    closed,
                });
                i = if closed {
                    body_end + "</thinking>".len()
                } else {
                    text.len()
                };
                prose_start = i;
                continue;
            }

            // Fenced code block: ``` or ~~~ at line start.
            if let Some((fence_len, fence_char)) = match_fence(bytes, i) {
                let lang_start = i + fence_len;
                let lang_end = bytes[lang_start..]
                    .iter()
                    .position(|&b| b == b'\n')
                    .map(|p| lang_start + p)
                    .unwrap_or(bytes.len());
                let body_start = (lang_end + 1).min(bytes.len());

                // Find matching closing fence at line start with same char and
                // at-least-as-long run.
                let mut j = body_start;
                let mut close_start = None;
                let mut close_end = None;
                while j < bytes.len() {
                    if at_line_start(text, j) {
                        let run = run_len(bytes, j, fence_char);
                        if run >= fence_len {
                            // Closing fences must not contain info text on the
                            // same line per CommonMark, but be lenient.
                            close_start = Some(j);
                            close_end = Some(
                                bytes[j..]
                                    .iter()
                                    .position(|&b| b == b'\n')
                                    .map(|p| j + p + 1)
                                    .unwrap_or(bytes.len()),
                            );
                            break;
                        }
                    }
                    j += 1;
                }

                flush_prose(text, prose_start, i, &mut out);
                let body_end_excl = close_start.unwrap_or(bytes.len());
                // Trim the trailing newline of the body so render doesn't double-space.
                let body_end_trimmed = if body_end_excl > body_start
                    && bytes[body_end_excl - 1] == b'\n'
                {
                    body_end_excl - 1
                } else {
                    body_end_excl
                };
                out.push(Segment::Code {
                    lang: text[lang_start..lang_end].trim(),
                    body: &text[body_start..body_end_trimmed],
                });
                i = close_end.unwrap_or(bytes.len());
                prose_start = i;
                continue;
            }
        }

        // Advance to next byte. Fast-skip non-special bytes.
        i += 1;
    }

    flush_prose(text, prose_start, text.len(), &mut out);
    out
}

fn flush_prose<'a>(text: &'a str, start: usize, end: usize, out: &mut Vec<Segment<'a>>) {
    if start < end {
        out.push(Segment::Markdown(&text[start..end]));
    }
}

/// Whether byte index `i` is at the start of a line (start of input, or the
/// previous byte is `\n`).
fn at_line_start(text: &str, i: usize) -> bool {
    i == 0 || text.as_bytes().get(i - 1) == Some(&b'\n')
}

/// If `text[i..]` starts with `tag`, return the byte index right after `tag`.
fn match_open(text: &str, i: usize, tag: &str) -> Option<usize> {
    if text[i..].starts_with(tag) {
        Some(i + tag.len())
    } else {
        None
    }
}

/// Find the next occurrence of `tag` at or after `from`. Returns the byte
/// index where `tag` starts and whether it was found at all.
fn find_close(text: &str, from: usize, tag: &str) -> (usize, bool) {
    text[from..]
        .find(tag)
        .map(|p| (from + p, true))
        .unwrap_or((text.len(), false))
}

/// If `bytes[i..]` is a CommonMark fence opener (``` or ~~~ of length >= 3),
/// return `(fence_len, fence_byte)`.
fn match_fence(bytes: &[u8], i: usize) -> Option<(usize, u8)> {
    let c = *bytes.get(i)?;
    if c != b'`' && c != b'~' {
        return None;
    }
    let len = run_len(bytes, i, c);
    if len >= 3 { Some((len, c)) } else { None }
}

fn run_len(bytes: &[u8], i: usize, c: u8) -> usize {
    let mut n = 0;
    while bytes.get(i + n) == Some(&c) {
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_prose() {
        let s = "hello world\nsecond line";
        assert_eq!(segments(s), vec![Segment::Markdown(s)]);
    }

    #[test]
    fn fenced_code_block_with_lang() {
        let s = "intro\n```rust\nfn main() {}\n```\noutro";
        let out = segments(s);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Segment::Markdown("intro\n"));
        assert_eq!(
            out[1],
            Segment::Code {
                lang: "rust",
                body: "fn main() {}"
            }
        );
        assert_eq!(out[2], Segment::Markdown("outro"));
    }

    #[test]
    fn fenced_code_no_lang() {
        let s = "```\nplain\n```";
        let out = segments(s);
        assert_eq!(
            out,
            vec![Segment::Code {
                lang: "",
                body: "plain"
            }]
        );
    }

    #[test]
    fn unclosed_fence_runs_to_end() {
        let s = "before\n```rust\nfn x()";
        let out = segments(s);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], Segment::Markdown(_)));
        assert!(matches!(out[1], Segment::Code { lang: "rust", .. }));
    }

    #[test]
    fn think_block() {
        let s = "<think>\nLet me reason\n</think>\nAnswer.";
        let out = segments(s);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            Segment::Reasoning {
                body: "\nLet me reason\n",
                closed: true
            }
        );
        assert_eq!(out[1], Segment::Markdown("\nAnswer."));
    }

    #[test]
    fn unclosed_think_block_during_streaming() {
        let s = "<think>\npartial reasoning";
        let out = segments(s);
        assert_eq!(
            out,
            vec![Segment::Reasoning {
                body: "\npartial reasoning",
                closed: false
            }]
        );
    }

    #[test]
    fn think_only_at_line_start() {
        // An inline `<think>` shouldn't be treated as a reasoning block
        let s = "see <think> tag here";
        let out = segments(s);
        assert_eq!(out, vec![Segment::Markdown(s)]);
    }

    #[test]
    fn tilde_fences() {
        let s = "~~~py\nx = 1\n~~~";
        let out = segments(s);
        assert_eq!(
            out,
            vec![Segment::Code {
                lang: "py",
                body: "x = 1"
            }]
        );
    }

    #[test]
    fn nested_backticks_in_code_block() {
        // 4 backticks open, 3 inside, 4 close
        let s = "````\n```inner```\n````";
        let out = segments(s);
        assert_eq!(out.len(), 1);
        match &out[0] {
            Segment::Code { lang, body } => {
                assert_eq!(*lang, "");
                assert_eq!(*body, "```inner```");
            }
            _ => panic!("expected code"),
        }
    }
}
