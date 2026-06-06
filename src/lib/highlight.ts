import { createHighlighter, type Highlighter } from "shiki";

const THEME = "github-dark-default";
const PRELOAD = ["rust", "typescript", "javascript", "python", "bash", "json"];

let hlPromise: Promise<Highlighter> | null = null;
const loaded = new Set<string>(PRELOAD);

function get(): Promise<Highlighter> {
  if (!hlPromise) {
    hlPromise = createHighlighter({ themes: [THEME], langs: PRELOAD });
  }
  return hlPromise;
}

const ALIASES: Record<string, string> = {
  ts: "typescript",
  js: "javascript",
  py: "python",
  sh: "bash",
  shell: "bash",
  rs: "rust",
  yml: "yaml",
};

/** Highlight `code` to HTML. Loads the language on demand; falls back to an
 *  escaped <pre> if the language is unknown or shiki fails. */
export async function highlight(code: string, lang: string): Promise<string> {
  const resolved = ALIASES[lang] || lang || "text";
  try {
    const hl = await get();
    if (resolved !== "text" && !loaded.has(resolved)) {
      try {
        await hl.loadLanguage(resolved as never);
        loaded.add(resolved);
      } catch {
        return escapePre(code);
      }
    }
    return hl.codeToHtml(code, {
      lang: loaded.has(resolved) ? resolved : "text",
      theme: THEME,
    });
  } catch {
    return escapePre(code);
  }
}

function escapePre(code: string): string {
  const esc = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return `<pre class="shiki"><code>${esc}</code></pre>`;
}
