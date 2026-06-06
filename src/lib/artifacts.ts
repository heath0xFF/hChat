import { parseThink } from "./segments";
import type { ChatMessage } from "../components/MessageItem";

export type ArtifactKind = "html" | "svg" | "markdown" | "mermaid" | "code";

export interface Artifact {
  id: string;
  kind: ArtifactKind;
  lang: string;
  code: string;
  title: string;
}

const FENCE = /```([^\n`]*)\n([\s\S]*?)```/g;

function classify(lang: string, code: string): ArtifactKind {
  const l = lang.trim().toLowerCase();
  const head = code.trimStart().toLowerCase();
  if (l === "html" || l === "htm" || head.startsWith("<!doctype html") || head.startsWith("<html")) {
    return "html";
  }
  if (l === "svg" || head.startsWith("<svg")) return "svg";
  if (l === "mermaid") return "mermaid";
  if (l === "markdown" || l === "md") return "markdown";
  return "code";
}

function title(kind: ArtifactKind, lang: string): string {
  switch (kind) {
    case "html":
      return "HTML";
    case "svg":
      return "SVG";
    case "mermaid":
      return "Diagram";
    case "markdown":
      return "Markdown";
    default:
      return lang.trim() || "code";
  }
}

/** Extract artifacts from one assistant message body. `previewable` kinds
 *  (html/svg/markdown) always count; plain code only if it's a few lines (so
 *  tiny inline snippets don't clutter the list). */
export function parseArtifacts(text: string, idPrefix: string): Artifact[] {
  const body = parseThink(text).body;
  const out: Artifact[] = [];
  let m: RegExpExecArray | null;
  let i = 0;
  FENCE.lastIndex = 0;
  while ((m = FENCE.exec(body)) !== null) {
    const lang = m[1] ?? "";
    const code = (m[2] ?? "").replace(/\n$/, "");
    const kind = classify(lang, code);
    const substantial = kind !== "code" || code.split("\n").length >= 3;
    if (!substantial) {
      i++;
      continue;
    }
    out.push({
      id: `${idPrefix}-${i}`,
      kind,
      lang: lang.trim(),
      code,
      title: title(kind, lang),
    });
    i++;
  }
  return out;
}

export function collectArtifacts(messages: ChatMessage[]): Artifact[] {
  const out: Artifact[] = [];
  messages.forEach((msg, idx) => {
    if (msg.role !== "assistant") return;
    out.push(...parseArtifacts(msg.text, `${msg.id ?? `live${idx}`}`));
  });
  return out;
}

export function isPreviewable(a: Artifact): boolean {
  return (
    a.kind === "html" ||
    a.kind === "svg" ||
    a.kind === "markdown" ||
    a.kind === "mermaid"
  );
}

/** Build a one-off artifact from a code block opened directly from chat. */
export function makeArtifact(code: string, lang: string, id = "adhoc"): Artifact {
  const kind = classify(lang, code);
  return { id, kind, lang: lang.trim(), code, title: title(kind, lang) };
}
