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

function clip(s: string, n = 48): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}

/** A human title for the artifact, derived from its content where possible
 *  (an HTML <title>/<h1>, an SVG <title>, the first Markdown heading, a mermaid
 *  diagram type), falling back to the kind/lang name. Uniqueness across a set
 *  of same-named artifacts is handled separately in collectArtifacts. */
function title(kind: ArtifactKind, lang: string, code = ""): string {
  switch (kind) {
    case "html": {
      const t = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(code);
      if (t?.[1]?.trim()) return clip(t[1]);
      const h1 = /<h1[^>]*>([\s\S]*?)<\/h1>/i.exec(code);
      const h1Text = h1?.[1]?.replace(/<[^>]+>/g, "");
      if (h1Text?.trim()) return clip(h1Text);
      return "HTML";
    }
    case "svg": {
      const t = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(code);
      if (t?.[1]?.trim()) return clip(t[1]);
      return "SVG";
    }
    case "mermaid": {
      const first = code.trim().split("\n")[0]?.trim().toLowerCase() ?? "";
      const types: [RegExp, string][] = [
        [/^sequencediagram/, "Sequence diagram"],
        [/^(graph|flowchart)/, "Flowchart"],
        [/^classdiagram/, "Class diagram"],
        [/^statediagram/, "State diagram"],
        [/^erdiagram/, "ER diagram"],
        [/^gantt/, "Gantt chart"],
        [/^pie/, "Pie chart"],
        [/^journey/, "Journey"],
        [/^mindmap/, "Mindmap"],
      ];
      return types.find(([re]) => re.test(first))?.[1] ?? "Diagram";
    }
    case "markdown": {
      const h = /^#{1,6}\s+(.+?)\s*#*$/m.exec(code);
      if (h?.[1]?.trim()) return clip(h[1]);
      return "Markdown";
    }
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
      title: title(kind, lang, code),
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
  // Disambiguate same-titled artifacts ("HTML" → "HTML 1", "HTML 2") so
  // multiple of the same kind aren't all labelled identically. Unique titles
  // (e.g. distinct content-derived names) are left untouched.
  const totals = new Map<string, number>();
  out.forEach((a) => totals.set(a.title, (totals.get(a.title) ?? 0) + 1));
  const seen = new Map<string, number>();
  out.forEach((a) => {
    if ((totals.get(a.title) ?? 0) > 1) {
      const n = (seen.get(a.title) ?? 0) + 1;
      seen.set(a.title, n);
      a.title = `${a.title} ${n}`;
    }
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
  return { id, kind, lang: lang.trim(), code, title: title(kind, lang, code) };
}
