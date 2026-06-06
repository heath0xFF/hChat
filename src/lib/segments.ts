// Split assistant text into reasoning (`<think>…</think>`) and body. Mirrors
// the storage convention where reasoning is persisted inline as a leading
// think block. Handles an unclosed `<think>` (reasoning still streaming).

export interface ParsedMessage {
  reasoning: string | null;
  reasoningOpen: boolean; // true while the think block hasn't closed yet
  body: string;
}

export function parseThink(text: string): ParsedMessage {
  const open = text.indexOf("<think>");
  if (open === -1) {
    return { reasoning: null, reasoningOpen: false, body: text };
  }
  const before = text.slice(0, open);
  const rest = text.slice(open + "<think>".length);
  const close = rest.indexOf("</think>");
  if (close === -1) {
    return {
      reasoning: rest,
      reasoningOpen: true,
      body: before,
    };
  }
  const reasoning = rest.slice(0, close);
  const after = rest.slice(close + "</think>".length);
  return {
    reasoning,
    reasoningOpen: false,
    body: (before + after).replace(/^\n/, ""),
  };
}
