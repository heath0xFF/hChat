// Mirrors src-tauri/src/core/slash.rs — intercept `/verb …` in the composer.

export type SlashCmd =
  | { type: "model"; arg: string }
  | { type: "temp"; value: number }
  | { type: "system"; text: string }
  | { type: "clear" }
  | { type: "copy" }
  | { type: "help" };

export type SlashResult =
  | { kind: "command"; cmd: SlashCmd }
  | { kind: "unknown"; verb: string }
  | { kind: "badargs"; reason: string }
  | { kind: "none" };

export const SLASH_HELP = [
  "Slash commands:",
  "/model <name>  — switch model (substring match)",
  "/temp <0..2>   — set temperature",
  "/system <text> — set system prompt (empty clears it)",
  "/clear         — start a new conversation",
  "/copy          — copy last reply to clipboard",
  "/help          — show this message",
].join("\n");

export function parseSlash(input: string): SlashResult {
  const trimmed = input.replace(/^\s+/, "");
  if (!trimmed.startsWith("/")) return { kind: "none" };
  const body = trimmed.slice(1).replace(/\s+$/, "");
  if (body.length === 0) return { kind: "none" };

  const wsIdx = body.search(/\s/);
  const verb = (wsIdx === -1 ? body : body.slice(0, wsIdx)).toLowerCase();
  const rest = wsIdx === -1 ? "" : body.slice(wsIdx + 1).trim();

  switch (verb) {
    case "model":
    case "m":
      return rest
        ? { kind: "command", cmd: { type: "model", arg: rest } }
        : { kind: "badargs", reason: "usage: /model <name-or-substring>" };
    case "temp":
    case "temperature":
    case "t": {
      const v = Number(rest);
      return rest !== "" && isFinite(v)
        ? { kind: "command", cmd: { type: "temp", value: v } }
        : { kind: "badargs", reason: "usage: /temp <0.0..=2.0>" };
    }
    case "system":
    case "sys":
      return { kind: "command", cmd: { type: "system", text: rest } };
    case "clear":
    case "new":
      return { kind: "command", cmd: { type: "clear" } };
    case "copy":
      return { kind: "command", cmd: { type: "copy" } };
    case "help":
    case "?":
    case "h":
      return { kind: "command", cmd: { type: "help" } };
    default:
      return { kind: "unknown", verb };
  }
}
