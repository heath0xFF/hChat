import type { Hotkeys } from "../types";

/** Does this keyboard event match the combo string (e.g. "mod+n")?
 *  `mod` matches Cmd (macOS) or Ctrl. */
export function matchCombo(e: KeyboardEvent, combo: string): boolean {
  if (!combo) return false;
  const parts = combo
    .toLowerCase()
    .split("+")
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length === 0) return false;
  const key = parts[parts.length - 1];
  const needMod = parts.includes("mod");
  const needShift = parts.includes("shift");
  const needAlt = parts.includes("alt");
  const mod = e.metaKey || e.ctrlKey;
  if (needMod !== mod) return false;
  if (needShift !== e.shiftKey) return false;
  if (needAlt !== e.altKey) return false;
  const k = e.key.toLowerCase() === " " ? "space" : e.key.toLowerCase();
  return k === key;
}

/** Build a combo string from a key event (for the rebind UI). Returns null
 *  while only modifier keys are held. */
export function comboFromEvent(e: KeyboardEvent): string | null {
  const k = e.key.toLowerCase();
  if (["control", "meta", "shift", "alt"].includes(k)) return null;
  const parts: string[] = [];
  if (e.metaKey || e.ctrlKey) parts.push("mod");
  if (e.shiftKey) parts.push("shift");
  if (e.altKey) parts.push("alt");
  parts.push(k === " " ? "space" : k);
  return parts.join("+");
}

export function comboIsBare(combo: string): boolean {
  const l = combo.toLowerCase();
  return !l.includes("mod") && !l.includes("alt");
}

export function isTypingTarget(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null;
  if (!el) return false;
  return (
    el.tagName === "INPUT" ||
    el.tagName === "TEXTAREA" ||
    el.isContentEditable
  );
}

export const HOTKEY_ACTIONS: { key: keyof Hotkeys; label: string }[] = [
  { key: "new_chat", label: "New chat" },
  { key: "focus_input", label: "Focus message input" },
  { key: "find", label: "Find in conversation" },
  { key: "settings", label: "Open settings" },
  { key: "toggle_artifacts", label: "Toggle artifacts panel" },
  { key: "stop", label: "Stop generation" },
];
