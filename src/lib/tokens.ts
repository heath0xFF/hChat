import { encode } from "gpt-tokenizer";

/** Approximate token count (cl100k_base). Good enough for a live composer
 *  counter; the authoritative count comes back in the usage event. */
export function countTokens(text: string): number {
  try {
    return encode(text).length;
  } catch {
    return Math.ceil(text.length / 4);
  }
}

/** Compact human-readable token count, e.g. `1.2K`, `3.40M`. */
export function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}
