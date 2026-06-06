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
