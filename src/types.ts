// Mirrors the Rust DTOs in src-tauri/src/commands.rs

export interface Endpoint {
  url: string;
  api_key?: string | null;
}

export interface Config {
  font_size: number;
  mono_font_size: number;
  ui_scale: number;
  dark_mode: boolean;
  default_endpoint: string;
  system_prompt: string;
  temperature: number;
  max_tokens: number;
  use_max_tokens: boolean;
  saved_endpoints: Endpoint[];
  font_family: string;
  mono_font_family: string;
  top_p: number | null;
  frequency_penalty: number | null;
  presence_penalty: number | null;
  stop_sequences: string[];
}

export interface ConversationDto {
  id: number;
  title: string;
  pinned: boolean;
}

export interface ToolCallDto {
  id: string;
  name: string;
  arguments: string;
}

export interface MessageDto {
  id: number | null;
  role: "system" | "user" | "assistant" | "tool";
  text: string;
  images: string[];
  tool_calls: ToolCallDto[] | null;
  tool_call_id: string | null;
}

export interface SettingsDto {
  model: string | null;
  endpoint: string;
  system_prompt: string;
  temperature: number;
  max_tokens: number;
  use_max_tokens: boolean;
  top_p: number | null;
  frequency_penalty: number | null;
  presence_penalty: number | null;
  stop_sequences: string[];
  working_dir: string | null;
}

export interface ConversationData {
  messages: MessageDto[];
  settings: SettingsDto;
}

export interface SendParams {
  conversation_id: number | null;
  endpoint: string;
  model: string;
  system_prompt: string;
  temperature: number | null;
  max_tokens: number | null;
  use_max_tokens: boolean;
  top_p: number | null;
  frequency_penalty: number | null;
  presence_penalty: number | null;
  stop_sequences: string[];
  user_text: string;
}

// Streaming events (serde tag = "type", snake_case)
export type ChatEvent =
  | { type: "started"; conversation_id: number }
  | { type: "token"; text: string }
  | { type: "reasoning"; text: string }
  | {
      type: "usage";
      prompt_tokens: number | null;
      completion_tokens: number | null;
      total_tokens: number | null;
      cost: number | null;
    }
  | {
      type: "request_metrics";
      ttft_ms: number | null;
      decode_tok_s: number | null;
      prefill_tok_s: number | null;
      duration_ms: number;
    }
  | { type: "done"; message_id: number | null }
  | { type: "error"; message: string };
