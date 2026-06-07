// Mirrors the Rust DTOs in src-tauri/src/commands.rs

export type RuntimeKind = "vllm" | "omlx" | "llamacpp" | "llamaswap" | "openai";
export type GpuKind = "none" | "macmon" | "agent";

export interface Endpoint {
  url: string;
  api_key?: string | null;
  runtime?: RuntimeKind;
  prometheus_url?: string | null;
  gpu?: GpuKind;
  agent_url?: string | null;
}

export interface Hotkeys {
  new_chat: string;
  focus_input: string;
  find: string;
  settings: string;
  toggle_artifacts: string;
  stop: string;
}

export interface McpServer {
  name: string;
  transport: string;
  command?: string | null;
  args?: string[];
  env?: Record<string, string>;
  url?: string | null;
  headers?: Record<string, string>;
  enabled?: boolean;
  auto_approve?: boolean;
}

export interface McpStatus {
  name: string;
  transport: string;
  connected: boolean;
  tool_count: number;
  auto_approve: boolean;
  error: string | null;
}

export interface Config {
  font_size: number;
  mono_font_size: number;
  ui_scale: number;
  dark_mode: boolean;
  theme: string;
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
  hotkeys: Hotkeys;
  mcp_servers?: McpServer[];
  /** Days of usage history to keep (0 = forever). */
  usage_retention_days: number;
}

export interface ConversationDto {
  id: number;
  title: string;
  pinned: boolean;
  project_id: number | null;
  runtime: string | null;
}

export interface ProjectDto {
  id: number;
  name: string;
  pinned: boolean;
}

export interface UsageByModel {
  model: string;
  endpoint: string;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost: number;
  avg_ttft_ms: number | null;
  avg_decode_tok_s: number | null;
}

export interface UsageDaily {
  date: string;
  total_tokens: number;
}

export interface BenchParams {
  endpoint: string;
  model: string;
  prompt: string;
  max_tokens: number;
  concurrency: number;
  total_requests: number;
}

export interface BenchAgg {
  avg: number | null;
  p50: number | null;
  p95: number | null;
}

export interface BenchResult {
  requests: number;
  ok: number;
  errors: number;
  wall_ms: number;
  ttft_ms: BenchAgg;
  decode_tok_s: BenchAgg;
  agg_decode_tok_s: number;
  total_completion_tokens: number;
  ttft_series: number[];
  decode_series: number[];
}

export interface UsageStats {
  total_requests: number;
  ok_requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  total_cost: number;
  ttft_p50_ms: number | null;
  ttft_p95_ms: number | null;
  decode_p50_tok_s: number | null;
  decode_p95_tok_s: number | null;
  by_model: UsageByModel[];
  daily: UsageDaily[];
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
  created_at: number | null;
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
  draft: string | null;
}

// Shared generation knobs (flattened into the request payloads server-side).
export interface GenParams {
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
}

export interface SendParams extends GenParams {
  conversation_id: number | null;
  user_text: string;
  images: string[];
}

export interface SiblingInfo {
  index: number;
  total: number;
  ids: number[];
}

export interface GpuDevice {
  name: string;
  util_pct: number;
  mem_used_gb: number;
  mem_total_gb: number;
  temp_c: number;
  power_w: number;
  power_limit_w: number;
}

export interface GpuStats {
  source: string;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  power_w: number | null;
  power_limit_w: number | null;
  temp_c: number | null;
  util_pct: number | null;
  devices: GpuDevice[];
}

export interface ServerStats {
  decode_tok_s: number | null;
  prefill_tok_s: number | null;
  ttft_ms: number | null;
  requests_running: number | null;
  requests_waiting: number | null;
  kv_cache_pct: number | null;
  prompt_tokens_total: number | null;
  generation_tokens_total: number | null;
}

export interface MetricsSnapshot {
  endpoint: string;
  runtime: string;
  gpu: GpuStats | null;
  server: ServerStats | null;
}

export interface AgentCommand {
  name: string;
  description: string;
  body: string;
}

export interface Skill {
  name: string;
  description: string;
  body: string;
}

export interface AgentsDto {
  commands: AgentCommand[];
  skills: Skill[];
}

export interface PresetDto {
  id: number;
  name: string;
  endpoint: string | null;
  model: string | null;
  system_prompt: string | null;
  temperature: number | null;
  max_tokens: number | null;
  use_max_tokens: boolean;
  top_p: number | null;
  frequency_penalty: number | null;
  presence_penalty: number | null;
  stop_sequences: string[];
}

// Streaming events (serde tag = "type", snake_case)
export type ChatEvent =
  | { type: "started"; conversation_id: number }
  | { type: "turn_start" }
  | { type: "token"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "tool_call"; id: string; name: string; arguments: string }
  | { type: "tool_approval"; id: string; name: string; arguments: string }
  | {
      type: "tool_result";
      id: string;
      name: string;
      result: string;
      is_error: boolean;
    }
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

export interface PendingApproval {
  id: string;
  name: string;
  arguments: string;
}
