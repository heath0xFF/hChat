import { invoke, Channel } from "@tauri-apps/api/core";
import type {
  Config,
  ConversationData,
  ConversationDto,
  AgentsDto,
  ChatEvent,
  GenParams,
  McpStatus,
  MessageDto,
  PresetDto,
  ProjectDto,
  SendParams,
  SiblingInfo,
  UsageStats,
  BenchParams,
  BenchResult,
} from "../types";

export const api = {
  getConfig: () => invoke<Config>("get_config"),
  saveConfig: (config: Config) => invoke<void>("save_config", { config }),
  fetchModels: (endpoint: string) =>
    invoke<string[]>("fetch_models", { endpoint }),

  listConversations: () => invoke<ConversationDto[]>("list_conversations"),
  loadConversation: (id: number) =>
    invoke<ConversationData>("load_conversation", { id }),
  deleteConversation: (id: number) =>
    invoke<void>("delete_conversation", { id }),
  deleteAllConversations: () => invoke<void>("delete_all_conversations"),
  renameConversation: (id: number, title: string) =>
    invoke<void>("rename_conversation", { id, title }),
  setPinned: (id: number, pinned: boolean) =>
    invoke<void>("set_pinned", { id, pinned }),

  usageStats: () => invoke<UsageStats>("usage_stats"),
  clearUsage: () => invoke<void>("clear_usage"),
  runBenchmark: (params: BenchParams) =>
    invoke<BenchResult>("run_benchmark", { params }),
  listProjects: () => invoke<ProjectDto[]>("list_projects"),
  createProject: (name: string) => invoke<number>("create_project", { name }),
  renameProject: (id: number, name: string) =>
    invoke<void>("rename_project", { id, name }),
  deleteProject: (id: number) => invoke<void>("delete_project", { id }),
  setProjectPinned: (id: number, pinned: boolean) =>
    invoke<void>("set_project_pinned", { id, pinned }),
  setConversationProject: (conversationId: number, projectId: number | null) =>
    invoke<void>("set_conversation_project", { conversationId, projectId }),
  searchConversations: (query: string) =>
    invoke<[number, string, string][]>("search_conversations", { query }),
  exportConversation: (id: number) =>
    invoke<string>("export_conversation", { id }),
  saveDraft: (id: number, text: string) =>
    invoke<void>("save_draft", { id, text }),

  cancelStream: (conversationId: number) =>
    invoke<void>("cancel_stream", { conversationId }),
  resolveTool: (callId: string, decision: "approve" | "approve_all" | "deny") =>
    invoke<void>("resolve_tool", { callId, decision }),
  exportConversationFile: (id: number) =>
    invoke<string>("export_conversation_file", { id }),

  /** Start a streaming generation. Returns the (possibly newly created)
   *  conversation id once the stream finishes. `onEvent` fires per chunk. */
  sendMessage: (params: SendParams, onEvent: (ev: ChatEvent) => void) => {
    const channel = new Channel<ChatEvent>();
    channel.onmessage = onEvent;
    return invoke<number>("send_message", { params, onEvent: channel });
  },

  regenerate: (
    params: GenParams & { conversation_id: number },
    onEvent: (ev: ChatEvent) => void,
  ) => {
    const channel = new Channel<ChatEvent>();
    channel.onmessage = onEvent;
    return invoke<number>("regenerate", { params, onEvent: channel });
  },

  editMessage: (
    params: GenParams & {
      conversation_id: number;
      message_id: number;
      new_text: string;
    },
    onEvent: (ev: ChatEvent) => void,
  ) => {
    const channel = new Channel<ChatEvent>();
    channel.onmessage = onEvent;
    return invoke<number>("edit_message", { params, onEvent: channel });
  },

  messageSiblings: (messageId: number) =>
    invoke<SiblingInfo>("message_siblings", { messageId }),
  walkFrom: (startId: number) =>
    invoke<MessageDto[]>("walk_from", { startId }),

  listPresets: () => invoke<PresetDto[]>("list_presets"),
  createPreset: (name: string, gp: GenParams) =>
    invoke<number>("create_preset", { name, gp }),
  deletePreset: (id: number) => invoke<void>("delete_preset", { id }),

  setMetricsTarget: (endpoint: string) =>
    invoke<void>("set_metrics_target", { endpoint }),

  listAgents: (workingDir: string | null) =>
    invoke<AgentsDto>("list_agents", { workingDir }),

  listMcpServers: () => invoke<McpStatus[]>("list_mcp_servers"),
  reconnectMcp: () => invoke<void>("reconnect_mcp"),
};
