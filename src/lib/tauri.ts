import { invoke, Channel } from "@tauri-apps/api/core";
import type {
  Config,
  ConversationData,
  ConversationDto,
  ChatEvent,
  GenParams,
  MessageDto,
  PresetDto,
  SendParams,
  SiblingInfo,
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
  renameConversation: (id: number, title: string) =>
    invoke<void>("rename_conversation", { id, title }),
  setPinned: (id: number, pinned: boolean) =>
    invoke<void>("set_pinned", { id, pinned }),
  searchConversations: (query: string) =>
    invoke<[number, string, string][]>("search_conversations", { query }),
  exportConversation: (id: number) =>
    invoke<string>("export_conversation", { id }),

  cancelStream: () => invoke<void>("cancel_stream"),
  resolveTool: (callId: string, approved: boolean) =>
    invoke<void>("resolve_tool", { callId, approved }),

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
};
