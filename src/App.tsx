import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api } from "./lib/tauri";
import type {
  ChatEvent,
  Config,
  ConversationDto,
  GenParams,
  MessageDto,
  MetricsSnapshot,
  PendingApproval,
  PresetDto,
  SettingsDto,
  SiblingInfo,
} from "./types";
import { Sidebar, type View } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";
import { StatusView, type LiveMetrics } from "./components/StatusView";
import { SettingsModal } from "./components/SettingsModal";
import { ArtifactPanel } from "./components/ArtifactPanel";
import type { ChatMessage } from "./components/MessageItem";
import {
  collectArtifacts,
  isPreviewable,
  makeArtifact,
  type Artifact,
} from "./lib/artifacts";

const EMPTY_METRICS: LiveMetrics = {
  decode: null,
  ttft: null,
  prefill: null,
  peakDecode: 0,
  peakTtft: 0,
  peakPrefill: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  durationMs: 0,
  cost: null,
  activeRequests: 0,
  throughputHistory: [],
  ttftHistory: [],
};

function defaultSettings(config: Config): SettingsDto {
  return {
    model: null,
    endpoint: config.default_endpoint,
    system_prompt: config.system_prompt,
    temperature: config.temperature,
    max_tokens: config.max_tokens,
    use_max_tokens: config.use_max_tokens,
    top_p: config.top_p,
    frequency_penalty: config.frequency_penalty,
    presence_penalty: config.presence_penalty,
    stop_sequences: config.stop_sequences,
    working_dir: null,
  };
}

function combine(reasoning: string, content: string): string {
  if (!reasoning) return content;
  if (!content) return `<think>${reasoning}`;
  return `<think>${reasoning}</think>\n${content}`;
}

function toChatMessage(m: MessageDto): ChatMessage {
  return {
    id: m.id,
    role: m.role,
    text: m.text,
    images: m.images,
    toolCalls: m.tool_calls
      ? m.tool_calls.map((tc) => ({
          id: tc.id,
          name: tc.name,
          arguments: tc.arguments,
        }))
      : undefined,
  };
}

export function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [conversations, setConversations] = useState<ConversationDto[]>([]);
  const [view, setView] = useState<View>("chat");
  const [activeConvId, setActiveConvId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [settings, setSettings] = useState<SettingsDto | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [metrics, setMetrics] = useState<LiveMetrics>(EMPTY_METRICS);
  const [searchQuery, setSearchQuery] = useState("");
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(
    null,
  );
  const [siblingMap, setSiblingMap] = useState<Record<number, SiblingInfo>>({});
  const [presets, setPresets] = useState<PresetDto[]>([]);
  const [snapshot, setSnapshot] = useState<MetricsSnapshot | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [currentArtifact, setCurrentArtifact] = useState<Artifact | null>(null);
  const [artifactOpen, setArtifactOpen] = useState(false);

  const reasoningBuf = useRef("");
  const contentBuf = useRef("");
  const convIdRef = useRef<number | null>(null);

  const refreshConversations = useCallback(async () => {
    setConversations(await api.listConversations());
  }, []);

  const loadModels = useCallback(async (endpoint: string) => {
    try {
      const m = await api.fetchModels(endpoint);
      setModels(m);
      return m;
    } catch (e) {
      setModels([]);
      setError(`Models: ${e}`);
      return [];
    }
  }, []);

  // initial load
  useEffect(() => {
    (async () => {
      const cfg = await api.getConfig();
      setConfig(cfg);
      const s = defaultSettings(cfg);
      setSettings(s);
      await refreshConversations();
      setPresets(await api.listPresets());
      const m = await loadModels(s.endpoint);
      if (m.length) setSettings((prev) => (prev ? { ...prev, model: m[0] } : prev));
    })();
  }, [refreshConversations, loadModels]);

  // Subscribe to the backend metrics poller. Each snapshot updates the GPU/
  // server view and feeds the live charts (so they move during idle polling,
  // not only when we send a request).
  useEffect(() => {
    const un = listen<MetricsSnapshot>("metrics", (e) => {
      const snap = e.payload;
      setSnapshot(snap);
      const decode = snap.server?.decode_tok_s ?? null;
      const ttft = snap.server?.ttft_ms ?? null;
      if (decode != null || ttft != null) {
        setMetrics((m) => ({
          ...m,
          throughputHistory:
            decode != null
              ? [...m.throughputHistory, decode].slice(-120)
              : m.throughputHistory,
          ttftHistory:
            ttft != null ? [...m.ttftHistory, ttft].slice(-120) : m.ttftHistory,
        }));
      }
    });
    return () => {
      un.then((f) => f());
    };
  }, []);

  // Point the metrics poller at the active conversation's endpoint.
  useEffect(() => {
    if (settings?.endpoint) void api.setMetricsTarget(settings.endpoint);
  }, [settings?.endpoint]);

  // Re-collect artifacts whenever the visible message path changes.
  useEffect(() => {
    setArtifacts(collectArtifacts(messages));
  }, [messages]);

  const openArtifact = (code: string, lang: string) => {
    const match = artifacts.find((a) => a.code === code) ?? makeArtifact(code, lang);
    setCurrentArtifact(match);
    setArtifactOpen(true);
  };

  const toggleArtifacts = () => {
    if (artifactOpen) {
      setArtifactOpen(false);
      return;
    }
    if (!currentArtifact && artifacts.length > 0) {
      setCurrentArtifact(artifacts[artifacts.length - 1]);
    }
    if (artifacts.length > 0 || currentArtifact) setArtifactOpen(true);
  };

  const newChat = () => {
    if (!config) return;
    setActiveConvId(null);
    setMessages([]);
    setView("chat");
    const s = defaultSettings(config);
    s.model = models[0] ?? null;
    setSettings((prev) => ({ ...s, endpoint: prev?.endpoint ?? s.endpoint, model: prev?.model ?? s.model }));
  };

  const selectConv = async (id: number) => {
    const data = await api.loadConversation(id);
    setActiveConvId(id);
    convIdRef.current = id;
    setMessages(data.messages.map(toChatMessage));
    void buildSiblingMap(data.messages);
    setSettings(data.settings);
    setView("chat");
    const m = await loadModels(data.settings.endpoint);
    if (data.settings.model === null && m.length) {
      setSettings((prev) => (prev ? { ...prev, model: m[0] } : prev));
    }
  };

  const changeModel = (model: string) =>
    setSettings((prev) => (prev ? { ...prev, model } : prev));

  const changeEndpoint = async (endpoint: string) => {
    setSettings((prev) => (prev ? { ...prev, endpoint } : prev));
    const m = await loadModels(endpoint);
    setSettings((prev) => (prev ? { ...prev, model: m[0] ?? prev.model } : prev));
  };

  // Mutate the most recent assistant message (the one currently streaming).
  const patchLastAssistant = (fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => {
      const next = prev.slice();
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") {
          next[i] = fn(next[i]);
          break;
        }
      }
      return next;
    });
  };

  const updateLastAssistant = () => {
    const text = combine(reasoningBuf.current, contentBuf.current);
    patchLastAssistant((m) => ({ ...m, text }));
  };

  const genParamsFrom = (s: SettingsDto): GenParams => ({
    endpoint: s.endpoint,
    model: s.model ?? "",
    system_prompt: s.system_prompt,
    temperature: s.temperature,
    max_tokens: s.max_tokens,
    use_max_tokens: s.use_max_tokens,
    top_p: s.top_p,
    frequency_penalty: s.frequency_penalty,
    presence_penalty: s.presence_penalty,
    stop_sequences: s.stop_sequences,
  });

  // Shared streaming event handler used by send / regenerate / edit.
  const handleEvent = (ev: ChatEvent) => {
        switch (ev.type) {
          case "started":
            convIdRef.current = ev.conversation_id;
            if (activeConvId === null) setActiveConvId(ev.conversation_id);
            break;
          case "turn_start":
            // Close out the previous assistant turn, reset buffers, and open a
            // fresh streaming placeholder for this turn.
            reasoningBuf.current = "";
            contentBuf.current = "";
            setMessages((prev) => {
              const next = prev.map((m) =>
                m.role === "assistant" && m.streaming
                  ? { ...m, streaming: false }
                  : m,
              );
              return [
                ...next,
                { id: null, role: "assistant", text: "", streaming: true },
              ];
            });
            break;
          case "token":
            contentBuf.current += ev.text;
            updateLastAssistant();
            break;
          case "reasoning":
            reasoningBuf.current += ev.text;
            updateLastAssistant();
            break;
          case "tool_call":
            patchLastAssistant((m) => ({
              ...m,
              toolCalls: [
                ...(m.toolCalls ?? []),
                { id: ev.id, name: ev.name, arguments: ev.arguments },
              ],
            }));
            break;
          case "tool_approval":
            setPendingApproval({
              id: ev.id,
              name: ev.name,
              arguments: ev.arguments,
            });
            break;
          case "tool_result":
            setPendingApproval(null);
            setMessages((prev) => [
              ...prev,
              {
                id: null,
                role: "tool",
                text: ev.result,
                toolName: ev.name,
                isError: ev.is_error,
              },
            ]);
            break;
          case "usage":
            setMetrics((m) => ({
              ...m,
              promptTokens: ev.prompt_tokens ?? m.promptTokens,
              completionTokens: ev.completion_tokens ?? m.completionTokens,
              totalTokens: ev.total_tokens ?? m.totalTokens,
              cost: ev.cost ?? m.cost,
            }));
            break;
          case "request_metrics":
            setMetrics((m) => ({
              ...m,
              decode: ev.decode_tok_s,
              ttft: ev.ttft_ms,
              prefill: ev.prefill_tok_s,
              durationMs: ev.duration_ms,
              peakDecode: Math.max(m.peakDecode, ev.decode_tok_s ?? 0),
              peakTtft: Math.max(m.peakTtft, ev.ttft_ms ?? 0),
              peakPrefill: Math.max(m.peakPrefill, ev.prefill_tok_s ?? 0),
              throughputHistory: ev.decode_tok_s
                ? [...m.throughputHistory, ev.decode_tok_s].slice(-60)
                : m.throughputHistory,
              ttftHistory: ev.ttft_ms
                ? [...m.ttftHistory, ev.ttft_ms].slice(-60)
                : m.ttftHistory,
            }));
            break;
          case "error":
            setError(ev.message);
            break;
          case "done":
            setMessages((prev) => {
              const next = prev.slice();
              for (let i = next.length - 1; i >= 0; i--) {
                if (next[i].role === "assistant") {
                  next[i] = { ...next[i], streaming: false, id: ev.message_id };
                  break;
                }
              }
              return next;
            });
            break;
        }
  };

  // Fetch ◀ N/M ▶ sibling info for branch points in the path.
  const buildSiblingMap = async (
    msgs: { id: number | null; role: string }[],
  ) => {
    const ids = msgs
      .filter((m) => m.id != null && (m.role === "user" || m.role === "assistant"))
      .map((m) => m.id as number);
    const infos = await Promise.all(ids.map((id) => api.messageSiblings(id)));
    const map: Record<number, SiblingInfo> = {};
    ids.forEach((id, i) => {
      if (infos[i].total > 1) map[id] = infos[i];
    });
    setSiblingMap(map);
  };

  // After a turn, reload the canonical active path from the DB so message ids
  // and branch structure are correct for subsequent edit/regenerate/navigate.
  const reloadAndIndex = async (id: number): Promise<ChatMessage[]> => {
    const data = await api.loadConversation(id);
    const msgs = data.messages.map(toChatMessage);
    setMessages(msgs);
    await buildSiblingMap(data.messages);
    return msgs;
  };

  const runStream = async (
    starter: (h: (ev: ChatEvent) => void) => Promise<number>,
  ) => {
    setError(null);
    setPendingApproval(null);
    reasoningBuf.current = "";
    contentBuf.current = "";
    setStreaming(true);
    setMetrics((m) => ({
      ...m,
      activeRequests: 1,
      decode: null,
      ttft: null,
      prefill: null,
    }));
    let id: number | null = convIdRef.current;
    try {
      id = await starter(handleEvent);
    } catch (e) {
      setError(String(e));
    } finally {
      setStreaming(false);
      setPendingApproval(null);
      setMessages((prev) =>
        prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)),
      );
      setMetrics((m) => ({ ...m, activeRequests: 0 }));
      if (id != null) {
        const msgs = await reloadAndIndex(id);
        // Auto-open a freshly generated previewable artifact (HTML/SVG/MD),
        // Claude-Desktop style.
        const preview = [...collectArtifacts(msgs)].reverse().find(isPreviewable);
        if (preview) {
          setCurrentArtifact(preview);
          setArtifactOpen(true);
        }
      }
      refreshConversations();
    }
  };

  const send = (text: string, images: string[] = []) => {
    if (!settings || !settings.model) return;
    setMessages((prev) => [
      ...prev,
      { id: null, role: "user", text, images },
    ]);
    runStream((h) =>
      api.sendMessage(
        {
          ...genParamsFrom(settings),
          conversation_id: activeConvId,
          user_text: text,
          images,
        },
        h,
      ),
    );
  };

  const regenerate = () => {
    if (!settings || !settings.model || activeConvId == null) return;
    // Drop the trailing assistant/tool turn back to the last user message.
    setMessages((prev) => {
      let cut = prev.length;
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === "user") break;
        cut = i;
      }
      return prev.slice(0, cut);
    });
    runStream((h) =>
      api.regenerate(
        { ...genParamsFrom(settings), conversation_id: activeConvId },
        h,
      ),
    );
  };

  const editMessage = (messageId: number, newText: string) => {
    if (!settings || !settings.model || activeConvId == null) return;
    setMessages((prev) => {
      const pos = prev.findIndex((m) => m.id === messageId);
      const prefix = pos >= 0 ? prev.slice(0, pos) : prev;
      return [...prefix, { id: null, role: "user", text: newText }];
    });
    runStream((h) =>
      api.editMessage(
        {
          ...genParamsFrom(settings),
          conversation_id: activeConvId!,
          message_id: messageId,
          new_text: newText,
        },
        h,
      ),
    );
  };

  const navigateSibling = async (messageId: number, dir: -1 | 1) => {
    const info = siblingMap[messageId];
    if (!info || info.total <= 1) return;
    const targetIdx = info.index + dir;
    if (targetIdx < 0 || targetIdx >= info.total) return;
    const pos = messages.findIndex((m) => m.id === messageId);
    if (pos < 0) return;
    const prefix = messages.slice(0, pos);
    const suffix = await api.walkFrom(info.ids[targetIdx]);
    const next = [...prefix, ...suffix.map(toChatMessage)];
    setMessages(next);
    await buildSiblingMap(next);
  };

  const stop = async () => {
    await api.cancelStream();
  };

  const resolveTool = async (approved: boolean) => {
    if (!pendingApproval) return;
    await api.resolveTool(pendingApproval.id, approved);
    setPendingApproval(null);
  };

  const saveConfig = async (next: Config) => {
    await api.saveConfig(next);
    setConfig(next);
    setShowSettings(false);
  };

  const applyPreset = async (p: PresetDto) => {
    setSettings((prev) =>
      prev
        ? {
            ...prev,
            endpoint: p.endpoint ?? prev.endpoint,
            model: p.model ?? prev.model,
            system_prompt: p.system_prompt ?? prev.system_prompt,
            temperature: p.temperature ?? prev.temperature,
            max_tokens: p.max_tokens ?? prev.max_tokens,
            use_max_tokens: p.use_max_tokens,
            top_p: p.top_p,
            frequency_penalty: p.frequency_penalty,
            presence_penalty: p.presence_penalty,
            stop_sequences: p.stop_sequences,
          }
        : prev,
    );
    if (p.endpoint) await loadModels(p.endpoint);
  };

  const savePreset = async (name: string) => {
    if (!settings) return;
    await api.createPreset(name, genParamsFrom(settings));
    setPresets(await api.listPresets());
  };

  const deletePreset = async (id: number) => {
    await api.deletePreset(id);
    setPresets(await api.listPresets());
  };

  const onSearch = async (q: string) => {
    setSearchQuery(q);
  };

  const shownConversations = searchQuery.trim()
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : conversations;

  if (!config || !settings) {
    return <div className="empty" style={{ height: "100vh" }}>Loading…</div>;
  }

  const showArtifact = artifactOpen && currentArtifact;

  return (
    <div className={`app${showArtifact ? " with-artifact" : ""}`}>
      <Sidebar
        view={view}
        setView={setView}
        conversations={shownConversations}
        activeConvId={activeConvId}
        onSelectConv={selectConv}
        onNewChat={newChat}
        onSearch={onSearch}
        onPin={async (id, pinned) => {
          await api.setPinned(id, pinned);
          refreshConversations();
        }}
        onRename={async (id, title) => {
          await api.renameConversation(id, title);
          refreshConversations();
        }}
        onDelete={async (id) => {
          await api.deleteConversation(id);
          if (activeConvId === id) newChat();
          refreshConversations();
        }}
        onOpenSettings={() => setShowSettings(true)}
      />

      <div className="main">
        {view === "status" ? (
          <StatusView
            settings={settings}
            model={settings.model}
            streaming={streaming}
            metrics={metrics}
            snapshot={snapshot}
          />
        ) : (
          <ChatView
            config={config}
            settings={settings}
            models={models}
            messages={messages}
            streaming={streaming}
            pendingApproval={pendingApproval}
            siblingMap={siblingMap}
            onResolveTool={resolveTool}
            onSend={send}
            onStop={stop}
            onRegenerate={regenerate}
            onEditMessage={editMessage}
            onNavigate={navigateSibling}
            presets={presets}
            onApplyPreset={applyPreset}
            onSavePreset={savePreset}
            onDeletePreset={deletePreset}
            onChangeModel={changeModel}
            onChangeEndpoint={changeEndpoint}
            onOpenParams={() => setShowSettings(true)}
            onOpenArtifact={openArtifact}
            artifactCount={artifacts.length}
            artifactOpen={artifactOpen}
            onToggleArtifacts={toggleArtifacts}
          />
        )}
      </div>

      {showArtifact && currentArtifact && (
        <ArtifactPanel
          artifact={currentArtifact}
          artifacts={artifacts.length > 0 ? artifacts : [currentArtifact]}
          onSelect={(id) => {
            const a =
              artifacts.find((x) => x.id === id) ??
              (currentArtifact.id === id ? currentArtifact : null);
            if (a) setCurrentArtifact(a);
          }}
          onClose={() => setArtifactOpen(false)}
        />
      )}

      {showSettings && (
        <SettingsModal
          config={config}
          onClose={() => setShowSettings(false)}
          onSave={saveConfig}
        />
      )}

      {error && (
        <div className="toast" onClick={() => setError(null)}>
          {error}
        </div>
      )}
    </div>
  );
}
