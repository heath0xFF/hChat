import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./lib/tauri";
import type { Config, ConversationDto, SettingsDto, SendParams } from "./types";
import { Sidebar, type View } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";
import { StatusView, type LiveMetrics } from "./components/StatusView";
import { SettingsModal } from "./components/SettingsModal";
import type { ChatMessage } from "./components/MessageItem";

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

  const reasoningBuf = useRef("");
  const contentBuf = useRef("");

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
      const m = await loadModels(s.endpoint);
      if (m.length) setSettings((prev) => (prev ? { ...prev, model: m[0] } : prev));
    })();
  }, [refreshConversations, loadModels]);

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
    setMessages(
      data.messages.map((m) => ({
        id: m.id,
        role: m.role,
        text: m.text,
        images: m.images,
      })),
    );
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

  const updateLastAssistant = () => {
    const text = combine(reasoningBuf.current, contentBuf.current);
    setMessages((prev) => {
      const next = prev.slice();
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") {
          next[i] = { ...next[i], text };
          break;
        }
      }
      return next;
    });
  };

  const send = async (text: string) => {
    if (!settings || !settings.model) return;
    setError(null);
    reasoningBuf.current = "";
    contentBuf.current = "";
    setMessages((prev) => [
      ...prev,
      { id: null, role: "user", text },
      { id: null, role: "assistant", text: "", streaming: true },
    ]);
    setStreaming(true);
    setMetrics((m) => ({ ...m, activeRequests: 1, decode: null, ttft: null, prefill: null }));

    const params: SendParams = {
      conversation_id: activeConvId,
      endpoint: settings.endpoint,
      model: settings.model,
      system_prompt: settings.system_prompt,
      temperature: settings.temperature,
      max_tokens: settings.max_tokens,
      use_max_tokens: settings.use_max_tokens,
      top_p: settings.top_p,
      frequency_penalty: settings.frequency_penalty,
      presence_penalty: settings.presence_penalty,
      stop_sequences: settings.stop_sequences,
      user_text: text,
    };

    try {
      await api.sendMessage(params, (ev) => {
        switch (ev.type) {
          case "started":
            if (activeConvId === null) setActiveConvId(ev.conversation_id);
            break;
          case "token":
            contentBuf.current += ev.text;
            updateLastAssistant();
            break;
          case "reasoning":
            reasoningBuf.current += ev.text;
            updateLastAssistant();
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
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setStreaming(false);
      setMetrics((m) => ({ ...m, activeRequests: 0 }));
      refreshConversations();
    }
  };

  const stop = async () => {
    await api.cancelStream();
  };

  const saveConfig = async (next: Config) => {
    await api.saveConfig(next);
    setConfig(next);
    setShowSettings(false);
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

  return (
    <div className="app">
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
          />
        ) : (
          <ChatView
            config={config}
            settings={settings}
            models={models}
            messages={messages}
            streaming={streaming}
            onSend={send}
            onStop={stop}
            onChangeModel={changeModel}
            onChangeEndpoint={changeEndpoint}
            onOpenParams={() => setShowSettings(true)}
          />
        )}
      </div>

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
