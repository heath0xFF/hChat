import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api } from "./lib/tauri";
import type {
  AgentsDto,
  ChatEvent,
  Config,
  ConversationDto,
  GenParams,
  MessageDto,
  MetricsSnapshot,
  PendingApproval,
  PresetDto,
  ProjectDto,
  SettingsDto,
  SiblingInfo,
} from "./types";
import { Sidebar, type View } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";
import { StatusView, type LiveMetrics } from "./components/StatusView";
import { UsageView } from "./components/UsageView";
import { SettingsModal } from "./components/SettingsModal";
import { ArtifactPanel } from "./components/ArtifactPanel";
import { RightDock, type DockTab } from "./components/RightDock";
import type { ApprovalDecision } from "./components/ApprovalCard";
import type { ChatMessage } from "./components/MessageItem";
import {
  collectArtifacts,
  isPreviewable,
  makeArtifact,
  type Artifact,
} from "./lib/artifacts";
import { parseSlash, SLASH_HELP } from "./lib/slash";
import { parseThink } from "./lib/segments";
import { countTokens } from "./lib/tokens";
import { matchCombo, comboIsBare, isTypingTarget } from "./lib/hotkeys";

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
  gpuUtilHistory: [],
  gpuPowerHistory: [],
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

const SANS_STACK = `-apple-system, "Inter", "Segoe UI", system-ui, sans-serif`;
const MONO_STACK = `"SF Mono", "JetBrains Mono", "Menlo", ui-monospace, monospace`;

function applyAppearance(c: Config) {
  const root = document.documentElement;
  root.classList.toggle("light", !c.dark_mode);
  root.style.setProperty("--app-font-size", `${c.font_size}px`);
  root.style.setProperty("--mono-font-size", `${c.mono_font_size}px`);
  root.style.setProperty("zoom", String(c.ui_scale));
  root.style.setProperty(
    "--sans",
    c.font_family ? `"${c.font_family}", ${SANS_STACK}` : SANS_STACK,
  );
  root.style.setProperty(
    "--mono",
    c.mono_font_family ? `"${c.mono_font_family}", ${MONO_STACK}` : MONO_STACK,
  );
}

function expandCommand(body: string, args: string): string {
  if (/\$ARGUMENTS|\{\{\s*args\s*\}\}/.test(body)) {
    return body
      .replace(/\$ARGUMENTS/g, args)
      .replace(/\{\{\s*args\s*\}\}/g, args);
  }
  return args ? `${body}\n\n${args}` : body;
}

function toChatMessage(m: MessageDto): ChatMessage {
  return {
    id: m.id,
    role: m.role,
    text: m.text,
    images: m.images,
    createdAt: m.created_at ?? undefined,
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
  const [projects, setProjects] = useState<ProjectDto[]>([]);
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
  const [statusEndpoint, setStatusEndpoint] = useState("");
  const [railWidth, setRailWidth] = useState(() => {
    const s = localStorage.getItem("railWidth");
    return s ? Number(s) : 248;
  });
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [currentArtifact, setCurrentArtifact] = useState<Artifact | null>(null);
  // The right dock hosts Status + Artifacts as tabs (open while chatting).
  const [dockOpen, setDockOpen] = useState(false);
  const [dockTab, setDockTab] = useState<DockTab>("status");
  const [composerInput, setComposerInput] = useState("");
  const [tokenCount, setTokenCount] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const [focusSignal, setFocusSignal] = useState(0);
  const [agents, setAgents] = useState<AgentsDto>({ commands: [], skills: [] });

  const reasoningBuf = useRef("");
  const contentBuf = useRef("");
  const convIdRef = useRef<number | null>(null);
  const hotkeyRef = useRef<(e: KeyboardEvent) => void>(() => {});
  const viewRef = useRef(view);
  viewRef.current = view;
  // Metrics should update (and the poller stay pointed at the status endpoint)
  // whenever the live metrics are visible — the full-page Status view OR the
  // right dock's Status tab. Kept in a ref so the "metrics" listener stays
  // stable while reading the latest value.
  const metricsVisible = view === "status" || (dockOpen && dockTab === "status");
  const metricsVisibleRef = useRef(metricsVisible);
  metricsVisibleRef.current = metricsVisible;

  const refreshConversations = useCallback(async () => {
    setConversations(await api.listConversations());
  }, []);

  const refreshProjects = useCallback(async () => {
    setProjects(await api.listProjects());
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
      setStatusEndpoint(cfg.default_endpoint);
      const s = defaultSettings(cfg);
      setSettings(s);
      await refreshConversations();
      await refreshProjects();
      setPresets(await api.listPresets());
      const m = await loadModels(s.endpoint);
      if (m.length) setSettings((prev) => (prev ? { ...prev, model: m[0] } : prev));
    })();
  }, [refreshConversations, refreshProjects, loadModels]);

  // Subscribe to the backend metrics poller. Each snapshot updates the GPU/
  // server view and feeds the live charts (so they move during idle polling,
  // not only when we send a request).
  useEffect(() => {
    const un = listen<MetricsSnapshot>("metrics", (e) => {
      // Only update metrics state while the live metrics are visible (Status
      // page or the dock's Status tab) — otherwise the ~1.5s poller would
      // re-render the chat every tick for no visible benefit.
      if (!metricsVisibleRef.current) return;
      const snap = e.payload;
      setSnapshot(snap);
      const decode = snap.server?.decode_tok_s ?? null;
      const ttft = snap.server?.ttft_ms ?? null;
      const util = snap.gpu?.util_pct ?? null;
      const power = snap.gpu?.power_w ?? null;
      if (decode != null || ttft != null || util != null || power != null) {
        setMetrics((m) => ({
          ...m,
          throughputHistory:
            decode != null
              ? [...m.throughputHistory, decode].slice(-120)
              : m.throughputHistory,
          ttftHistory:
            ttft != null ? [...m.ttftHistory, ttft].slice(-120) : m.ttftHistory,
          gpuUtilHistory:
            util != null
              ? [...m.gpuUtilHistory, util].slice(-120)
              : m.gpuUtilHistory,
          gpuPowerHistory:
            power != null
              ? [...m.gpuPowerHistory, power].slice(-120)
              : m.gpuPowerHistory,
        }));
      }
    });
    return () => {
      un.then((f) => f());
    };
  }, []);

  // Point the metrics poller at the Status-selected endpoint while on the Status
  // view, otherwise at the active conversation's endpoint.
  useEffect(() => {
    const ep = metricsVisible ? statusEndpoint : settings?.endpoint;
    if (ep) void api.setMetricsTarget(ep);
  }, [metricsVisible, statusEndpoint, settings?.endpoint]);

  // Clear the metric histories when the Status endpoint changes so charts
  // reflect the newly-selected device.
  useEffect(() => {
    setSnapshot(null);
    setMetrics((m) => ({
      ...m,
      throughputHistory: [],
      ttftHistory: [],
      gpuUtilHistory: [],
      gpuPowerHistory: [],
    }));
  }, [statusEndpoint]);

  // Load ~/.agents commands/skills (+ project .agents under the working dir).
  useEffect(() => {
    api
      .listAgents(settings?.working_dir ?? null)
      .then(setAgents)
      .catch(() => setAgents({ commands: [], skills: [] }));
  }, [settings?.working_dir]);

  // Apply theme / fonts / UI scale from config.
  useEffect(() => {
    if (config) applyAppearance(config);
  }, [config]);

  // Global hotkeys — bound once, dispatched through a ref to the latest closure.
  useEffect(() => {
    const h = (e: KeyboardEvent) => hotkeyRef.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  // Re-collect artifacts whenever the visible message path changes.
  useEffect(() => {
    setArtifacts(collectArtifacts(messages));
  }, [messages]);

  // Close the right dock when leaving a chat (switch conversation or view).
  useEffect(() => {
    setDockOpen(false);
  }, [activeConvId, view]);

  // Debounced live token count of the composer.
  useEffect(() => {
    const t = setTimeout(() => {
      setTokenCount(composerInput.trim() ? countTokens(composerInput) : 0);
    }, 150);
    return () => clearTimeout(t);
  }, [composerInput]);

  // Debounced per-conversation draft persistence.
  useEffect(() => {
    if (activeConvId == null) return;
    const id = activeConvId;
    const t = setTimeout(() => void api.saveDraft(id, composerInput), 500);
    return () => clearTimeout(t);
  }, [composerInput, activeConvId]);

  const handleSlash = (raw: string): boolean => {
    const r = parseSlash(raw);
    if (r.kind === "none") return false;
    if (r.kind === "unknown") {
      // Fall through to ~/.agents commands + skills before declaring unknown.
      const args = raw.trim().match(/^\/\S+\s*([\s\S]*)$/)?.[1] ?? "";
      const cmd = agents.commands.find((c) => c.name.toLowerCase() === r.verb);
      if (cmd) {
        send(expandCommand(cmd.body, args));
        return true;
      }
      const skill = agents.skills.find((s) => s.name.toLowerCase() === r.verb);
      if (skill) {
        send(args ? `${skill.body}\n\n${args}` : skill.body);
        return true;
      }
      setError(`Unknown command: /${r.verb} — try /help`);
      return true;
    }
    if (r.kind === "badargs") {
      setError(r.reason);
      return true;
    }
    const c = r.cmd;
    switch (c.type) {
      case "model": {
        const match = models.find((m) =>
          m.toLowerCase().includes(c.arg.toLowerCase()),
        );
        if (match) setSettings((p) => (p ? { ...p, model: match } : p));
        else setError(`No model matches "${c.arg}"`);
        break;
      }
      case "temp":
        setSettings((p) =>
          p ? { ...p, temperature: Math.min(2, Math.max(0, c.value)) } : p,
        );
        break;
      case "system":
        setSettings((p) => (p ? { ...p, system_prompt: c.text } : p));
        break;
      case "clear":
        newChat();
        break;
      case "copy": {
        const last = [...messages].reverse().find((m) => m.role === "assistant");
        if (last) {
          const body = parseThink(last.text).body || last.text;
          void navigator.clipboard.writeText(body);
        }
        break;
      }
      case "help": {
        let h = SLASH_HELP;
        if (agents.commands.length) {
          h +=
            "\n\nCommands (~/.agents):\n" +
            agents.commands
              .map((cmd) => `/${cmd.name}${cmd.description ? "  — " + cmd.description : ""}`)
              .join("\n");
        }
        if (agents.skills.length) {
          h +=
            "\n\nSkills (/name, or the model can use them):\n" +
            agents.skills
              .map((s) => `/${s.name}${s.description ? "  — " + s.description : ""}`)
              .join("\n");
        }
        setNotice(h);
        break;
      }
    }
    return true;
  };

  const openArtifact = useCallback(
    (code: string, lang: string) => {
      const match =
        artifacts.find((a) => a.code === code) ?? makeArtifact(code, lang);
      setCurrentArtifact(match);
      setDockTab("artifacts");
      setDockOpen(true);
    },
    [artifacts],
  );

  // Toggle the dock's Artifacts tab (the chat-topbar ◧ button + hotkey).
  const toggleArtifacts = () => {
    if (dockOpen && dockTab === "artifacts") {
      setDockOpen(false);
      return;
    }
    // Nothing to show — don't open an empty Artifacts tab (the hotkey can fire
    // with no artifacts present).
    if (!currentArtifact && artifacts.length === 0) return;
    if (!currentArtifact) {
      setCurrentArtifact(artifacts[artifacts.length - 1]);
    }
    setDockTab("artifacts");
    setDockOpen(true);
  };

  // Toggle the dock's Status tab (watch live metrics while chatting).
  const toggleStatusDock = () => {
    if (dockOpen && dockTab === "status") {
      setDockOpen(false);
      return;
    }
    setDockTab("status");
    setDockOpen(true);
  };

  const newChat = () => {
    if (!config) return;
    setActiveConvId(null);
    convIdRef.current = null;
    setMessages([]);
    setComposerInput("");
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
    setComposerInput(data.draft ?? "");
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
            if (activeConvId === null) {
              // First message in a new chat: the backend just created the
              // conversation row, so surface it in the sidebar immediately
              // (titled "untitled chat" until auto-titling runs post-response).
              setActiveConvId(ev.conversation_id);
              refreshConversations();
            }
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
          setDockTab("artifacts");
          setDockOpen(true);
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

  const resolveTool = async (decision: ApprovalDecision) => {
    if (!pendingApproval) return;
    await api.resolveTool(pendingApproval.id, decision);
    setPendingApproval(null);
  };

  const exportConversation = async () => {
    if (activeConvId == null) return;
    try {
      const path = await api.exportConversationFile(activeConvId);
      setNotice(`Exported to ${path}`);
    } catch (e) {
      setError(String(e));
    }
  };

  const saveConfig = async (next: Config) => {
    await api.saveConfig(next);
    setConfig(next);
    setShowSettings(false);
    // Apply MCP server changes (connect/disconnect) without a restart.
    void api.reconnectMcp().catch(() => {});
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

  hotkeyRef.current = (e: KeyboardEvent) => {
    if (!config) return;
    const hk = config.hotkeys;
    const fire = (combo: string, fn: () => void): boolean => {
      if (!matchCombo(e, combo)) return false;
      if (comboIsBare(combo) && isTypingTarget(e)) return false;
      e.preventDefault();
      fn();
      return true;
    };
    fire(hk.settings, () => setShowSettings(true)) ||
      fire(hk.new_chat, newChat) ||
      fire(hk.toggle_artifacts, toggleArtifacts) ||
      fire(hk.focus_input, () => setFocusSignal((s) => s + 1)) ||
      fire(hk.stop, () => {
        if (streaming) void stop();
      });
  };

  // The dock only docks alongside the chat (not the full-page Status/Usage).
  const showDock = dockOpen && view === "chat";

  const startResize = (e: React.MouseEvent) => {
    e.preventDefault();
    document.body.style.userSelect = "none";
    const onMove = (ev: MouseEvent) => {
      const w = Math.min(480, Math.max(180, ev.clientX));
      setRailWidth(w);
      localStorage.setItem("railWidth", String(w));
    };
    const onUp = () => {
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const gridTemplateColumns = showDock
    ? `${railWidth}px minmax(0, 1fr) minmax(380px, 0.9fr)`
    : `${railWidth}px 1fr`;

  return (
    <div className="app" style={{ gridTemplateColumns }}>
      <div
        className="rail-resize"
        style={{ left: railWidth - 2 }}
        onMouseDown={startResize}
      />
      <Sidebar
        view={view}
        setView={setView}
        conversations={shownConversations}
        projects={projects}
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
        onCreateProject={async () => {
          const name = window.prompt("Project name");
          if (name?.trim()) {
            await api.createProject(name.trim());
            refreshProjects();
          }
        }}
        onRenameProject={async (id, name) => {
          await api.renameProject(id, name);
          refreshProjects();
        }}
        onDeleteProject={async (id) => {
          await api.deleteProject(id);
          await refreshProjects();
          refreshConversations();
        }}
        onPinProject={async (id, pinned) => {
          await api.setProjectPinned(id, pinned);
          refreshProjects();
        }}
        onMoveConv={async (convId, projectId) => {
          await api.setConversationProject(convId, projectId);
          refreshConversations();
        }}
        onOpenSettings={() => setShowSettings(true)}
      />

      <div className="main">
        {view === "status" ? (
          <StatusView
            endpoints={config.saved_endpoints.map((e) => e.url)}
            endpoint={statusEndpoint}
            liveEndpoint={settings.endpoint}
            onChangeEndpoint={setStatusEndpoint}
            metrics={metrics}
            snapshot={snapshot}
          />
        ) : view === "usage" ? (
          <UsageView />
        ) : (
          <ChatView
            config={config}
            settings={settings}
            models={models}
            messages={messages}
            streaming={streaming}
            pendingApproval={pendingApproval}
            siblingMap={siblingMap}
            input={composerInput}
            onInputChange={setComposerInput}
            tokenCount={tokenCount}
            findCombo={config.hotkeys.find}
            focusSignal={focusSignal}
            onResolveTool={resolveTool}
            onSend={send}
            onSlash={handleSlash}
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
            onExport={exportConversation}
            canExport={activeConvId != null}
            onOpenArtifact={openArtifact}
            artifactCount={artifacts.length}
            artifactOpen={dockOpen && dockTab === "artifacts"}
            onToggleArtifacts={toggleArtifacts}
            statusDockOpen={dockOpen && dockTab === "status"}
            onToggleStatus={toggleStatusDock}
          />
        )}
      </div>

      {showDock && (
        <RightDock
          tab={dockTab}
          onTab={setDockTab}
          onClose={() => setDockOpen(false)}
          hasArtifacts={artifacts.length > 0 || currentArtifact != null}
          artifactCount={artifacts.length}
          status={
            <StatusView
              endpoints={config.saved_endpoints.map((e) => e.url)}
              endpoint={statusEndpoint}
              liveEndpoint={settings.endpoint}
              onChangeEndpoint={setStatusEndpoint}
              metrics={metrics}
              snapshot={snapshot}
            />
          }
          artifacts={
            currentArtifact ? (
              <ArtifactPanel
                embedded
                artifact={currentArtifact}
                artifacts={artifacts.length > 0 ? artifacts : [currentArtifact]}
                onSelect={(id) => {
                  const a =
                    artifacts.find((x) => x.id === id) ??
                    (currentArtifact.id === id ? currentArtifact : null);
                  if (a) setCurrentArtifact(a);
                }}
                onClose={() => setDockOpen(false)}
              />
            ) : (
              <div className="dock-empty">No artifacts in this chat yet.</div>
            )
          }
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

      {notice && (
        <div className="toast notice" onClick={() => setNotice(null)}>
          {notice}
        </div>
      )}
    </div>
  );
}
