import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { ModelsView } from "./components/ModelsView";
import { SettingsModal } from "./components/SettingsModal";
import { useDialog } from "./components/Dialog";
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
  root.dataset.theme = c.theme || (c.dark_mode ? "dark" : "light");
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

// Per-conversation streaming runtime. Each chat owns its own message list and
// streaming flag so multiple chats can stream concurrently — switching chats
// just changes which entry is on screen; background streams keep updating their
// own entry. Brand-new chats live under a negative temp id until the backend
// assigns a real conversation_id (see the `started` event in `handleEvent`).
interface ConvRuntime {
  messages: ChatMessage[];
  streaming: boolean;
}

const EMPTY_CONV: ConvRuntime = { messages: [], streaming: false };

// One in-flight stream's mutable context. `convId` starts as the temp id of an
// unsaved chat and is rewritten to the real conversation_id on `started`, so
// later events from the same stream route to the right conversation.
interface StreamSession {
  convId: number;
  reasoning: string;
  content: string;
}

export function App() {
  const dialog = useDialog();
  const [config, setConfig] = useState<Config | null>(null);
  const [conversations, setConversations] = useState<ConversationDto[]>([]);
  const [projects, setProjects] = useState<ProjectDto[]>([]);
  const [view, setView] = useState<View>("chat");
  const [activeConvId, setActiveConvId] = useState<number | null>(null);
  // All chats' runtimes keyed by conversation_id (negative = unsaved new chat).
  const [convStore, setConvStore] = useState<Record<number, ConvRuntime>>({});
  const [settings, setSettings] = useState<SettingsDto | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [metrics, setMetrics] = useState<LiveMetrics>(EMPTY_METRICS);
  const [searchQuery, setSearchQuery] = useState("");
  // Parked tool approvals keyed by conversation_id — a background chat can be
  // waiting on approval while you read another; resolving shows per active chat.
  const [pendingByConv, setPendingByConv] = useState<
    Record<number, PendingApproval>
  >({});
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

  // Monotonically-decreasing temp ids for unsaved new chats (before the backend
  // assigns a real conversation_id).
  const nextTempId = useRef(-1);
  // Latest active conversation, readable from stream callbacks that closed over
  // a stale value (e.g. the post-turn artifact auto-open in `runStream`).
  const activeConvIdRef = useRef<number | null>(null);
  activeConvIdRef.current = activeConvId;
  const configRef = useRef<Config | null>(null);
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

  // ----- derived view of the active conversation -----
  const messages = useMemo<ChatMessage[]>(
    () => (activeConvId != null ? convStore[activeConvId]?.messages ?? [] : []),
    [convStore, activeConvId],
  );
  const streaming =
    activeConvId != null ? convStore[activeConvId]?.streaming ?? false : false;
  const pendingApproval =
    activeConvId != null ? pendingByConv[activeConvId] ?? null : null;
  // Conversation ids with a stream in flight (drives the sidebar indicator).
  const streamingIds = useMemo(
    () =>
      Object.entries(convStore)
        .filter(([, v]) => v.streaming)
        .map(([k]) => Number(k)),
    [convStore],
  );

  // ----- per-conversation store mutators -----
  const patchConvMessages = useCallback(
    (id: number, fn: (msgs: ChatMessage[]) => ChatMessage[]) =>
      setConvStore((s) => {
        const cur = s[id] ?? EMPTY_CONV;
        return { ...s, [id]: { ...cur, messages: fn(cur.messages) } };
      }),
    [],
  );
  const setConvMessages = useCallback(
    (id: number, msgs: ChatMessage[]) => patchConvMessages(id, () => msgs),
    [patchConvMessages],
  );
  const setConvStreaming = useCallback(
    (id: number, on: boolean) =>
      setConvStore((s) => ({
        ...s,
        [id]: { ...(s[id] ?? EMPTY_CONV), streaming: on },
      })),
    [],
  );
  const clearPending = useCallback(
    (id: number) =>
      setPendingByConv((p) => {
        if (!(id in p)) return p;
        const n = { ...p };
        delete n[id];
        return n;
      }),
    [],
  );

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

  // Apply theme / fonts / UI scale from config; mirror config into a ref so the
  // debounced appearance save can read the latest value.
  useEffect(() => {
    if (config) applyAppearance(config);
    configRef.current = config;
  }, [config]);

  // Global hotkeys — bound once, dispatched through a ref to the latest closure.
  useEffect(() => {
    const h = (e: KeyboardEvent) => hotkeyRef.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  // Auto-dismiss toasts after 5s (still clickable to dismiss sooner).
  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 5000);
    return () => clearTimeout(t);
  }, [error]);
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 5000);
    return () => clearTimeout(t);
  }, [notice]);

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

  // Debounced per-conversation draft persistence (saved chats only).
  useEffect(() => {
    if (activeConvId == null || activeConvId < 0) return;
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
    // Default the Status view to the endpoint this chat is talking to.
    if (settings?.endpoint) setStatusEndpoint(settings.endpoint);
    setDockTab("status");
    setDockOpen(true);
  };

  const newChat = () => {
    if (!config) return;
    // Empty new chat stays unkeyed (activeConvId null) until the first send,
    // which assigns a temp id. Background streams in other chats keep running.
    setActiveConvId(null);
    setComposerInput("");
    setView("chat");
    const s = defaultSettings(config);
    s.model = models[0] ?? null;
    setSettings((prev) => ({ ...s, endpoint: prev?.endpoint ?? s.endpoint, model: prev?.model ?? s.model }));
  };

  const selectConv = async (id: number) => {
    const data = await api.loadConversation(id);
    setActiveConvId(id);
    // Don't clobber a chat that's mid-stream with the (stale) DB snapshot —
    // its live messages are authoritative until the turn finishes and reloads.
    // The sibling map reflects committed branch structure, so rebuild it either
    // way (in-progress messages have null ids and don't appear in it).
    const live = convStore[id]?.streaming ?? false;
    if (!live) setConvMessages(id, data.messages.map(toChatMessage));
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

  // From the Models page: point the current chat at this endpoint + model and
  // jump to the conversation.
  const useModel = (endpoint: string, model: string) => {
    setSettings((prev) => (prev ? { ...prev, endpoint, model } : prev));
    void loadModels(endpoint);
    setView("chat");
  };

  // Mutate the most recent assistant message of a given conversation (the one
  // currently streaming there).
  const patchLastAssistant = (
    id: number,
    fn: (m: ChatMessage) => ChatMessage,
  ) =>
    patchConvMessages(id, (prev) => {
      const next = prev.slice();
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") {
          next[i] = fn(next[i]);
          break;
        }
      }
      return next;
    });

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

  // Shared streaming event handler used by send / regenerate / edit. Bound to a
  // `session` so every mutation targets that stream's conversation, regardless
  // of which chat is currently on screen.
  const handleEvent = (ev: ChatEvent, session: StreamSession) => {
        switch (ev.type) {
          case "started": {
            const realId = ev.conversation_id;
            const tempId = session.convId;
            if (tempId !== realId) {
              // The backend just created the conversation row. Migrate the
              // optimistic temp-id entry onto the real id and surface it in the
              // sidebar (titled "untitled chat" until auto-titling runs).
              setConvStore((s) => {
                const entry = s[tempId] ?? EMPTY_CONV;
                const n = { ...s };
                delete n[tempId];
                n[realId] = entry;
                return n;
              });
              setPendingByConv((p) => {
                if (!(tempId in p)) return p;
                const n = { ...p };
                n[realId] = n[tempId];
                delete n[tempId];
                return n;
              });
              // Follow the migration on screen only if still viewing this chat.
              setActiveConvId((cur) => (cur === tempId ? realId : cur));
              session.convId = realId;
              refreshConversations();
            }
            break;
          }
          case "turn_start":
            // Close out the previous assistant turn, reset buffers, and open a
            // fresh streaming placeholder for this turn.
            session.reasoning = "";
            session.content = "";
            patchConvMessages(session.convId, (prev) => {
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
            session.content += ev.text;
            patchLastAssistant(session.convId, (m) => ({
              ...m,
              text: combine(session.reasoning, session.content),
            }));
            break;
          case "reasoning":
            session.reasoning += ev.text;
            patchLastAssistant(session.convId, (m) => ({
              ...m,
              text: combine(session.reasoning, session.content),
            }));
            break;
          case "tool_call":
            patchLastAssistant(session.convId, (m) => ({
              ...m,
              toolCalls: [
                ...(m.toolCalls ?? []),
                { id: ev.id, name: ev.name, arguments: ev.arguments },
              ],
            }));
            break;
          case "tool_approval":
            setPendingByConv((p) => ({
              ...p,
              [session.convId]: {
                id: ev.id,
                name: ev.name,
                arguments: ev.arguments,
              },
            }));
            break;
          case "tool_result":
            clearPending(session.convId);
            patchConvMessages(session.convId, (prev) => [
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
            patchConvMessages(session.convId, (prev) => {
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

  // After a turn, reload the canonical path for that conversation from the DB so
  // message ids and branch structure are correct for subsequent edit/
  // regenerate/navigate. The sibling map is only relevant for the chat on
  // screen, so rebuild it only when this conversation is the active one.
  const reloadConv = async (id: number): Promise<ChatMessage[]> => {
    const data = await api.loadConversation(id);
    const msgs = data.messages.map(toChatMessage);
    setConvMessages(id, msgs);
    if (id === activeConvIdRef.current) await buildSiblingMap(data.messages);
    return msgs;
  };

  const runStream = async (
    startId: number,
    starter: (h: (ev: ChatEvent) => void) => Promise<number>,
  ) => {
    // Snapshot artifacts already in this chat so we only auto-open the dock for
    // a genuinely new one — not re-open after every reply in a chat that
    // already has artifacts.
    const priorArtifactCodes = new Set(
      collectArtifacts(convStore[startId]?.messages ?? []).map((a) => a.code),
    );
    const session: StreamSession = {
      convId: startId,
      reasoning: "",
      content: "",
    };
    setError(null);
    clearPending(startId);
    setConvStreaming(startId, true);
    setMetrics((m) => ({
      ...m,
      activeRequests: m.activeRequests + 1,
      decode: null,
      ttft: null,
      prefill: null,
    }));
    try {
      await starter((ev) => handleEvent(ev, session));
    } catch (e) {
      setError(String(e));
    } finally {
      // session.convId is the real id after `started` migrated any temp id.
      const fid = session.convId;
      setConvStreaming(fid, false);
      patchConvMessages(fid, (prev) =>
        prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)),
      );
      clearPending(fid);
      setMetrics((m) => ({
        ...m,
        activeRequests: Math.max(0, m.activeRequests - 1),
      }));
      // A still-negative id means the request failed before `started` ever
      // arrived (e.g. a DB error creating the conversation), so there's nothing
      // in the DB to reload — keep the optimistic messages so the user sees
      // what they typed alongside the error toast, and skip the reload that
      // would otherwise wipe them.
      if (fid > 0) {
        const msgs = await reloadConv(fid);
        // Auto-open a freshly generated previewable artifact (HTML/SVG/MD),
        // Claude-Desktop style — but only when this chat is the one on screen,
        // so a background stream doesn't yank the dock open under you.
        if (fid === activeConvIdRef.current) {
          const preview = [...collectArtifacts(msgs)]
            .reverse()
            .find((a) => isPreviewable(a) && !priorArtifactCodes.has(a.code));
          if (preview) {
            setCurrentArtifact(preview);
            setDockTab("artifacts");
            setDockOpen(true);
          }
        }
      }
      refreshConversations();
    }
  };

  const send = (text: string, images: string[] = []) => {
    if (!settings || !settings.model) return;
    // New chat: allocate a temp id and surface it on screen; existing chat:
    // append to its own store. The temp id becomes real on the `started` event.
    const startId = activeConvId ?? nextTempId.current--;
    if (activeConvId == null) setActiveConvId(startId);
    patchConvMessages(startId, (prev) => [
      ...prev,
      { id: null, role: "user", text, images },
    ]);
    runStream(startId, (h) =>
      api.sendMessage(
        {
          ...genParamsFrom(settings),
          conversation_id: startId > 0 ? startId : null,
          user_text: text,
          images,
        },
        h,
      ),
    );
  };

  const regenerate = () => {
    if (!settings || !settings.model || activeConvId == null) return;
    const id = activeConvId;
    // Drop the trailing assistant/tool turn back to the last user message.
    patchConvMessages(id, (prev) => {
      let cut = prev.length;
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === "user") break;
        cut = i;
      }
      return prev.slice(0, cut);
    });
    runStream(id, (h) =>
      api.regenerate({ ...genParamsFrom(settings), conversation_id: id }, h),
    );
  };

  const editMessage = (messageId: number, newText: string) => {
    if (!settings || !settings.model || activeConvId == null) return;
    const id = activeConvId;
    patchConvMessages(id, (prev) => {
      const pos = prev.findIndex((m) => m.id === messageId);
      const prefix = pos >= 0 ? prev.slice(0, pos) : prev;
      return [...prefix, { id: null, role: "user", text: newText }];
    });
    runStream(id, (h) =>
      api.editMessage(
        {
          ...genParamsFrom(settings),
          conversation_id: id,
          message_id: messageId,
          new_text: newText,
        },
        h,
      ),
    );
  };

  const navigateSibling = async (messageId: number, dir: -1 | 1) => {
    if (activeConvId == null) return;
    const info = siblingMap[messageId];
    if (!info || info.total <= 1) return;
    const targetIdx = info.index + dir;
    if (targetIdx < 0 || targetIdx >= info.total) return;
    const pos = messages.findIndex((m) => m.id === messageId);
    if (pos < 0) return;
    const prefix = messages.slice(0, pos);
    const suffix = await api.walkFrom(info.ids[targetIdx]);
    const next = [...prefix, ...suffix.map(toChatMessage)];
    setConvMessages(activeConvId, next);
    await buildSiblingMap(next);
  };

  const stop = async () => {
    if (activeConvId == null || activeConvId < 0) return;
    await api.cancelStream(activeConvId);
  };

  const resolveTool = async (decision: ApprovalDecision) => {
    if (!pendingApproval || activeConvId == null) return;
    await api.resolveTool(pendingApproval.id, decision);
    clearPending(activeConvId);
  };

  const exportConversation = async () => {
    if (activeConvId == null || activeConvId < 0) return;
    try {
      const path = await api.exportConversationFile(activeConvId);
      setNotice(`Exported to ${path}`);
    } catch (e) {
      setError(String(e));
    }
  };

  // Live appearance edits from the Settings modal: apply to the DOM immediately
  // (via the config effect) and persist on a short debounce, so theme/fonts/
  // scale take effect without clicking Save.
  const appearanceSaveTimer = useRef<number | null>(null);
  const applyAppearanceLive = (patch: Partial<Config>) => {
    setConfig((prev) => (prev ? { ...prev, ...patch } : prev));
    if (appearanceSaveTimer.current) clearTimeout(appearanceSaveTimer.current);
    appearanceSaveTimer.current = window.setTimeout(() => {
      if (configRef.current) void api.saveConfig(configRef.current);
    }, 400);
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
    // Esc closes the right dock (Status or Artifacts tab) when it's open.
    // The settings modal and find bar handle their own Esc, so defer to them.
    if (e.key === "Escape" && dockOpen && view === "chat" && !showSettings) {
      e.preventDefault();
      setDockOpen(false);
      return;
    }
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

  // Artifact to render in the dock: the explicitly-selected one, else fall back
  // to the latest so switching to the Artifacts tab always shows something.
  const shownArtifact =
    currentArtifact ?? (artifacts.length ? artifacts[artifacts.length - 1] : null);

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
        streamingIds={streamingIds}
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
          const title = conversations.find((c) => c.id === id)?.title ?? "chat";
          const ok = await dialog.confirm(`Delete "${title}"?`, {
            title: "Delete chat",
            confirmText: "Delete",
            danger: true,
          });
          if (!ok) return;
          await api.deleteConversation(id);
          setConvStore((s) => {
            if (!(id in s)) return s;
            const n = { ...s };
            delete n[id];
            return n;
          });
          clearPending(id);
          if (activeConvId === id) newChat();
          refreshConversations();
        }}
        onCreateProject={async () => {
          const name = await dialog.prompt("Project name", {
            title: "New project",
            placeholder: "Project name",
            confirmText: "Create",
          });
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
          const name = projects.find((p) => p.id === id)?.name ?? "project";
          const ok = await dialog.confirm(
            `Delete project "${name}"? Its chats are kept.`,
            { title: "Delete project", confirmText: "Delete", danger: true },
          );
          if (!ok) return;
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
        ) : view === "models" ? (
          <ModelsView
            endpoints={config.saved_endpoints}
            activeEndpoint={settings.endpoint}
            activeModel={settings.model}
            onUseModel={useModel}
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
            canExport={activeConvId != null && activeConvId > 0}
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
              embedded
              endpoints={config.saved_endpoints.map((e) => e.url)}
              endpoint={statusEndpoint}
              liveEndpoint={settings.endpoint}
              onChangeEndpoint={setStatusEndpoint}
              metrics={metrics}
              snapshot={snapshot}
            />
          }
          artifacts={
            shownArtifact ? (
              <ArtifactPanel
                embedded
                artifact={shownArtifact}
                artifacts={artifacts.length > 0 ? artifacts : [shownArtifact]}
                onSelect={(id) => {
                  const a =
                    artifacts.find((x) => x.id === id) ??
                    (shownArtifact.id === id ? shownArtifact : null);
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
          onApplyAppearance={applyAppearanceLive}
          onClearHistory={async () => {
            const ok = await dialog.confirm(
              "Delete ALL chat history? Every conversation and its messages will be permanently removed. Projects are kept.",
              {
                title: "Delete all chat history",
                confirmText: "Delete everything",
                danger: true,
              },
            );
            if (!ok) return;
            await api.deleteAllConversations();
            newChat();
            refreshConversations();
          }}
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
