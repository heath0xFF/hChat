import { useEffect, useMemo, useRef, useState } from "react";
import type {
  Config,
  PendingApproval,
  PresetDto,
  SettingsDto,
  SiblingInfo,
} from "../types";
import type { ChatMessage } from "./MessageItem";
import { MessageItem } from "./MessageItem";
import { ApprovalCard } from "./ApprovalCard";

interface Props {
  config: Config;
  settings: SettingsDto;
  models: string[];
  messages: ChatMessage[];
  streaming: boolean;
  pendingApproval: PendingApproval | null;
  siblingMap: Record<number, SiblingInfo>;
  input: string;
  onInputChange: (v: string | ((prev: string) => string)) => void;
  tokenCount: number;
  onResolveTool: (approved: boolean) => void;
  onSend: (text: string, images: string[]) => void;
  onSlash: (input: string) => boolean;
  onStop: () => void;
  onRegenerate: () => void;
  onEditMessage: (messageId: number, newText: string) => void;
  onNavigate: (messageId: number, dir: -1 | 1) => void;
  onChangeModel: (model: string) => void;
  onChangeEndpoint: (endpoint: string) => void;
  onOpenParams: () => void;
  onOpenArtifact?: (code: string, lang: string) => void;
  artifactCount: number;
  artifactOpen: boolean;
  onToggleArtifacts: () => void;
  presets: PresetDto[];
  onApplyPreset: (p: PresetDto) => void;
  onSavePreset: (name: string) => void;
  onDeletePreset: (id: number) => void;
}

function runtimeBadge(endpoint: string): string {
  if (/openrouter\.ai/.test(endpoint)) return "OPENROUTER";
  if (/:42069/.test(endpoint)) return "OMLX";
  if (/:8000/.test(endpoint)) return "OMLX";
  if (/:8080/.test(endpoint)) return "LLAMA.CPP";
  return "OPENAI";
}

const IMG_RE = /\.(png|jpe?g|webp|gif)$/i;
const TEXT_RE = /\.(txt|md|rs|ts|tsx|js|jsx|py|json|toml|yaml|yml|c|cpp|h|go|sh|css|html)$/i;

function readDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function readText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsText(file);
  });
}

export function ChatView(props: Props) {
  const { input, onInputChange } = props;
  const [attachments, setAttachments] = useState<string[]>([]);
  const [selectedPreset, setSelectedPreset] = useState("");
  const [findOpen, setFindOpen] = useState(false);
  const [findQuery, setFindQuery] = useState("");
  const [matchPos, setMatchPos] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const findRef = useRef<HTMLInputElement>(null);
  const atBottom = useRef(true);

  const matches = useMemo(() => {
    const q = findQuery.trim().toLowerCase();
    if (!q) return [];
    const out: number[] = [];
    props.messages.forEach((m, i) => {
      if (m.text.toLowerCase().includes(q)) out.push(i);
    });
    return out;
  }, [findQuery, props.messages]);
  const matchSet = useMemo(() => new Set(matches), [matches]);

  // Cmd/Ctrl+F toggles find; Esc closes it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "f") {
        e.preventDefault();
        setFindOpen((v) => !v);
        setTimeout(() => findRef.current?.focus(), 0);
      } else if (e.key === "Escape" && findOpen) {
        setFindOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [findOpen]);

  // Scroll the active match into view.
  useEffect(() => {
    if (!findOpen || matches.length === 0) return;
    const idx = matches[Math.min(matchPos, matches.length - 1)];
    const el = scrollRef.current?.querySelector(`[data-find-idx="${idx}"]`);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [matchPos, matches, findOpen]);

  const stepMatch = (dir: 1 | -1) =>
    setMatchPos((p) =>
      matches.length ? (p + dir + matches.length) % matches.length : 0,
    );

  const ingestFiles = async (files: FileList | File[]) => {
    for (const file of Array.from(files)) {
      if (IMG_RE.test(file.name) || file.type.startsWith("image/")) {
        const url = await readDataUrl(file);
        setAttachments((a) => [...a, url]);
      } else if (TEXT_RE.test(file.name) || file.type.startsWith("text/")) {
        const text = await readText(file);
        const fence = "```";
        onInputChange(
          (v) => `${v}${v ? "\n\n" : ""}${fence} ${file.name}\n${text}\n${fence}\n`,
        );
      }
    }
  };

  // autoscroll while near bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (el && atBottom.current) el.scrollTop = el.scrollHeight;
  }, [props.messages]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  const autoGrow = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 220) + "px";
  };

  const resetHeight = () =>
    requestAnimationFrame(() => {
      if (taRef.current) taRef.current.style.height = "auto";
    });

  const send = () => {
    if (props.streaming) return;
    // Intercept slash commands before sending.
    if (input.trim().startsWith("/") && props.onSlash(input)) {
      onInputChange("");
      resetHeight();
      return;
    }
    const t = input.trim();
    if (!t && attachments.length === 0) return;
    props.onSend(t, attachments);
    onInputChange("");
    setAttachments([]);
    resetHeight();
  };

  const endpoints = props.config.saved_endpoints.map((e) => e.url);

  return (
    <>
      <div className="topbar">
        <span className="badge dot">{runtimeBadge(props.settings.endpoint)}</span>
        <select
          className="model-select"
          value={props.settings.model ?? ""}
          onChange={(e) => props.onChangeModel(e.target.value)}
        >
          {props.settings.model &&
            !props.models.includes(props.settings.model) && (
              <option value={props.settings.model}>{props.settings.model}</option>
            )}
          {props.models.length === 0 && <option value="">no models</option>}
          {props.models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <select
          className="model-select"
          value={props.settings.endpoint}
          onChange={(e) => props.onChangeEndpoint(e.target.value)}
          title="Endpoint"
        >
          {!endpoints.includes(props.settings.endpoint) && (
            <option value={props.settings.endpoint}>{props.settings.endpoint}</option>
          )}
          {endpoints.map((u) => (
            <option key={u} value={u}>
              {u}
            </option>
          ))}
        </select>
        <div className="spacer" />
        <select
          className="model-select"
          value={selectedPreset}
          title="Presets"
          onChange={(e) => {
            const val = e.target.value;
            if (val === "__save__") {
              const name = window.prompt("Preset name");
              if (name?.trim()) props.onSavePreset(name.trim());
              setSelectedPreset("");
            } else if (val) {
              const p = props.presets.find((x) => String(x.id) === val);
              if (p) props.onApplyPreset(p);
              setSelectedPreset(val);
            }
          }}
        >
          <option value="">Presets…</option>
          {props.presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
          <option value="__save__">＋ Save current…</option>
        </select>
        {selectedPreset && selectedPreset !== "__save__" && (
          <button
            className="tbtn danger"
            title="Delete preset"
            onClick={() => {
              props.onDeletePreset(Number(selectedPreset));
              setSelectedPreset("");
            }}
          >
            ✕
          </button>
        )}
        {props.artifactCount > 0 && (
          <button
            className={`tbtn${props.artifactOpen ? " accent" : ""}`}
            title="Artifacts"
            onClick={props.onToggleArtifacts}
          >
            ◧ {props.artifactCount}
          </button>
        )}
        <button className="tbtn" onClick={props.onOpenParams}>
          params
        </button>
      </div>

      {findOpen && (
        <div className="find-bar">
          <input
            ref={findRef}
            placeholder="Find in conversation…"
            value={findQuery}
            onChange={(e) => {
              setFindQuery(e.target.value);
              setMatchPos(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") stepMatch(e.shiftKey ? -1 : 1);
              if (e.key === "Escape") setFindOpen(false);
            }}
          />
          <span className="find-count">
            {matches.length ? `${matchPos + 1}/${matches.length}` : "0/0"}
          </span>
          <button onClick={() => stepMatch(-1)} disabled={!matches.length}>
            ↑
          </button>
          <button onClick={() => stepMatch(1)} disabled={!matches.length}>
            ↓
          </button>
          <button onClick={() => setFindOpen(false)}>✕</button>
        </div>
      )}

      <div className="chat-scroll" ref={scrollRef} onScroll={onScroll}>
        <div className="chat-inner">
          {props.messages.length === 0 ? (
            <div className="empty" style={{ height: "60vh" }}>
              <div>
                <div className="glyph">✦</div>
                <div>Start a conversation</div>
              </div>
            </div>
          ) : (
            props.messages.map((m, i) => {
              const lastAssistant =
                m.role === "assistant" &&
                i === props.messages.length - 1 &&
                !m.streaming &&
                !props.streaming;
              const cls =
                findOpen && matches.length
                  ? matches[matchPos] === i
                    ? "find-current"
                    : matchSet.has(i)
                      ? "find-match"
                      : ""
                  : "";
              return (
                <div key={m.id ?? `live-${i}`} data-find-idx={i} className={cls}>
                  <MessageItem
                    msg={m}
                    onOpenArtifact={props.onOpenArtifact}
                    sibling={m.id != null ? props.siblingMap[m.id] : undefined}
                    onNavigate={
                      m.id != null
                        ? (dir) => props.onNavigate(m.id as number, dir)
                        : undefined
                    }
                    onEdit={
                      m.role === "user" && m.id != null && !props.streaming
                        ? (text) => props.onEditMessage(m.id as number, text)
                        : undefined
                    }
                    onRegenerate={lastAssistant ? props.onRegenerate : undefined}
                  />
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="composer">
        <div className="composer-inner">
          {props.pendingApproval && (
            <ApprovalCard
              approval={props.pendingApproval}
              onResolve={props.onResolveTool}
            />
          )}
          <div
            className="composer-box"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files.length) ingestFiles(e.dataTransfer.files);
            }}
          >
            {attachments.length > 0 && (
              <div className="attach-row">
                {attachments.map((src, i) => (
                  <div className="attach-thumb" key={i}>
                    <img src={src} />
                    <button
                      onClick={() =>
                        setAttachments((a) => a.filter((_, j) => j !== i))
                      }
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
            <textarea
              ref={taRef}
              rows={1}
              placeholder="Message…  (Enter to send, Shift+Enter for newline)"
              value={input}
              onChange={(e) => {
                onInputChange(e.target.value);
                autoGrow();
              }}
              onPaste={(e) => {
                const files = Array.from(e.clipboardData.files);
                if (files.length) {
                  e.preventDefault();
                  ingestFiles(files);
                }
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <div className="composer-row">
              <input
                ref={fileRef}
                type="file"
                multiple
                style={{ display: "none" }}
                onChange={(e) => {
                  if (e.target.files) ingestFiles(e.target.files);
                  e.target.value = "";
                }}
              />
              <button
                className="attach-btn"
                title="Attach files"
                onClick={() => fileRef.current?.click()}
              >
                ＋
              </button>
              <span className="hint">{props.settings.model ?? "select a model"}</span>
              {props.tokenCount > 0 && (
                <span className="hint token-count">{props.tokenCount} tok</span>
              )}
              {props.streaming ? (
                <button className="stop-btn" onClick={props.onStop}>
                  ■ Stop
                </button>
              ) : (
                <button
                  className="send-btn"
                  onClick={send}
                  disabled={
                    (!input.trim() && attachments.length === 0) ||
                    !props.settings.model
                  }
                >
                  Send
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
