import { useEffect, useRef, useState } from "react";
import type { Config, PendingApproval, SettingsDto, SiblingInfo } from "../types";
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
  onResolveTool: (approved: boolean) => void;
  onSend: (text: string, images: string[]) => void;
  onStop: () => void;
  onRegenerate: () => void;
  onEditMessage: (messageId: number, newText: string) => void;
  onNavigate: (messageId: number, dir: -1 | 1) => void;
  onChangeModel: (model: string) => void;
  onChangeEndpoint: (endpoint: string) => void;
  onOpenParams: () => void;
  onOpenArtifact?: (code: string, lang: string) => void;
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
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const atBottom = useRef(true);

  const ingestFiles = async (files: FileList | File[]) => {
    for (const file of Array.from(files)) {
      if (IMG_RE.test(file.name) || file.type.startsWith("image/")) {
        const url = await readDataUrl(file);
        setAttachments((a) => [...a, url]);
      } else if (TEXT_RE.test(file.name) || file.type.startsWith("text/")) {
        const text = await readText(file);
        const fence = "```";
        setInput(
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

  const send = () => {
    const t = input.trim();
    if ((!t && attachments.length === 0) || props.streaming) return;
    props.onSend(t, attachments);
    setInput("");
    setAttachments([]);
    requestAnimationFrame(() => {
      if (taRef.current) taRef.current.style.height = "auto";
    });
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
        <button className="tbtn" onClick={props.onOpenParams}>
          params
        </button>
      </div>

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
              return (
                <MessageItem
                  key={m.id ?? `live-${i}`}
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
                setInput(e.target.value);
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
