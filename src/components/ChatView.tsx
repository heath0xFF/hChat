import { useEffect, useRef, useState } from "react";
import type { Config, SettingsDto } from "../types";
import type { ChatMessage } from "./MessageItem";
import { MessageItem } from "./MessageItem";

interface Props {
  config: Config;
  settings: SettingsDto;
  models: string[];
  messages: ChatMessage[];
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
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

export function ChatView(props: Props) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const atBottom = useRef(true);

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
    if (!t || props.streaming) return;
    props.onSend(t);
    setInput("");
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
            props.messages.map((m, i) => (
              <MessageItem
                key={m.id ?? `live-${i}`}
                msg={m}
                onOpenArtifact={props.onOpenArtifact}
              />
            ))
          )}
        </div>
      </div>

      <div className="composer">
        <div className="composer-inner">
          <div className="composer-box">
            <textarea
              ref={taRef}
              rows={1}
              placeholder="Message…  (Enter to send, Shift+Enter for newline)"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                autoGrow();
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <div className="composer-row">
              <span className="hint">{props.settings.model ?? "select a model"}</span>
              {props.streaming ? (
                <button className="stop-btn" onClick={props.onStop}>
                  ■ Stop
                </button>
              ) : (
                <button
                  className="send-btn"
                  onClick={send}
                  disabled={!input.trim() || !props.settings.model}
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
