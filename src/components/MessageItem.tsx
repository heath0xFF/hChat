import { useState } from "react";
import type { SiblingInfo } from "../types";
import { parseThink } from "../lib/segments";
import { Markdown } from "./Markdown";

export interface ToolCallChip {
  id: string;
  name: string;
  arguments: string;
}

export interface ChatMessage {
  id: number | null;
  role: "system" | "user" | "assistant" | "tool";
  text: string;
  images?: string[];
  streaming?: boolean;
  toolCalls?: ToolCallChip[];
  toolName?: string;
  isError?: boolean;
}

interface Props {
  msg: ChatMessage;
  onOpenArtifact?: (code: string, lang: string) => void;
  sibling?: SiblingInfo;
  onNavigate?: (dir: -1 | 1) => void;
  onEdit?: (newText: string) => void;
  onRegenerate?: () => void;
}

function NavControls({
  sibling,
  onNavigate,
}: {
  sibling?: SiblingInfo;
  onNavigate?: (dir: -1 | 1) => void;
}) {
  if (!sibling || sibling.total <= 1 || !onNavigate) return null;
  return (
    <span className="branch-nav">
      <button
        disabled={sibling.index <= 0}
        onClick={() => onNavigate(-1)}
        title="Previous version"
      >
        ◀
      </button>
      <span className="branch-count">
        {sibling.index + 1}/{sibling.total}
      </span>
      <button
        disabled={sibling.index >= sibling.total - 1}
        onClick={() => onNavigate(1)}
        title="Next version"
      >
        ▶
      </button>
    </span>
  );
}

const AVATAR: Record<string, string> = {
  user: "U",
  assistant: "AI",
  tool: "fn",
  system: "S",
};

export function MessageItem({
  msg,
  onOpenArtifact,
  sibling,
  onNavigate,
  onEdit,
  onRegenerate,
}: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  if (msg.role === "tool") {
    return (
      <div className="msg tool">
        <div className="avatar">fn</div>
        <div className="body">
          <div className="who">
            {msg.toolName ?? "tool"} {msg.isError ? "· error" : "· result"}
          </div>
          <details className="think" open={false}>
            <summary>{msg.isError ? "error output" : "result"}</summary>
            <div className="think-body">
              <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                <code>{msg.text}</code>
              </pre>
            </div>
          </details>
        </div>
      </div>
    );
  }

  if (msg.role === "user") {
    return (
      <div className="msg user">
        <div className="avatar">U</div>
        <div className="body">
          <div className="who">you</div>
          {msg.images && msg.images.length > 0 && (
            <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
              {msg.images.map((src, i) => (
                <img key={i} src={src} style={{ maxHeight: 160, borderRadius: 8 }} />
              ))}
            </div>
          )}
          {editing ? (
            <div className="edit-box">
              <textarea
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    onEdit?.(draft);
                    setEditing(false);
                  }
                  if (e.key === "Escape") setEditing(false);
                }}
              />
              <div className="edit-actions">
                <button
                  className="ap-approve"
                  onClick={() => {
                    onEdit?.(draft);
                    setEditing(false);
                  }}
                >
                  Save &amp; resend
                </button>
                <button className="ap-deny" onClick={() => setEditing(false)}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="md" style={{ whiteSpace: "pre-wrap" }}>
              {msg.text}
            </div>
          )}
          {!editing && (onEdit || (sibling && sibling.total > 1)) && (
            <div className="msg-actions">
              {onEdit && msg.id != null && (
                <button
                  onClick={() => {
                    setDraft(msg.text);
                    setEditing(true);
                  }}
                >
                  edit
                </button>
              )}
              <NavControls sibling={sibling} onNavigate={onNavigate} />
            </div>
          )}
        </div>
      </div>
    );
  }

  // assistant
  const { reasoning, reasoningOpen, body } = parseThink(msg.text);
  const showCursor = msg.streaming && body.length === 0 && !reasoning;

  return (
    <div className="msg assistant">
      <div className="avatar">{AVATAR.assistant}</div>
      <div className="body">
        <div className="who">assistant</div>
        {reasoning !== null && reasoning.trim().length > 0 && (
          <details className="think" open={reasoningOpen}>
            <summary>reasoning{reasoningOpen ? " …" : ""}</summary>
            <div className="think-body" style={{ whiteSpace: "pre-wrap" }}>
              {reasoning}
            </div>
          </details>
        )}
        {body.length > 0 && (
          <div className={msg.streaming ? "cursor" : ""}>
            <Markdown text={body} onOpenArtifact={onOpenArtifact} />
          </div>
        )}
        {msg.toolCalls && msg.toolCalls.length > 0 && (
          <div className="tool-chips">
            {msg.toolCalls.map((tc) => (
              <span className="tool-chip" key={tc.id} title={tc.arguments}>
                <span className="fn-ico">ƒ</span>
                {tc.name}
                <span className="chip-args">{shortArgs(tc.arguments)}</span>
              </span>
            ))}
          </div>
        )}
        {showCursor && <span className="cursor" />}
        {!msg.streaming && (onRegenerate || (sibling && sibling.total > 1)) && (
          <div className="msg-actions">
            {onRegenerate && (
              <button onClick={onRegenerate} title="Regenerate">
                ↻ regenerate
              </button>
            )}
            <NavControls sibling={sibling} onNavigate={onNavigate} />
          </div>
        )}
      </div>
    </div>
  );
}

function shortArgs(json: string): string {
  try {
    const obj = JSON.parse(json);
    const s = Object.entries(obj)
      .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
      .join(", ");
    return s.length > 60 ? s.slice(0, 60) + "…" : s;
  } catch {
    return "";
  }
}
