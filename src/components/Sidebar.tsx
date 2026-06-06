import { useState } from "react";
import type { ConversationDto } from "../types";

export type View = "status" | "chat";

interface Props {
  view: View;
  setView: (v: View) => void;
  conversations: ConversationDto[];
  activeConvId: number | null;
  onSelectConv: (id: number) => void;
  onNewChat: () => void;
  onSearch: (q: string) => void;
  onPin: (id: number, pinned: boolean) => void;
  onRename: (id: number, title: string) => void;
  onDelete: (id: number) => void;
  onOpenSettings: () => void;
}

const NAV: { key: View; label: string; ico: string }[] = [
  { key: "status", label: "Status", ico: "▤" },
  { key: "chat", label: "Chat", ico: "✦" },
];

export function Sidebar(props: Props) {
  const [renaming, setRenaming] = useState<number | null>(null);
  const [renameBuf, setRenameBuf] = useState("");

  const commitRename = (id: number) => {
    if (renameBuf.trim()) props.onRename(id, renameBuf.trim());
    setRenaming(null);
  };

  return (
    <div className="rail">
      <div className="rail-search">
        <input
          placeholder="Search…"
          onChange={(e) => props.onSearch(e.target.value)}
        />
      </div>

      <div className="rail-section">Workspace</div>
      {NAV.map((n) => (
        <div
          key={n.key}
          className={`nav-item ${props.view === n.key ? "active" : ""}`}
          onClick={() => props.setView(n.key)}
        >
          <span className="ico">{n.ico}</span>
          {n.label}
        </div>
      ))}

      <div className="rail-section">
        <span>Conversations</span>
        <span
          className="icon-btn"
          title="New chat"
          onClick={props.onNewChat}
          style={{ cursor: "pointer" }}
        >
          +
        </span>
      </div>

      <div className="conv-list">
        {props.conversations.map((c) => (
          <div
            key={c.id}
            className={`conv-item ${
              props.activeConvId === c.id && props.view === "chat" ? "active" : ""
            }`}
            onClick={() => props.onSelectConv(c.id)}
            onDoubleClick={() => {
              setRenaming(c.id);
              setRenameBuf(c.title);
            }}
          >
            {renaming === c.id ? (
              <input
                autoFocus
                value={renameBuf}
                onChange={(e) => setRenameBuf(e.target.value)}
                onBlur={() => commitRename(c.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitRename(c.id);
                  if (e.key === "Escape") setRenaming(null);
                }}
                onClick={(e) => e.stopPropagation()}
                style={{
                  flex: 1,
                  background: "var(--panel-2)",
                  border: "1px solid var(--accent-dim)",
                  borderRadius: 5,
                  color: "var(--text)",
                  padding: "2px 6px",
                  fontSize: 13,
                  outline: "none",
                }}
              />
            ) : (
              <>
                <span className="title">{c.title}</span>
                <span
                  className={`pin ${c.pinned ? "on" : ""}`}
                  title={c.pinned ? "Unpin" : "Pin"}
                  onClick={(e) => {
                    e.stopPropagation();
                    props.onPin(c.id, !c.pinned);
                  }}
                >
                  ★
                </span>
                <span
                  className="pin"
                  title="Delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm(`Delete "${c.title}"?`)) props.onDelete(c.id);
                  }}
                >
                  ✕
                </span>
              </>
            )}
          </div>
        ))}
      </div>

      <div className="rail-footer">
        <div className="nav-item" onClick={props.onOpenSettings}>
          <span className="ico">⚙</span>
          Settings
        </div>
      </div>
    </div>
  );
}
