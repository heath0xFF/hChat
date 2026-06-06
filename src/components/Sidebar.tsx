import { useState } from "react";
import type { ConversationDto, ProjectDto } from "../types";

export type View = "status" | "usage" | "models" | "chat";

interface Props {
  view: View;
  setView: (v: View) => void;
  conversations: ConversationDto[];
  projects: ProjectDto[];
  activeConvId: number | null;
  onSelectConv: (id: number) => void;
  onNewChat: () => void;
  onSearch: (q: string) => void;
  onPin: (id: number, pinned: boolean) => void;
  onRename: (id: number, title: string) => void;
  onDelete: (id: number) => void;
  onCreateProject: () => void;
  onRenameProject: (id: number, name: string) => void;
  onDeleteProject: (id: number) => void;
  onPinProject: (id: number, pinned: boolean) => void;
  onMoveConv: (convId: number, projectId: number | null) => void;
  onOpenSettings: () => void;
}

const NAV: { key: View; label: string; ico: string }[] = [
  { key: "status", label: "Status", ico: "▤" },
  { key: "usage", label: "Usage", ico: "◷" },
  { key: "models", label: "Models", ico: "◇" },
  { key: "chat", label: "Chat", ico: "✦" },
];

const RT_BADGE: Record<string, string> = {
  omlx: "ML",
  llamaswap: "LS",
  llamacpp: "LC",
  vllm: "VL",
  openai: "AI",
};

export function Sidebar(props: Props) {
  const [renaming, setRenaming] = useState<number | null>(null);
  const [renameBuf, setRenameBuf] = useState("");
  const [renamingProject, setRenamingProject] = useState<number | null>(null);
  const [projBuf, setProjBuf] = useState("");
  const [collapsed, setCollapsed] = useState<Set<number>>(new Set());
  const [dropTarget, setDropTarget] = useState<number | "loose" | null>(null);

  const commitRename = (id: number) => {
    if (renameBuf.trim()) props.onRename(id, renameBuf.trim());
    setRenaming(null);
  };
  const commitProjectRename = (id: number) => {
    if (projBuf.trim()) props.onRenameProject(id, projBuf.trim());
    setRenamingProject(null);
  };
  const toggleCollapsed = (id: number) =>
    setCollapsed((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  const byProject = (pid: number | null) =>
    props.conversations.filter((c) => (c.project_id ?? null) === pid);

  const drop = (e: React.DragEvent, target: number | null) => {
    e.preventDefault();
    setDropTarget(null);
    const id = Number(e.dataTransfer.getData("text/conv"));
    if (id) props.onMoveConv(id, target);
  };

  const renderConv = (c: ConversationDto) => (
    <div
      key={c.id}
      className={`conv-item ${
        props.activeConvId === c.id && props.view === "chat" ? "active" : ""
      }`}
      draggable={renaming !== c.id}
      onDragStart={(e) => e.dataTransfer.setData("text/conv", String(c.id))}
      onClick={() => props.onSelectConv(c.id)}
      onDoubleClick={() => {
        setRenaming(c.id);
        setRenameBuf(c.title);
      }}
    >
      {renaming === c.id ? (
        <input
          autoFocus
          className="rail-inline-input"
          value={renameBuf}
          onChange={(e) => setRenameBuf(e.target.value)}
          onBlur={() => commitRename(c.id)}
          onKeyDown={(e) => {
            if (e.key === "Enter") commitRename(c.id);
            if (e.key === "Escape") setRenaming(null);
          }}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <>
          {c.runtime && (
            <span className={`conv-rt rt-${c.runtime}`} title={c.runtime}>
              {RT_BADGE[c.runtime] ?? "··"}
            </span>
          )}
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
  );

  return (
    <div className="rail">
      <div className="rail-search">
        <input placeholder="Search…" onChange={(e) => props.onSearch(e.target.value)} />
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

      <div className="conv-list">
        {/* Projects */}
        <div className="rail-section">
          <span>Projects</span>
          <span
            className="icon-btn"
            title="New project"
            style={{ cursor: "pointer" }}
            onClick={props.onCreateProject}
          >
            +
          </span>
        </div>
        {props.projects.map((p) => {
          const convs = byProject(p.id);
          const isCollapsed = collapsed.has(p.id);
          return (
            <div key={p.id} className="project-group">
              <div
                className={`project-header ${dropTarget === p.id ? "drop" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDropTarget(p.id);
                }}
                onDragLeave={() => setDropTarget(null)}
                onDrop={(e) => drop(e, p.id)}
                onClick={() => toggleCollapsed(p.id)}
                onDoubleClick={() => {
                  setRenamingProject(p.id);
                  setProjBuf(p.name);
                }}
              >
                <span className="proj-caret">{isCollapsed ? "▸" : "▾"}</span>
                {renamingProject === p.id ? (
                  <input
                    autoFocus
                    className="rail-inline-input"
                    value={projBuf}
                    onChange={(e) => setProjBuf(e.target.value)}
                    onBlur={() => commitProjectRename(p.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitProjectRename(p.id);
                      if (e.key === "Escape") setRenamingProject(null);
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <>
                    <span className="proj-name">{p.name}</span>
                    <span className="proj-count">{convs.length}</span>
                    <span
                      className={`pin ${p.pinned ? "on" : ""}`}
                      title={p.pinned ? "Unpin" : "Pin"}
                      onClick={(e) => {
                        e.stopPropagation();
                        props.onPinProject(p.id, !p.pinned);
                      }}
                    >
                      ★
                    </span>
                    <span
                      className="pin"
                      title="Delete project"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Delete project "${p.name}"? Its chats are kept.`))
                          props.onDeleteProject(p.id);
                      }}
                    >
                      ✕
                    </span>
                  </>
                )}
              </div>
              {!isCollapsed && (
                <div className="project-chats">{convs.map(renderConv)}</div>
              )}
            </div>
          );
        })}

        {/* Loose conversations */}
        <div
          className={`rail-section drop-zone ${dropTarget === "loose" ? "drop" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDropTarget("loose");
          }}
          onDragLeave={() => setDropTarget(null)}
          onDrop={(e) => drop(e, null)}
        >
          <span>Conversations</span>
          <span
            className="icon-btn"
            title="New chat"
            style={{ cursor: "pointer" }}
            onClick={props.onNewChat}
          >
            +
          </span>
        </div>
        {byProject(null).map(renderConv)}
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
