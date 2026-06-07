import { useEffect, useState } from "react";
import type { ConversationDto, ProjectDto } from "../types";

export type View = "status" | "usage" | "models" | "chat";

interface Props {
  view: View;
  setView: (v: View) => void;
  conversations: ConversationDto[];
  projects: ProjectDto[];
  activeConvId: number | null;
  // Conversation ids with a stream in flight — shows a live indicator.
  streamingIds: number[];
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
  // "Move to project" context menu, opened by the inline ▤ button or right-click.
  const [menu, setMenu] = useState<{
    conv: ConversationDto;
    x: number;
    y: number;
  } | null>(null);

  const openMenu = (e: React.MouseEvent, c: ConversationDto) => {
    e.preventDefault();
    e.stopPropagation(); // don't let this click reach the close-on-outside handler
    setMenu({ conv: c, x: e.clientX, y: e.clientY });
  };

  // Close the menu on any outside click or Esc.
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenu(null);
    };
    window.addEventListener("click", close);
    window.addEventListener("contextmenu", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("contextmenu", close);
      window.removeEventListener("keydown", onKey);
    };
  }, [menu]);

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
      onContextMenu={(e) => {
        if (renaming === c.id) return; // keep the native menu while editing
        openMenu(e, c);
      }}
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
          {props.streamingIds.includes(c.id) ? (
            <span className="conv-streaming" title="Generating…" />
          ) : (
            c.runtime && (
              <span className={`conv-rt rt-${c.runtime}`} title={c.runtime}>
                {RT_BADGE[c.runtime] ?? "··"}
              </span>
            )
          )}
          <span className="title">{c.title}</span>
          <span
            className="pin"
            title="Move to project"
            onClick={(e) => openMenu(e, c)}
          >
            ▤
          </span>
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
              props.onDelete(c.id);
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

      {menu &&
        (() => {
          // Clamp to the viewport so rows near the edges don't push it off-screen.
          const rows =
            props.projects.length +
            (menu.conv.project_id != null ? 1 : 0) +
            (props.projects.length === 0 ? 1 : 0);
          const estH = 34 + rows * 30 + 8;
          const left = Math.min(menu.x, window.innerWidth - 248);
          const top = Math.min(menu.y, window.innerHeight - estH);
          return (
            <div
              className="ctx-menu"
              style={{ left: Math.max(8, left), top: Math.max(8, top) }}
              onClick={(e) => e.stopPropagation()}
            >
          <div className="ctx-menu-label">Move to project</div>
          {props.projects.length === 0 && (
            <div className="ctx-menu-item disabled">No projects yet</div>
          )}
          {props.projects.map((p) => {
            const here = (menu.conv.project_id ?? null) === p.id;
            return (
              <div
                key={p.id}
                className={`ctx-menu-item${here ? " disabled" : ""}`}
                onClick={() => {
                  if (!here) props.onMoveConv(menu.conv.id, p.id);
                  setMenu(null);
                }}
              >
                <span className="ctx-menu-dot">{here ? "●" : ""}</span>
                {p.name}
              </div>
            );
          })}
          {menu.conv.project_id != null && (
            <>
              <div className="ctx-menu-sep" />
              <div
                className="ctx-menu-item"
                onClick={() => {
                  props.onMoveConv(menu.conv.id, null);
                  setMenu(null);
                }}
              >
                Remove from project
              </div>
            </>
          )}
            </div>
          );
        })()}
    </div>
  );
}
