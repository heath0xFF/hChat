import type { ReactNode } from "react";

export type DockTab = "status" | "artifacts";

interface Props {
  tab: DockTab;
  onTab: (t: DockTab) => void;
  onClose: () => void;
  hasArtifacts: boolean;
  artifactCount: number;
  status: ReactNode;
  artifacts: ReactNode;
}

/**
 * The collapsible right-side dock that hosts the live Status metrics and the
 * Artifacts preview as switchable tabs (vllm-studio's right panel), so you can
 * watch a backend or read an artifact without leaving the chat.
 */
export function RightDock({
  tab,
  onTab,
  onClose,
  hasArtifacts,
  artifactCount,
  status,
  artifacts,
}: Props) {
  return (
    <aside className="dock">
      <div className="dock-tabs">
        <button
          className={`dock-tab${tab === "status" ? " active" : ""}`}
          onClick={() => onTab("status")}
        >
          ⚡ Status
        </button>
        <button
          className={`dock-tab${tab === "artifacts" ? " active" : ""}`}
          onClick={() => onTab("artifacts")}
          disabled={!hasArtifacts}
          title={hasArtifacts ? "Artifacts" : "No artifacts yet"}
        >
          ◧ Artifacts{artifactCount > 0 ? ` ${artifactCount}` : ""}
        </button>
        <div style={{ flex: 1 }} />
        <button className="dock-close" title="Close panel" onClick={onClose}>
          ✕
        </button>
      </div>
      <div className="dock-body">
        {tab === "status" ? status : artifacts}
      </div>
    </aside>
  );
}
