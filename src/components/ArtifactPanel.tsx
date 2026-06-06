import { useEffect, useState } from "react";
import type { Artifact } from "../lib/artifacts";
import { isPreviewable } from "../lib/artifacts";
import { CodeBlock } from "./CodeBlock";
import { Markdown } from "./Markdown";

interface Props {
  artifact: Artifact;
  artifacts: Artifact[];
  onSelect: (id: string) => void;
  onClose: () => void;
}

export function ArtifactPanel({ artifact, artifacts, onSelect, onClose }: Props) {
  const previewable = isPreviewable(artifact);
  const [view, setView] = useState<"preview" | "code">(
    previewable ? "preview" : "code",
  );

  // Reset the view when switching to a different artifact.
  useEffect(() => {
    setView(isPreviewable(artifact) ? "preview" : "code");
  }, [artifact.id, artifact.kind]);

  return (
    <div className="artifact-panel">
      <div className="artifact-head">
        <select
          className="artifact-select"
          value={artifact.id}
          onChange={(e) => onSelect(e.target.value)}
        >
          {artifacts.map((a) => (
            <option key={a.id} value={a.id}>
              {a.title}
              {a.lang && a.kind === "code" ? "" : ` · ${a.kind}`}
            </option>
          ))}
        </select>
        <div className="artifact-actions">
          {previewable && (
            <div className="seg">
              <button
                className={view === "preview" ? "on" : ""}
                onClick={() => setView("preview")}
              >
                preview
              </button>
              <button
                className={view === "code" ? "on" : ""}
                onClick={() => setView("code")}
              >
                code
              </button>
            </div>
          )}
          <button className="icon-btn" title="Close" onClick={onClose}>
            ✕
          </button>
        </div>
      </div>

      <div className="artifact-body">
        {view === "code" || !previewable ? (
          <div style={{ padding: 14 }}>
            <CodeBlock code={artifact.code} lang={artifact.lang || "text"} />
          </div>
        ) : artifact.kind === "html" ? (
          <iframe
            className="artifact-frame"
            title={artifact.title}
            sandbox="allow-scripts allow-forms allow-modals"
            srcDoc={artifact.code}
          />
        ) : artifact.kind === "svg" ? (
          <div
            className="artifact-svg"
            dangerouslySetInnerHTML={{ __html: artifact.code }}
          />
        ) : (
          <div className="artifact-md">
            <Markdown text={artifact.code} />
          </div>
        )}
      </div>
    </div>
  );
}
