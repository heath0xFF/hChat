import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/tauri";
import type { Endpoint } from "../types";

interface Props {
  endpoints: Endpoint[];
  activeEndpoint: string | null;
  activeModel: string | null;
  onUseModel: (endpoint: string, model: string) => void;
}

type EpState =
  | { status: "loading" }
  | { status: "ok"; models: string[] }
  | { status: "error"; error: string };

function hostOf(url: string): string {
  try {
    return new URL(url).host || url;
  } catch {
    return url;
  }
}

const RT_BADGE: Record<string, string> = {
  omlx: "ML",
  llamaswap: "LS",
  llamacpp: "LC",
  vllm: "VL",
  openai: "AI",
};

export function ModelsView({
  endpoints,
  activeEndpoint,
  activeModel,
  onUseModel,
}: Props) {
  const [states, setStates] = useState<Record<string, EpState>>({});

  const refresh = useCallback(async () => {
    setStates(
      Object.fromEntries(endpoints.map((e) => [e.url, { status: "loading" }])),
    );
    await Promise.all(
      endpoints.map(async (e) => {
        try {
          const models = await api.fetchModels(e.url);
          setStates((s) => ({ ...s, [e.url]: { status: "ok", models } }));
        } catch (err) {
          setStates((s) => ({
            ...s,
            [e.url]: { status: "error", error: String(err) },
          }));
        }
      }),
    );
  }, [endpoints]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <div className="status">
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          margin: "8px 0 6px",
        }}
      >
        <h1 style={{ margin: 0 }}>Models</h1>
        <div style={{ flex: 1 }} />
        <button className="tbtn" onClick={() => void refresh()}>
          refresh
        </button>
      </div>
      <div className="sub" style={{ margin: "0 0 14px" }}>
        Available across your saved endpoints. Click a model to use it in the
        current chat.
      </div>

      {endpoints.map((e) => {
        const st = states[e.url];
        const badge = RT_BADGE[e.runtime ?? "openai"] ?? "AI";
        return (
          <div className="model-group" key={e.url}>
            <div className="model-group-head">
              <span className={`rt-badge rt-${e.runtime ?? "openai"}`}>
                {badge}
              </span>
              <span className="model-group-host">{hostOf(e.url)}</span>
              <span className="model-group-count">
                {st?.status === "ok"
                  ? `${st.models.length} model${st.models.length === 1 ? "" : "s"}`
                  : st?.status === "loading"
                    ? "…"
                    : ""}
              </span>
            </div>

            {(!st || st.status === "loading") && (
              <div className="model-note">Loading…</div>
            )}
            {st?.status === "error" && (
              <div className="model-note model-note-err">
                Unreachable — {st.error}
              </div>
            )}
            {st?.status === "ok" && st.models.length === 0 && (
              <div className="model-note">
                No models loaded (server is up but nothing is loaded/pulled).
              </div>
            )}
            {st?.status === "ok" && st.models.length > 0 && (
              <div className="model-list">
                {st.models.map((m) => {
                  const active =
                    e.url === activeEndpoint && m === activeModel;
                  return (
                    <button
                      key={m}
                      className={`model-row${active ? " active" : ""}`}
                      onClick={() => onUseModel(e.url, m)}
                      title={`Use ${m}`}
                    >
                      <span className="model-name">{m}</span>
                      {active && <span className="model-active">active</span>}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}

      {endpoints.length === 0 && (
        <div className="model-note">
          No endpoints configured — add one in Settings → Endpoints.
        </div>
      )}
    </div>
  );
}
