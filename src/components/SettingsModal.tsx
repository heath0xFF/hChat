import { useState } from "react";
import type { Config } from "../types";

interface Props {
  config: Config;
  onClose: () => void;
  onSave: (config: Config) => void;
}

export function SettingsModal({ config, onClose, onSave }: Props) {
  const [c, setC] = useState<Config>(structuredClone(config));

  const set = <K extends keyof Config>(k: K, v: Config[K]) =>
    setC((prev) => ({ ...prev, [k]: v }));

  const updateEndpoint = (i: number, field: "url" | "api_key", val: string) => {
    const eps = c.saved_endpoints.slice();
    eps[i] = { ...eps[i], [field]: field === "api_key" && !val ? null : val };
    set("saved_endpoints", eps);
  };

  const numOrNull = (s: string): number | null =>
    s.trim() === "" ? null : Number(s);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Settings</h2>

        <div className="field">
          <label>Endpoints</label>
          {c.saved_endpoints.map((ep, i) => (
            <div className="endpoint-row" key={i}>
              <input
                placeholder="http://localhost:42069/v1"
                value={ep.url}
                onChange={(e) => updateEndpoint(i, "url", e.target.value)}
              />
              <input
                placeholder="api key (optional)"
                value={ep.api_key ?? ""}
                type="password"
                style={{ flex: "0 0 150px" }}
                onChange={(e) => updateEndpoint(i, "api_key", e.target.value)}
              />
              <button
                className="icon-btn"
                onClick={() =>
                  set(
                    "saved_endpoints",
                    c.saved_endpoints.filter((_, j) => j !== i),
                  )
                }
              >
                ✕
              </button>
            </div>
          ))}
          <button
            className="tbtn"
            style={{ marginTop: 6 }}
            onClick={() =>
              set("saved_endpoints", [
                ...c.saved_endpoints,
                { url: "", api_key: null },
              ])
            }
          >
            + add endpoint
          </button>
        </div>

        <div className="field">
          <label>Default endpoint</label>
          <select
            value={c.default_endpoint}
            onChange={(e) => set("default_endpoint", e.target.value)}
          >
            {c.saved_endpoints.map((ep, i) => (
              <option key={i} value={ep.url}>
                {ep.url}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label>System prompt</label>
          <textarea
            rows={3}
            value={c.system_prompt}
            onChange={(e) => set("system_prompt", e.target.value)}
          />
        </div>

        <div className="field">
          <div className="row">
            <div style={{ flex: 1 }}>
              <label>Temperature</label>
              <input
                type="number"
                step="0.1"
                value={c.temperature}
                onChange={(e) => set("temperature", Number(e.target.value))}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label>Max tokens {c.use_max_tokens ? "" : "(off)"}</label>
              <input
                type="number"
                value={c.max_tokens}
                onChange={(e) => set("max_tokens", Number(e.target.value))}
              />
            </div>
            <div style={{ flex: "0 0 auto", paddingTop: 22 }}>
              <label style={{ textTransform: "none", letterSpacing: 0 }}>
                <input
                  type="checkbox"
                  checked={c.use_max_tokens}
                  onChange={(e) => set("use_max_tokens", e.target.checked)}
                />{" "}
                send
              </label>
            </div>
          </div>
        </div>

        <div className="field">
          <div className="row">
            <div style={{ flex: 1 }}>
              <label>top_p</label>
              <input
                type="number"
                step="0.05"
                value={c.top_p ?? ""}
                onChange={(e) => set("top_p", numOrNull(e.target.value))}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label>freq. penalty</label>
              <input
                type="number"
                step="0.1"
                value={c.frequency_penalty ?? ""}
                onChange={(e) =>
                  set("frequency_penalty", numOrNull(e.target.value))
                }
              />
            </div>
            <div style={{ flex: 1 }}>
              <label>pres. penalty</label>
              <input
                type="number"
                step="0.1"
                value={c.presence_penalty ?? ""}
                onChange={(e) =>
                  set("presence_penalty", numOrNull(e.target.value))
                }
              />
            </div>
          </div>
        </div>

        <div className="field">
          <label>Appearance</label>
          <div className="row" style={{ alignItems: "center" }}>
            <label
              style={{ textTransform: "none", letterSpacing: 0, margin: 0 }}
            >
              <input
                type="checkbox"
                checked={c.dark_mode}
                onChange={(e) => set("dark_mode", e.target.checked)}
              />{" "}
              dark mode
            </label>
            <div style={{ flex: 1 }}>
              <label>UI scale</label>
              <input
                type="number"
                step="0.05"
                min="0.75"
                max="2"
                value={c.ui_scale}
                onChange={(e) => set("ui_scale", Number(e.target.value))}
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <div style={{ flex: 1 }}>
              <label>Font size</label>
              <input
                type="number"
                value={c.font_size}
                onChange={(e) => set("font_size", Number(e.target.value))}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label>Mono size</label>
              <input
                type="number"
                value={c.mono_font_size}
                onChange={(e) => set("mono_font_size", Number(e.target.value))}
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <div style={{ flex: 1 }}>
              <label>Font family</label>
              <input
                placeholder="(default)"
                value={c.font_family}
                onChange={(e) => set("font_family", e.target.value)}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label>Mono family</label>
              <input
                placeholder="(default)"
                value={c.mono_font_family}
                onChange={(e) => set("mono_font_family", e.target.value)}
              />
            </div>
          </div>
        </div>

        <div className="modal-actions">
          <button className="tbtn" onClick={onClose}>
            Cancel
          </button>
          <button className="tbtn accent" onClick={() => onSave(c)}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
