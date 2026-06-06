import { useState } from "react";
import type { Config, GpuKind, RuntimeKind } from "../types";
import { comboFromEvent, HOTKEY_ACTIONS } from "../lib/hotkeys";

interface Props {
  config: Config;
  onClose: () => void;
  onSave: (config: Config) => void;
}

type Section = "general" | "endpoints" | "generation" | "appearance" | "keyboard";

const SECTIONS: { key: Section; label: string }[] = [
  { key: "general", label: "General" },
  { key: "endpoints", label: "Endpoints" },
  { key: "generation", label: "Generation" },
  { key: "appearance", label: "Appearance" },
  { key: "keyboard", label: "Keyboard" },
];

const RUNTIMES: RuntimeKind[] = ["openai", "vllm", "omlx", "llamacpp", "llamaswap"];
const GPUS: GpuKind[] = ["none", "macmon", "agent"];

function HotkeyInput({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const [capturing, setCapturing] = useState(false);
  return (
    <input
      readOnly
      className={capturing ? "hk-capturing" : ""}
      value={capturing ? "press keys…" : value}
      onFocus={() => setCapturing(true)}
      onBlur={() => setCapturing(false)}
      onKeyDown={(e) => {
        e.preventDefault();
        if (e.key === "Escape") {
          (e.target as HTMLInputElement).blur();
          return;
        }
        const combo = comboFromEvent(e.nativeEvent);
        if (combo) {
          onChange(combo);
          (e.target as HTMLInputElement).blur();
        }
      }}
    />
  );
}

export function SettingsModal({ config, onClose, onSave }: Props) {
  const [c, setC] = useState<Config>(structuredClone(config));
  const [section, setSection] = useState<Section>("general");

  const set = <K extends keyof Config>(k: K, v: Config[K]) =>
    setC((prev) => ({ ...prev, [k]: v }));
  const numOrNull = (s: string): number | null =>
    s.trim() === "" ? null : Number(s);

  const updateEndpoint = (
    i: number,
    patch: Partial<Config["saved_endpoints"][number]>,
  ) => {
    const eps = c.saved_endpoints.slice();
    eps[i] = { ...eps[i], ...patch };
    set("saved_endpoints", eps);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal settings-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="settings-nav">
          <h2 style={{ marginBottom: 14 }}>Settings</h2>
          {SECTIONS.map((s) => (
            <div
              key={s.key}
              className={`settings-tab ${section === s.key ? "active" : ""}`}
              onClick={() => setSection(s.key)}
            >
              {s.label}
            </div>
          ))}
        </div>

        <div className="settings-main">
          <div className="settings-content">
          {section === "general" && (
            <>
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
                  rows={5}
                  value={c.system_prompt}
                  onChange={(e) => set("system_prompt", e.target.value)}
                />
              </div>
            </>
          )}

          {section === "endpoints" && (
            <div className="field">
              <label>Endpoints &amp; metrics</label>
              {c.saved_endpoints.map((ep, i) => (
                <div className="endpoint-card" key={i}>
                  <div className="endpoint-row">
                    <input
                      placeholder="http://localhost:42069/v1"
                      value={ep.url}
                      onChange={(e) => updateEndpoint(i, { url: e.target.value })}
                    />
                    <input
                      placeholder="api key (optional)"
                      type="password"
                      style={{ flex: "0 0 140px" }}
                      value={ep.api_key ?? ""}
                      onChange={(e) =>
                        updateEndpoint(i, {
                          api_key: e.target.value || null,
                        })
                      }
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
                  <div className="row" style={{ marginTop: 8 }}>
                    <div style={{ flex: 1 }}>
                      <label>Runtime</label>
                      <select
                        value={ep.runtime ?? "openai"}
                        onChange={(e) =>
                          updateEndpoint(i, {
                            runtime: e.target.value as RuntimeKind,
                          })
                        }
                      >
                        {RUNTIMES.map((r) => (
                          <option key={r} value={r}>
                            {r}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div style={{ flex: 1 }}>
                      <label>GPU source</label>
                      <select
                        value={ep.gpu ?? "none"}
                        onChange={(e) =>
                          updateEndpoint(i, { gpu: e.target.value as GpuKind })
                        }
                      >
                        {GPUS.map((g) => (
                          <option key={g} value={g}>
                            {g}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 8 }}>
                    <div style={{ flex: 1 }}>
                      <label>Prometheus URL</label>
                      <input
                        placeholder="http://host:8000/metrics"
                        value={ep.prometheus_url ?? ""}
                        onChange={(e) =>
                          updateEndpoint(i, {
                            prometheus_url: e.target.value || null,
                          })
                        }
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <label>Agent URL</label>
                      <input
                        placeholder="http://host:9099"
                        value={ep.agent_url ?? ""}
                        onChange={(e) =>
                          updateEndpoint(i, {
                            agent_url: e.target.value || null,
                          })
                        }
                      />
                    </div>
                  </div>
                </div>
              ))}
              <button
                className="tbtn"
                style={{ marginTop: 8 }}
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
          )}

          {section === "generation" && (
            <>
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
                    <label>Max tokens</label>
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
                <label>Stop sequences (max 4)</label>
                {c.stop_sequences.map((s, i) => (
                  <div className="endpoint-row" key={i}>
                    <input
                      value={s}
                      onChange={(e) => {
                        const arr = c.stop_sequences.slice();
                        arr[i] = e.target.value;
                        set("stop_sequences", arr);
                      }}
                    />
                    <button
                      className="icon-btn"
                      onClick={() =>
                        set(
                          "stop_sequences",
                          c.stop_sequences.filter((_, j) => j !== i),
                        )
                      }
                    >
                      ✕
                    </button>
                  </div>
                ))}
                {c.stop_sequences.length < 4 && (
                  <button
                    className="tbtn"
                    style={{ marginTop: 6 }}
                    onClick={() =>
                      set("stop_sequences", [...c.stop_sequences, ""])
                    }
                  >
                    + add stop sequence
                  </button>
                )}
              </div>
            </>
          )}

          {section === "appearance" && (
            <>
              <div className="field">
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
              </div>
              <div className="field">
                <div className="row">
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
                      onChange={(e) =>
                        set("mono_font_size", Number(e.target.value))
                      }
                    />
                  </div>
                </div>
              </div>
              <div className="field">
                <div className="row">
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
            </>
          )}

          {section === "keyboard" && (
            <div className="field">
              <label>Shortcuts (click a field, then press the keys)</label>
              <div className="hk-hint">
                <code>mod</code> = Cmd on macOS, Ctrl elsewhere.
              </div>
              {HOTKEY_ACTIONS.map((a) => (
                <div className="hk-row" key={a.key}>
                  <span className="hk-label">{a.label}</span>
                  <HotkeyInput
                    value={c.hotkeys[a.key]}
                    onChange={(v) =>
                      set("hotkeys", { ...c.hotkeys, [a.key]: v })
                    }
                  />
                </div>
              ))}
            </div>
          )}

          </div>
          <div className="settings-footer">
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
      </div>
    </div>
  );
}
