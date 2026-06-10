import { useEffect, useState } from "react";
import type { Config, GpuKind, McpServer, McpStatus, RuntimeKind } from "../types";
import { comboFromEvent, HOTKEY_ACTIONS } from "../lib/hotkeys";
import { api } from "../lib/tauri";

interface Props {
  config: Config;
  onClose: () => void;
  onSave: (config: Config) => void;
  onClearHistory: () => void;
  /** Apply (and persist) an appearance change immediately, without Save. */
  onApplyAppearance: (patch: Partial<Config>) => void;
}

type Section =
  | "general"
  | "endpoints"
  | "generation"
  | "context"
  | "appearance"
  | "keyboard"
  | "mcp";

const SECTIONS: { key: Section; label: string }[] = [
  { key: "general", label: "General" },
  { key: "endpoints", label: "Endpoints" },
  { key: "generation", label: "Generation" },
  { key: "context", label: "Context" },
  { key: "appearance", label: "Appearance" },
  { key: "keyboard", label: "Keyboard" },
  { key: "mcp", label: "MCP" },
];

const THEMES: { key: string; label: string }[] = [
  { key: "dark", label: "Dark (default)" },
  { key: "light", label: "Light" },
  { key: "catppuccin-mocha", label: "Catppuccin Mocha" },
  { key: "catppuccin-macchiato", label: "Catppuccin Macchiato" },
  { key: "catppuccin-frappe", label: "Catppuccin Frappé" },
  { key: "catppuccin-latte", label: "Catppuccin Latte" },
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
          // Cancel hotkey capture without bubbling up to close the modal.
          e.stopPropagation();
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

export function SettingsModal({
  config,
  onClose,
  onSave,
  onClearHistory,
  onApplyAppearance,
}: Props) {
  const [c, setC] = useState<Config>(structuredClone(config));
  const [section, setSection] = useState<Section>("general");
  const [mcpStatus, setMcpStatus] = useState<McpStatus[]>([]);

  const refreshMcp = () =>
    api.listMcpServers().then(setMcpStatus).catch(() => setMcpStatus([]));
  useEffect(() => {
    void refreshMcp();
  }, []);

  // Esc closes the settings modal (hotkey-capture inputs swallow their own Esc).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const servers = c.mcp_servers ?? [];
  const setServers = (next: McpServer[]) => set("mcp_servers", next);
  const updateServer = (i: number, patch: Partial<McpServer>) => {
    const next = servers.slice();
    next[i] = { ...next[i], ...patch };
    setServers(next);
  };

  const set = <K extends keyof Config>(k: K, v: Config[K]) =>
    setC((prev) => ({ ...prev, [k]: v }));
  // Appearance fields update the local draft AND apply live (persisted on a
  // debounce by the parent), so they take effect without clicking Save.
  const setAppearance = <K extends keyof Config>(k: K, v: Config[K]) => {
    set(k, v);
    onApplyAppearance({ [k]: v } as Partial<Config>);
  };
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
              <div className="field">
                <label>Usage history retention (days)</label>
                <input
                  type="number"
                  min={0}
                  value={c.usage_retention_days ?? 0}
                  onChange={(e) =>
                    set(
                      "usage_retention_days",
                      Math.max(0, Math.floor(Number(e.target.value) || 0)),
                    )
                  }
                />
                <div className="field-hint">
                  Prune recorded usage older than this. 0 keeps everything
                  forever.
                </div>
              </div>
              <div className="field">
                <label>Danger zone</label>
                <button className="tbtn danger" onClick={onClearHistory}>
                  Delete all chat history
                </button>
                <div className="field-hint">
                  Permanently deletes every conversation and its messages.
                  Projects are kept. This cannot be undone.
                </div>
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
                  <div className="row" style={{ marginTop: 8 }}>
                    <div style={{ flex: 1 }}>
                      <label>Context window (tokens)</label>
                      <input
                        type="number"
                        placeholder="e.g. 131072 — enables auto-compaction"
                        value={ep.context_window ?? ""}
                        onChange={(e) =>
                          updateEndpoint(i, {
                            context_window: e.target.value
                              ? Number(e.target.value)
                              : null,
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

          {section === "context" && (
            <>
              <div className="field">
                <label style={{ textTransform: "none", letterSpacing: 0 }}>
                  <input
                    type="checkbox"
                    checked={c.auto_compact}
                    onChange={(e) => set("auto_compact", e.target.checked)}
                  />{" "}
                  Auto-compact long conversations
                </label>
                <div className="hint" style={{ marginTop: 4 }}>
                  When a chat nears its endpoint's context window, older messages
                  are summarized by the model and replaced with the summary.
                  Requires the endpoint's <strong>Context window</strong> to be set.
                </div>
              </div>
              <div className="field">
                <div className="row">
                  <div style={{ flex: 1 }}>
                    <label>Compact at (% of window)</label>
                    <input
                      type="number"
                      step="5"
                      min="30"
                      max="95"
                      value={Math.round(c.compact_threshold_pct * 100)}
                      onChange={(e) =>
                        set(
                          "compact_threshold_pct",
                          e.target.value
                            ? Number(e.target.value) / 100
                            : c.compact_threshold_pct,
                        )
                      }
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label>Keep recent messages</label>
                    <input
                      type="number"
                      step="1"
                      min="2"
                      max="64"
                      value={c.compact_keep_recent}
                      onChange={(e) =>
                        set(
                          "compact_keep_recent",
                          e.target.value
                            ? Number(e.target.value)
                            : c.compact_keep_recent,
                        )
                      }
                    />
                  </div>
                </div>
              </div>
              <div className="field">
                <label>Summarization prompt (optional)</label>
                <textarea
                  rows={5}
                  placeholder="Leave empty to use the built-in compaction prompt."
                  value={c.compact_prompt}
                  onChange={(e) => set("compact_prompt", e.target.value)}
                />
              </div>
            </>
          )}

          {section === "appearance" && (
            <>
              <div className="field">
                <div className="row">
                  <div style={{ flex: 1 }}>
                    <label>Theme</label>
                    <select
                      value={c.theme || (c.dark_mode ? "dark" : "light")}
                      onChange={(e) => setAppearance("theme", e.target.value)}
                    >
                      {THEMES.map((t) => (
                        <option key={t.key} value={t.key}>
                          {t.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div style={{ flex: 1 }}>
                    <label>UI scale</label>
                    <input
                      type="number"
                      step="0.05"
                      min="0.75"
                      max="2"
                      value={c.ui_scale}
                      onChange={(e) =>
                        setAppearance("ui_scale", Number(e.target.value))
                      }
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
                      onChange={(e) =>
                        setAppearance("font_size", Number(e.target.value))
                      }
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label>Mono size</label>
                    <input
                      type="number"
                      value={c.mono_font_size}
                      onChange={(e) =>
                        setAppearance("mono_font_size", Number(e.target.value))
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
                      onChange={(e) => setAppearance("font_family", e.target.value)}
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label>Mono family</label>
                    <input
                      placeholder="(default)"
                      value={c.mono_font_family}
                      onChange={(e) =>
                        setAppearance("mono_font_family", e.target.value)
                      }
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

          {section === "mcp" && (
            <div className="field">
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 6,
                }}
              >
                <label style={{ margin: 0 }}>MCP servers</label>
                <button
                  className="tbtn"
                  onClick={() => void api.reconnectMcp().then(refreshMcp)}
                >
                  Reconnect all
                </button>
              </div>
              <div className="hk-hint">
                Save to apply changes (servers reconnect automatically).{" "}
                <code>stdio</code> spawns a command; <code>http</code> connects to
                a streamable-HTTP URL.
              </div>
              {servers.map((srv, i) => {
                const st = mcpStatus.find((s) => s.name === srv.name);
                return (
                  <div className="endpoint-card" key={i}>
                    <div className="endpoint-row">
                      <input
                        placeholder="name"
                        style={{ flex: "0 0 130px" }}
                        value={srv.name}
                        onChange={(e) => updateServer(i, { name: e.target.value })}
                      />
                      <select
                        style={{ flex: "0 0 90px" }}
                        value={srv.transport}
                        onChange={(e) =>
                          updateServer(i, { transport: e.target.value })
                        }
                      >
                        <option value="stdio">stdio</option>
                        <option value="http">http</option>
                      </select>
                      <span
                        className="mcp-status"
                        title={st?.error ?? ""}
                        style={{ flex: 1, textAlign: "right" }}
                      >
                        {st
                          ? st.connected
                            ? `● ${st.tool_count} tools`
                            : `✕ ${st.error ?? "off"}`
                          : "—"}
                      </span>
                      <button
                        className="icon-btn"
                        onClick={() => setServers(servers.filter((_, j) => j !== i))}
                      >
                        ✕
                      </button>
                    </div>
                    {srv.transport === "http" ? (
                      <div className="row" style={{ marginTop: 8 }}>
                        <div style={{ flex: 1 }}>
                          <label>URL</label>
                          <input
                            placeholder="http://host:port/mcp"
                            value={srv.url ?? ""}
                            onChange={(e) =>
                              updateServer(i, { url: e.target.value || null })
                            }
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="row" style={{ marginTop: 8 }}>
                        <div style={{ flex: "0 0 130px" }}>
                          <label>Command</label>
                          <input
                            placeholder="npx"
                            value={srv.command ?? ""}
                            onChange={(e) =>
                              updateServer(i, { command: e.target.value || null })
                            }
                          />
                        </div>
                        <div style={{ flex: 1 }}>
                          <label>Args (space-separated)</label>
                          <input
                            placeholder="-y @modelcontextprotocol/server-filesystem ~/code"
                            value={(srv.args ?? []).join(" ")}
                            onChange={(e) =>
                              updateServer(i, {
                                args: e.target.value.split(/\s+/).filter(Boolean),
                              })
                            }
                          />
                        </div>
                      </div>
                    )}
                    <div className="row" style={{ marginTop: 8, alignItems: "center" }}>
                      <label style={{ textTransform: "none", letterSpacing: 0, margin: 0 }}>
                        <input
                          type="checkbox"
                          checked={srv.enabled ?? true}
                          onChange={(e) => updateServer(i, { enabled: e.target.checked })}
                        />{" "}
                        enabled
                      </label>
                      <label style={{ textTransform: "none", letterSpacing: 0, margin: 0 }}>
                        <input
                          type="checkbox"
                          checked={srv.auto_approve ?? false}
                          onChange={(e) =>
                            updateServer(i, { auto_approve: e.target.checked })
                          }
                        />{" "}
                        auto-approve tools
                      </label>
                    </div>
                  </div>
                );
              })}
              <button
                className="tbtn"
                style={{ marginTop: 8 }}
                onClick={() =>
                  setServers([
                    ...servers,
                    { name: "", transport: "stdio", enabled: true, auto_approve: false },
                  ])
                }
              >
                + add MCP server
              </button>
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
