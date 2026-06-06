import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/tauri";
import type { UsageStats } from "../types";
import { Chart } from "./Chart";

function fmtInt(n: number): string {
  return n.toLocaleString();
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function hostOf(url: string): string {
  try {
    return new URL(url).host || url;
  } catch {
    return url;
  }
}

export function UsageView() {
  const [stats, setStats] = useState<UsageStats | null>(null);

  const refresh = useCallback(async () => {
    setStats(await api.usageStats());
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const clear = useCallback(async () => {
    if (!window.confirm("Clear all recorded usage history?")) return;
    await api.clearUsage();
    await refresh();
  }, [refresh]);

  const successRate =
    stats && stats.total_requests > 0
      ? (stats.ok_requests / stats.total_requests) * 100
      : null;

  const ms = (n: number | null | undefined) =>
    n == null ? "—" : `${Math.round(n)}ms`;
  const tps = (n: number | null | undefined) =>
    n == null ? "—" : n.toFixed(1);
  const hasCost = (stats?.total_cost ?? 0) > 0;

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
        <h1 style={{ margin: 0 }}>Usage</h1>
        <div style={{ flex: 1 }} />
        <button className="tbtn" onClick={() => void refresh()}>
          refresh
        </button>
        <button className="tbtn" onClick={() => void clear()}>
          clear
        </button>
      </div>
      <div className="sub" style={{ margin: "0 0 10px" }}>
        Your own per-request history — every completed turn, recorded locally.
      </div>

      <div className="tiles">
        <div className="tile">
          <div className="label">Total tokens</div>
          <div className="value" style={{ fontSize: 22 }}>
            {fmtTokens(stats?.total_tokens ?? 0)}
          </div>
          <div className="sub">{fmtInt(stats?.total_tokens ?? 0)}</div>
        </div>
        {hasCost && (
          <div className="tile">
            <div className="label">Cost</div>
            <div className="value" style={{ fontSize: 22 }}>
              ${(stats?.total_cost ?? 0).toFixed(2)}
            </div>
            <div className="sub">reported by provider</div>
          </div>
        )}
        <div className="tile">
          <div className="label">Requests</div>
          <div className="value" style={{ fontSize: 24 }}>
            {fmtInt(stats?.total_requests ?? 0)}
          </div>
        </div>
        <div className="tile">
          <div className="label">Prompt</div>
          <div className="value" style={{ fontSize: 22 }}>
            {fmtTokens(stats?.prompt_tokens ?? 0)}
          </div>
          <div className="sub">{fmtInt(stats?.prompt_tokens ?? 0)}</div>
        </div>
        <div className="tile">
          <div className="label">Completion</div>
          <div className="value" style={{ fontSize: 22 }}>
            {fmtTokens(stats?.completion_tokens ?? 0)}
          </div>
          <div className="sub">{fmtInt(stats?.completion_tokens ?? 0)}</div>
        </div>
        <div className="tile">
          <div className="label">Success</div>
          <div className="value">
            {successRate === null ? "—" : successRate.toFixed(0)}
            <span className="unit">%</span>
          </div>
          <div className="sub">{fmtInt(stats?.ok_requests ?? 0)} ok</div>
        </div>
      </div>

      <div className="totals">
        <span>
          <span className="k">TTFT p50</span>
          {ms(stats?.ttft_p50_ms)}
        </span>
        <span>
          <span className="k">TTFT p95</span>
          {ms(stats?.ttft_p95_ms)}
        </span>
        <span>
          <span className="k">Decode p50</span>
          {tps(stats?.decode_p50_tok_s)} tok/s
        </span>
        <span>
          <span className="k">Decode p95</span>
          {tps(stats?.decode_p95_tok_s)} tok/s
        </span>
      </div>

      <div className="charts">
        <div className="chart-card">
          <div className="chart-head">
            <span>Tokens by day</span>
            <span>{stats?.daily.length ?? 0}d</span>
          </div>
          <Chart data={(stats?.daily ?? []).map((d) => d.total_tokens)} />
        </div>
      </div>

      <div className="chart-card" style={{ marginTop: 12 }}>
        <div className="chart-head">
          <span>By model</span>
          <span>{stats?.by_model.length ?? 0}</span>
        </div>
        <table className="usage-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Endpoint</th>
              <th className="num">Reqs</th>
              <th className="num">Tokens</th>
              <th className="num">Prompt</th>
              <th className="num">Compl.</th>
              {hasCost && <th className="num">Cost</th>}
              <th className="num">TTFT</th>
              <th className="num">tok/s</th>
            </tr>
          </thead>
          <tbody>
            {(stats?.by_model ?? []).map((m, i) => (
              <tr key={`${m.model}@${m.endpoint}@${i}`}>
                <td title={m.model}>{m.model || "—"}</td>
                <td title={m.endpoint}>{hostOf(m.endpoint)}</td>
                <td className="num">{fmtInt(m.requests)}</td>
                <td className="num">{fmtTokens(m.total_tokens)}</td>
                <td className="num">{fmtTokens(m.prompt_tokens)}</td>
                <td className="num">{fmtTokens(m.completion_tokens)}</td>
                {hasCost && <td className="num">${m.cost.toFixed(3)}</td>}
                <td className="num">
                  {m.avg_ttft_ms != null ? `${Math.round(m.avg_ttft_ms)}ms` : "—"}
                </td>
                <td className="num">
                  {m.avg_decode_tok_s != null ? m.avg_decode_tok_s.toFixed(1) : "—"}
                </td>
              </tr>
            ))}
            {(!stats || stats.by_model.length === 0) && (
              <tr>
                <td colSpan={hasCost ? 9 : 8} className="usage-empty">
                  No usage recorded yet — send a message to start tracking.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
