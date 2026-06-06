import type { SettingsDto } from "../types";
import { Sparkline } from "./Sparkline";

export interface LiveMetrics {
  decode: number | null;
  ttft: number | null;
  prefill: number | null;
  peakDecode: number;
  peakTtft: number;
  peakPrefill: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  durationMs: number;
  cost: number | null;
  activeRequests: number;
  throughputHistory: number[];
  ttftHistory: number[];
}

interface Props {
  settings: SettingsDto;
  model: string | null;
  streaming: boolean;
  metrics: LiveMetrics;
}

function fmt(n: number | null, digits = 1): string {
  if (n === null || !isFinite(n)) return "—";
  return n.toFixed(digits);
}

export function StatusView({ settings, model, streaming, metrics }: Props) {
  const m = metrics;
  return (
    <div className="status">
      <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "8px 0 6px" }}>
        <span className="badge dot">{streaming ? "ACTIVE" : "IDLE"}</span>
        <span className="badge">{settings.endpoint}</span>
      </div>
      <h1>{model ?? "No model selected"}</h1>

      <div className="tiles">
        <div className="tile">
          <div className="label">Decode</div>
          <div className="value">
            {fmt(m.decode)}
            <span className="unit">tok/s</span>
          </div>
          <div className="sub">peak {fmt(m.peakDecode)}</div>
        </div>
        <div className="tile">
          <div className="label">TTFT</div>
          <div className="value">
            {m.ttft === null ? "—" : Math.round(m.ttft)}
            <span className="unit">ms</span>
          </div>
          <div className="sub">peak {m.peakTtft ? Math.round(m.peakTtft) : "—"} ms</div>
        </div>
        <div className="tile">
          <div className="label">Prefill</div>
          <div className="value">
            {fmt(m.prefill)}
            <span className="unit">t/s</span>
          </div>
          <div className="sub">peak {fmt(m.peakPrefill)}</div>
        </div>
        <div className="tile">
          <div className="label">Req</div>
          <div className="value" style={{ fontSize: 24 }}>
            {m.activeRequests}/1
          </div>
        </div>
        <div className="tile">
          <div className="label">VRAM</div>
          <div className="value" style={{ fontSize: 20 }}>
            —
          </div>
          <div className="sub">no gpu source</div>
        </div>
        <div className="tile">
          <div className="label">Power</div>
          <div className="value" style={{ fontSize: 20 }}>
            —
          </div>
          <div className="sub">no gpu source</div>
        </div>
      </div>

      <div className="totals">
        <span>
          <span className="k">Total tokens</span>
          {m.totalTokens.toLocaleString()}
        </span>
        <span>
          <span className="k">Prompt</span>
          {m.promptTokens.toLocaleString()}
        </span>
        <span>
          <span className="k">Completion</span>
          {m.completionTokens.toLocaleString()}
        </span>
        <span>
          <span className="k">Duration</span>
          {(m.durationMs / 1000).toFixed(2)}s
        </span>
        {m.cost !== null && (
          <span>
            <span className="k">Cost</span>${m.cost.toFixed(4)}
          </span>
        )}
      </div>

      <div className="charts">
        <div className="chart-card">
          <div className="chart-head">
            <span>Throughput (tok/s)</span>
            <span>recent</span>
          </div>
          <Sparkline data={m.throughputHistory} />
        </div>
        <div className="chart-card">
          <div className="chart-head">
            <span>TTFT (ms)</span>
            <span>recent</span>
          </div>
          <Sparkline data={m.ttftHistory} color="#e2a03f" />
        </div>
      </div>

      <div className="gpu-list">
        <div className="chart-head" style={{ marginBottom: 10 }}>
          <span>GPUs</span>
        </div>
        <div className="dashboard-empty">
          GPU metrics (VRAM, power, temp, util) arrive in Phase B — via macmon
          locally and the hchat-agent on the Spark.
        </div>
      </div>

      <div className="logs">
        <div className="chart-head" style={{ marginBottom: 10 }}>
          <span>Controller logs</span>
        </div>
        <div className="dashboard-empty">Server log tail wires up in Phase B.</div>
      </div>
    </div>
  );
}
