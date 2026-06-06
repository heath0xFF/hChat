import type { MetricsSnapshot, SettingsDto } from "../types";
import { Chart } from "./Chart";

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
  gpuUtilHistory: number[];
  gpuPowerHistory: number[];
}

interface Props {
  settings: SettingsDto;
  model: string | null;
  streaming: boolean;
  metrics: LiveMetrics;
  snapshot: MetricsSnapshot | null;
}

function fmt(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  return n.toFixed(digits);
}

export function StatusView({ settings, model, streaming, metrics, snapshot }: Props) {
  const m = metrics;
  const server = snapshot?.server ?? null;
  const gpu = snapshot?.gpu ?? null;

  // Prefer server-scraped values (whole-server view); fall back to the last
  // request's client-measured numbers.
  const decode = server?.decode_tok_s ?? m.decode;
  const ttft = server?.ttft_ms ?? m.ttft;
  const prefill = server?.prefill_tok_s ?? m.prefill;
  const running = server?.requests_running ?? m.activeRequests;
  const waiting = server?.requests_waiting ?? 0;

  return (
    <div className="status">
      <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "8px 0 6px" }}>
        <span className="badge dot">{streaming ? "ACTIVE" : "IDLE"}</span>
        {snapshot?.runtime && <span className="badge">{snapshot.runtime}</span>}
        <span className="badge">{settings.endpoint}</span>
      </div>
      <h1>{model ?? "No model selected"}</h1>

      <div className="tiles">
        <div className="tile">
          <div className="label">Decode</div>
          <div className="value">
            {fmt(decode)}
            <span className="unit">tok/s</span>
          </div>
          <div className="sub">peak {fmt(m.peakDecode)}</div>
        </div>
        <div className="tile">
          <div className="label">TTFT</div>
          <div className="value">
            {ttft === null ? "—" : Math.round(ttft)}
            <span className="unit">ms</span>
          </div>
          <div className="sub">peak {m.peakTtft ? Math.round(m.peakTtft) : "—"} ms</div>
        </div>
        <div className="tile">
          <div className="label">Prefill</div>
          <div className="value">
            {fmt(prefill)}
            <span className="unit">t/s</span>
          </div>
          <div className="sub">peak {fmt(m.peakPrefill)}</div>
        </div>
        <div className="tile">
          <div className="label">Req</div>
          <div className="value" style={{ fontSize: 24 }}>
            {running}/{running + waiting}
          </div>
        </div>
        <div className="tile">
          <div className="label">VRAM</div>
          <div className="value" style={{ fontSize: 20 }}>
            {gpu?.vram_used_gb != null
              ? `${gpu.vram_used_gb.toFixed(1)}/${(gpu.vram_total_gb ?? 0).toFixed(0)}G`
              : "—"}
          </div>
          <div className="sub">{gpu?.source ? gpu.source : "no gpu source"}</div>
        </div>
        <div className="tile">
          <div className="label">Power</div>
          <div className="value" style={{ fontSize: 20 }}>
            {gpu?.power_w != null
              ? `${gpu.power_w.toFixed(0)}${gpu.power_limit_w ? "/" + gpu.power_limit_w.toFixed(0) : ""}W`
              : "—"}
          </div>
          <div className="sub">
            {gpu?.temp_c != null ? `${gpu.temp_c.toFixed(0)}° temp` : "—"}
          </div>
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
        {server?.kv_cache_pct != null && (
          <span>
            <span className="k">KV cache</span>
            {server.kv_cache_pct.toFixed(0)}%
          </span>
        )}
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
          <Chart data={m.throughputHistory} />
        </div>
        <div className="chart-card">
          <div className="chart-head">
            <span>TTFT (ms)</span>
            <span>recent</span>
          </div>
          <Chart data={m.ttftHistory} color="#e2a03f" />
        </div>
        {(m.gpuUtilHistory.length > 1 || m.gpuPowerHistory.length > 1) && (
          <>
            <div className="chart-card">
              <div className="chart-head">
                <span>GPU util (%)</span>
                <span>recent</span>
              </div>
              <Chart data={m.gpuUtilHistory} color="#6f9fff" />
            </div>
            <div className="chart-card">
              <div className="chart-head">
                <span>GPU power (W)</span>
                <span>recent</span>
              </div>
              <Chart data={m.gpuPowerHistory} color="#e5484d" />
            </div>
          </>
        )}
      </div>

      <div className="gpu-list">
        <div className="chart-head" style={{ marginBottom: 10 }}>
          <span>GPUs {gpu?.devices?.length ? gpu.devices.length : ""}</span>
          {gpu?.source && <span>{gpu.source}</span>}
        </div>
        {gpu?.devices && gpu.devices.length > 0 ? (
          <>
            <div className="gpu-row gpu-head">
              <span>GPU</span>
              <span>NAME</span>
              <span>MEMORY</span>
              <span>UTIL</span>
              <span>TEMP</span>
              <span>POWER</span>
            </div>
            {gpu.devices.map((d, i) => (
            <div className="gpu-row" key={i}>
              <span>G{i}</span>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {d.name}
              </span>
              <span>
                {d.mem_used_gb.toFixed(1)}/{d.mem_total_gb.toFixed(0)}G
              </span>
              <span>{d.util_pct.toFixed(0)}%</span>
              <span>{d.temp_c.toFixed(0)}°</span>
              <span>
                {d.power_w.toFixed(0)}
                {d.power_limit_w ? "/" + d.power_limit_w.toFixed(0) : ""}W
              </span>
            </div>
            ))}
          </>
        ) : (
          <div className="dashboard-empty">
            No GPU source for this backend. Configure <code>gpu = "macmon"</code>{" "}
            (local Mac) or <code>gpu = "agent"</code> + <code>agent_url</code> (the
            Spark) on the endpoint in <code>config.toml</code>.
          </div>
        )}
      </div>
    </div>
  );
}
