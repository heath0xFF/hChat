import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/tauri";
import type { BenchResult } from "../types";
import { Chart } from "./Chart";

interface Props {
  endpoints: string[];
  defaultEndpoint: string;
}

function fmt(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  return n.toFixed(digits);
}

function hostOf(url: string): string {
  try {
    return new URL(url).host || url;
  } catch {
    return url;
  }
}

export function BenchView({ endpoints, defaultEndpoint }: Props) {
  const [endpoint, setEndpoint] = useState(defaultEndpoint || endpoints[0] || "");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState(
    "Write a few sentences about the history of computing.",
  );
  const [concurrency, setConcurrency] = useState(4);
  const [requests, setRequests] = useState(16);
  const [maxTokens, setMaxTokens] = useState(128);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BenchResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Refresh the model list whenever the target endpoint changes.
  useEffect(() => {
    let live = true;
    if (!endpoint) return;
    api
      .fetchModels(endpoint)
      .then((ms) => {
        if (!live) return;
        setModels(ms);
        setModel((cur) => (cur && ms.includes(cur) ? cur : (ms[0] ?? "")));
      })
      .catch(() => live && setModels([]));
    return () => {
      live = false;
    };
  }, [endpoint]);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await api.runBenchmark({
        endpoint,
        model,
        prompt,
        max_tokens: maxTokens,
        concurrency,
        total_requests: requests,
      });
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }, [endpoint, model, prompt, maxTokens, concurrency, requests]);

  return (
    <div className="bench">
      <div className="bench-form">
        <label className="bench-field bench-field-wide">
          <span>Endpoint</span>
          <select
            className="model-select"
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
          >
            {!endpoints.includes(endpoint) && endpoint && (
              <option value={endpoint}>{endpoint}</option>
            )}
            {endpoints.map((u) => (
              <option key={u} value={u}>
                {hostOf(u)}
              </option>
            ))}
          </select>
        </label>
        <label className="bench-field bench-field-wide">
          <span>Model</span>
          <select
            className="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {models.length === 0 && <option value="">no models</option>}
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label className="bench-field">
          <span>Concurrency</span>
          <input
            type="number"
            min={1}
            max={64}
            value={concurrency}
            onChange={(e) =>
              setConcurrency(Math.max(1, Math.min(64, Number(e.target.value) || 1)))
            }
          />
        </label>
        <label className="bench-field">
          <span>Requests</span>
          <input
            type="number"
            min={1}
            max={500}
            value={requests}
            onChange={(e) =>
              setRequests(Math.max(1, Math.min(500, Number(e.target.value) || 1)))
            }
          />
        </label>
        <label className="bench-field">
          <span>Max tokens</span>
          <input
            type="number"
            min={1}
            max={4096}
            value={maxTokens}
            onChange={(e) =>
              setMaxTokens(Math.max(1, Math.min(4096, Number(e.target.value) || 1)))
            }
          />
        </label>
        <button
          className="bench-run"
          disabled={running || !model}
          onClick={() => void run()}
        >
          {running ? "running…" : "Run benchmark"}
        </button>
      </div>

      <label className="bench-field bench-prompt">
        <span>Prompt</span>
        <textarea
          rows={2}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </label>

      {error && <div className="model-note model-note-err">{error}</div>}

      {running && !result && (
        <div className="model-note">
          Firing {requests} requests, {concurrency} at a time…
        </div>
      )}

      {result && (
        <>
          <div className="tiles">
            <div className="tile">
              <div className="label">Throughput</div>
              <div className="value">
                {fmt(result.agg_decode_tok_s)}
                <span className="unit">tok/s</span>
              </div>
              <div className="sub">aggregate decode</div>
            </div>
            <div className="tile">
              <div className="label">TTFT p50</div>
              <div className="value">
                {result.ttft_ms.p50 === null ? "—" : Math.round(result.ttft_ms.p50)}
                <span className="unit">ms</span>
              </div>
              <div className="sub">
                p95 {result.ttft_ms.p95 ? Math.round(result.ttft_ms.p95) : "—"} ms
              </div>
            </div>
            <div className="tile">
              <div className="label">Decode p50</div>
              <div className="value">
                {fmt(result.decode_tok_s.p50)}
                <span className="unit">tok/s</span>
              </div>
              <div className="sub">p95 {fmt(result.decode_tok_s.p95)}</div>
            </div>
            <div className="tile">
              <div className="label">OK</div>
              <div className="value" style={{ fontSize: 24 }}>
                {result.ok}/{result.requests}
              </div>
              <div className="sub">{result.errors} errored</div>
            </div>
            <div className="tile">
              <div className="label">Wall</div>
              <div className="value" style={{ fontSize: 20 }}>
                {(result.wall_ms / 1000).toFixed(2)}s
              </div>
              <div className="sub">
                {result.total_completion_tokens.toLocaleString()} tok
              </div>
            </div>
          </div>

          <div className="charts">
            <div className="chart-card">
              <div className="chart-head">
                <span>TTFT per request (ms)</span>
                <span>{result.ttft_series.length}</span>
              </div>
              <Chart data={result.ttft_series} color="#d98c4a" />
            </div>
            <div className="chart-card">
              <div className="chart-head">
                <span>Decode per request (tok/s)</span>
                <span>{result.decode_series.length}</span>
              </div>
              <Chart data={result.decode_series} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
