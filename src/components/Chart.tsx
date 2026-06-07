import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

interface Props {
  data: number[];
  color?: string;
  height?: number;
}

/** Abbreviate axis tick values so they fit the narrow y-gutter: 77224 → "77k",
 *  1281 → "1.3k", 48.2 → "48". Without this, large ms values (TTFT) overflow
 *  the axis width and render clipped. */
function fmtTick(v: number): string {
  if (!isFinite(v)) return "";
  const a = Math.abs(v);
  if (a >= 1000) {
    const k = v / 1000;
    return `${a >= 10000 ? Math.round(k) : Math.round(k * 10) / 10}k`;
  }
  return String(Math.round(v * 10) / 10);
}

/** Minimal uPlot line/area chart for the metrics dashboard. x is the sample
 *  index; the series fills under the line. Resizes with its container. */
export function Chart({ data, color = "#38bdb4", height = 150 }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  const toData = (d: number[]): uPlot.AlignedData => {
    // A single sample (all a no-scrape backend gives per request) can't draw a
    // line; duplicate it into a short flat segment so it's always visible.
    if (d.length === 1) return [[0, 1], [d[0], d[0]]];
    return [d.map((_, i) => i), d.length ? d : [0]];
  };

  // Create once.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const opts: uPlot.Options = {
      width: el.clientWidth || 400,
      height,
      cursor: { show: false },
      legend: { show: false },
      scales: { x: { time: false } },
      axes: [
        { show: false },
        {
          stroke: "#5a5a62",
          grid: { stroke: "#1c1c20", width: 1 },
          ticks: { show: false },
          size: 44,
          font: "10px ui-monospace, monospace",
          values: (_u, splits) => splits.map(fmtTick),
        },
      ],
      series: [
        {},
        {
          stroke: color,
          width: 1.5,
          fill: `${color}18`,
          // Adaptive points: uPlot shows dots when samples are sparse and hides
          // them once the line is dense. Backends without a live metrics scrape
          // (e.g. llama-swap) only contribute one point per request, which a
          // line alone can't render — the dot makes it visible.
          points: { size: 5 },
        },
      ],
    };
    const u = new uPlot(opts, toData(data), el);
    plotRef.current = u;

    const ro = new ResizeObserver(() => {
      if (el.clientWidth) u.setSize({ width: el.clientWidth, height });
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      u.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update on data change.
  useEffect(() => {
    plotRef.current?.setData(toData(data));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  return <div ref={wrapRef} className="uplot-wrap" />;
}
