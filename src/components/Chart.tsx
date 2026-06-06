import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

interface Props {
  data: number[];
  color?: string;
  height?: number;
}

/** Minimal uPlot line/area chart for the metrics dashboard. x is the sample
 *  index; the series fills under the line. Resizes with its container. */
export function Chart({ data, color = "#38bdb4", height = 150 }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  const toData = (d: number[]): uPlot.AlignedData => [
    d.map((_, i) => i),
    d.length ? d : [0],
  ];

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
          size: 38,
          font: "10px ui-monospace, monospace",
        },
      ],
      series: [
        {},
        {
          stroke: color,
          width: 1.5,
          fill: `${color}18`,
          points: { show: false },
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
