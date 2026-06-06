interface Props {
  data: number[];
  height?: number;
  color?: string;
}

export function Sparkline({ data, height = 150, color = "#38bdb4" }: Props) {
  const w = 600;
  const h = height;
  if (data.length < 2) {
    return (
      <svg className="uplot-wrap" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" width="100%" height={h}>
        <line x1="0" y1={h - 1} x2={w} y2={h - 1} stroke="#26262b" strokeWidth="1" />
      </svg>
    );
  }
  const max = Math.max(...data, 0.0001);
  const min = Math.min(...data, 0);
  const span = max - min || 1;
  const step = w / (data.length - 1);
  const pts = data
    .map((v, i) => `${(i * step).toFixed(1)},${(h - ((v - min) / span) * (h - 8) - 4).toFixed(1)}`)
    .join(" ");
  const area = `0,${h} ${pts} ${w},${h}`;
  return (
    <svg className="uplot-wrap" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" width="100%" height={h}>
      <polygon points={area} fill={color} opacity="0.08" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}
