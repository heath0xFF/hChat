import { useEffect, useState } from "react";

let counter = 0;

/** Render a mermaid diagram. Mermaid is lazy-loaded so it doesn't weigh down
 *  the main bundle until an artifact actually needs it. */
export function MermaidView({ code }: { code: string }) {
  const [svg, setSvg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setSvg(null);
    setErr(null);
    (async () => {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: "default",
          securityLevel: "strict",
        });
        const { svg } = await mermaid.render(`mmd-${counter++}`, code);
        if (alive) setSvg(svg);
      } catch (e) {
        if (alive) setErr(String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [code]);

  if (err) {
    return (
      <div className="dashboard-empty" style={{ padding: 20 }}>
        Mermaid error: {err}
      </div>
    );
  }
  if (!svg) {
    return (
      <div className="dashboard-empty" style={{ padding: 20 }}>
        Rendering diagram…
      </div>
    );
  }
  return (
    <div className="artifact-mermaid" dangerouslySetInnerHTML={{ __html: svg }} />
  );
}
