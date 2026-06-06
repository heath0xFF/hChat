import { useEffect, useState } from "react";
import { highlight } from "../lib/highlight";

interface Props {
  code: string;
  lang: string;
  streaming?: boolean;
  onOpenArtifact?: (code: string, lang: string) => void;
}

export function CodeBlock({ code, lang, streaming, onOpenArtifact }: Props) {
  const [html, setHtml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Skip syntax highlighting while streaming — re-highlighting on every token
  // flickers. Render plain text live, then highlight once the stream settles.
  useEffect(() => {
    if (streaming) {
      setHtml(null);
      return;
    }
    let alive = true;
    highlight(code, lang).then((h) => {
      if (alive) setHtml(h);
    });
    return () => {
      alive = false;
    };
  }, [code, lang, streaming]);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className="code-block">
      <div className="code-head">
        <span>{lang || "text"}</span>
        <span className="acts">
          {onOpenArtifact && (
            <button onClick={() => onOpenArtifact(code, lang)}>open</button>
          )}
          <button onClick={copy}>{copied ? "copied" : "copy"}</button>
        </span>
      </div>
      {html ? (
        <div dangerouslySetInnerHTML={{ __html: html }} />
      ) : (
        <pre>
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}
