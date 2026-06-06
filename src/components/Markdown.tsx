import { useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { CodeBlock } from "./CodeBlock";

const REMARK_PLUGINS = [remarkGfm];

interface Props {
  text: string;
  streaming?: boolean;
  onOpenArtifact?: (code: string, lang: string) => void;
}

export function Markdown({ text, streaming, onOpenArtifact }: Props) {
  // Memoize the components so the `code`/`pre` renderers keep a stable identity
  // across re-renders. Without this, react-markdown sees a new component type
  // each render and remounts every code block — which resets the shiki-
  // highlighted HTML to the plain fallback and flickers.
  const components: Components = useMemo(
    () => ({
      pre({ children }) {
        return <>{children}</>;
      },
      code({ className, children, ...props }) {
        const match = /language-(\w+)/.exec(className || "");
        const raw = String(children ?? "");
        const isBlock = !!match || raw.includes("\n");
        if (isBlock) {
          return (
            <CodeBlock
              code={raw.replace(/\n$/, "")}
              lang={match?.[1] || "text"}
              streaming={streaming}
              onOpenArtifact={onOpenArtifact}
            />
          );
        }
        return (
          <code className={className} {...props}>
            {children}
          </code>
        );
      },
    }),
    [streaming, onOpenArtifact],
  );

  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
