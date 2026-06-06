import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CodeBlock } from "./CodeBlock";

interface Props {
  text: string;
  onOpenArtifact?: (code: string, lang: string) => void;
}

export function Markdown({ text, onOpenArtifact }: Props) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
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
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
