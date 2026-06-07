import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

// In-app replacements for window.confirm / window.prompt, which don't render
// reliably inside the Tauri (WKWebView) webview. Exposed as async functions
// via the useDialog() hook.

interface ConfirmOpts {
  title?: string;
  confirmText?: string;
  cancelText?: string;
  danger?: boolean;
}
interface PromptOpts {
  title?: string;
  defaultValue?: string;
  placeholder?: string;
  confirmText?: string;
  cancelText?: string;
}

type Req =
  | {
      kind: "confirm";
      message: string;
      opts: ConfirmOpts;
      resolve: (v: boolean) => void;
    }
  | {
      kind: "prompt";
      message: string;
      opts: PromptOpts;
      resolve: (v: string | null) => void;
    };

interface DialogApi {
  confirm: (message: string, opts?: ConfirmOpts) => Promise<boolean>;
  prompt: (message: string, opts?: PromptOpts) => Promise<string | null>;
}

const DialogContext = createContext<DialogApi | null>(null);

export function useDialog(): DialogApi {
  const ctx = useContext(DialogContext);
  if (!ctx) throw new Error("useDialog must be used within a DialogProvider");
  return ctx;
}

export function DialogProvider({ children }: { children: ReactNode }) {
  const [req, setReq] = useState<Req | null>(null);
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  // Mirrors `req` so a superseding request can settle the in-flight one
  // without its awaiting caller hanging forever.
  const reqRef = useRef<Req | null>(null);

  // Resolve the pending promise and dismiss. `r` is always in scope at the
  // call sites, so the promise is settled outside the state updater (keeping
  // setReq pure).
  const finish = (r: Req, value: boolean | string | null) => {
    if (r.kind === "confirm") r.resolve(value as boolean);
    else r.resolve(value as string | null);
    reqRef.current = null;
    setReq(null);
  };
  const cancel = (r: Req) => finish(r, r.kind === "confirm" ? false : null);

  const open = (r: Req) => {
    if (reqRef.current) cancel(reqRef.current); // settle any superseded dialog
    reqRef.current = r;
    setReq(r);
  };

  const confirm = useCallback(
    (message: string, opts: ConfirmOpts = {}) =>
      new Promise<boolean>((resolve) =>
        open({ kind: "confirm", message, opts, resolve }),
      ),
    [],
  );
  const prompt = useCallback(
    (message: string, opts: PromptOpts = {}) =>
      new Promise<string | null>((resolve) => {
        setInput(opts.defaultValue ?? "");
        open({ kind: "prompt", message, opts, resolve });
      }),
    [],
  );

  // Focus the prompt input when it opens.
  useEffect(() => {
    if (req?.kind === "prompt") {
      const t = setTimeout(() => inputRef.current?.focus(), 0);
      return () => clearTimeout(t);
    }
  }, [req]);

  // Esc cancels; Enter confirms a confirm dialog. Capture phase so this wins
  // over other global Esc handlers (dock, settings) while a dialog is open.
  useEffect(() => {
    if (!req) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        cancel(req);
      } else if (
        e.key === "Enter" &&
        req.kind === "confirm" &&
        !req.opts.danger
      ) {
        // Don't let a stray Enter auto-confirm a destructive action.
        e.preventDefault();
        e.stopPropagation();
        finish(req, true);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [req]);

  return (
    <DialogContext.Provider value={{ confirm, prompt }}>
      {children}
      {req && (
        <div className="modal-backdrop" onClick={() => cancel(req)}>
          <div
            className="modal dialog-modal"
            onClick={(e) => e.stopPropagation()}
          >
            {req.opts.title && <h2>{req.opts.title}</h2>}
            <p className="dialog-message">{req.message}</p>
            {req.kind === "prompt" && (
              <input
                ref={inputRef}
                className="dialog-input"
                value={input}
                placeholder={req.opts.placeholder}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && input.trim()) {
                    e.preventDefault();
                    finish(req, input);
                  }
                }}
              />
            )}
            <div className="modal-actions">
              <button className="tbtn" onClick={() => cancel(req)}>
                {req.opts.cancelText ?? "Cancel"}
              </button>
              <button
                className={`tbtn ${
                  req.kind === "confirm" && req.opts.danger ? "danger" : "accent"
                }`}
                disabled={req.kind === "prompt" && !input.trim()}
                onClick={() => finish(req, req.kind === "confirm" ? true : input)}
              >
                {req.opts.confirmText ?? (req.kind === "confirm" ? "OK" : "Save")}
              </button>
            </div>
          </div>
        </div>
      )}
    </DialogContext.Provider>
  );
}
