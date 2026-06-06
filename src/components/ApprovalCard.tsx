import type { PendingApproval } from "../types";

export type ApprovalDecision = "approve" | "approve_all" | "deny";

interface Props {
  approval: PendingApproval;
  onResolve: (decision: ApprovalDecision) => void;
}

export function ApprovalCard({ approval, onResolve }: Props) {
  let pretty = approval.arguments;
  try {
    pretty = JSON.stringify(JSON.parse(approval.arguments), null, 2);
  } catch {
    /* leave raw */
  }
  return (
    <div className="approval">
      <div className="ap-head">Tool approval required</div>
      <div className="ap-name">ƒ {approval.name}</div>
      <pre>{pretty}</pre>
      <div className="ap-actions">
        <button className="ap-approve" onClick={() => onResolve("approve")}>
          Approve &amp; run
        </button>
        <button className="ap-allow" onClick={() => onResolve("approve_all")}>
          Approve all ({approval.name})
        </button>
        <button className="ap-deny" onClick={() => onResolve("deny")}>
          Deny
        </button>
      </div>
    </div>
  );
}
