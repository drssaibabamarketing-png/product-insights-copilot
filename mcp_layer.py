"""MCP-style tool layer with an approval gate.

Mirrors the Model Context Protocol shape (tools/list, tools/call, JSON schemas) but runs
in-process so it works with no accounts. Swap the two tool bodies for Google Docs / Gmail
MCP server calls to go live; the approval gate stays identical.

Rule enforced in code: NO tool executes without an ApprovalGate that a human approved.
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

OUT = Path(__file__).parent / "outputs"


class ApprovalRequired(PermissionError):
    pass


class ApprovalGate:
    """Created pending; only a human action (button click / CLI prompt) flips it."""

    def __init__(self, summary: str):
        self.id = secrets.token_hex(4)
        self.summary = summary
        self.status = "pending"          # pending | approved | rejected
        self.decided_at = None

    def decide(self, approve: bool):
        self.status = "approved" if approve else "rejected"
        self.decided_at = datetime.now().isoformat(timespec="seconds")


TOOLS = {
    "append_to_notes": {
        "description": "Append the weekly entry to the team notes doc (outputs/notes_log.md + .jsonl).",
        "inputSchema": {"type": "object", "required": ["date", "top_themes", "weekly_pulse", "identified_fee_issue",
                                                       "explanation_bullets", "source_links"]},
    },
    "create_email_draft": {
        "description": "Create an email DRAFT (.eml, X-Unsent). Never sends.",
        "inputSchema": {"type": "object", "required": ["subject", "body"]},
    },
}


def list_tools() -> list[dict]:
    return [{"name": n, **t} for n, t in TOOLS.items()]


def _append_to_notes(a: dict) -> dict:
    OUT.mkdir(exist_ok=True)
    md = OUT / "notes_log.md"
    entry = (f"\n---\n## Weekly entry - {a['date']}\n"
             f"**Top themes:** {', '.join(a['top_themes'])}\n\n"
             f"**Identified fee issue:** {a['identified_fee_issue']}\n\n"
             f"### Weekly pulse\n{a['weekly_pulse']}\n\n"
             f"### Explanation bullets\n" + "\n".join(f"- {b}" for b in a["explanation_bullets"]) +
             "\n\n**Source links:**\n" + "\n".join(f"- {u}" for u in a["source_links"]) + "\n")
    with md.open("a", encoding="utf-8") as f:
        f.write(entry)
    with (OUT / "notes_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return {"ok": True, "path": str(md.name), "chars_appended": len(entry)}


def _create_email_draft(a: dict) -> dict:
    OUT.mkdir(exist_ok=True)
    m = EmailMessage()
    m["Subject"] = a["subject"]
    m["To"] = a.get("to", "")
    m["X-Unsent"] = "1"                  # opens as an unsent draft in Outlook/Thunderbird
    m.set_content(a["body"])
    p = OUT / "email_draft.eml"
    p.write_bytes(bytes(m))
    (OUT / "email_draft.txt").write_text(f"Subject: {a['subject']}\n\n{a['body']}", encoding="utf-8")
    return {"ok": True, "path": p.name, "sent": False}


_IMPL = {"append_to_notes": _append_to_notes, "create_email_draft": _create_email_draft}


def call_tool(name: str, arguments: dict, gate: ApprovalGate) -> dict:
    """tools/call equivalent. Refuses unless the gate was approved by a human."""
    if gate is None or gate.status != "approved":
        raise ApprovalRequired(f"'{name}' blocked: approval gate is {getattr(gate, 'status', 'missing')}")
    missing = [k for k in TOOLS[name]["inputSchema"]["required"] if k not in arguments]
    if missing:
        raise ValueError(f"missing arguments: {missing}")
    res = _IMPL[name](arguments)
    OUT.mkdir(exist_ok=True)
    with (OUT / "audit_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "tool": name,
                            "gate": gate.id, "approved_at": gate.decided_at, "result": res}) + "\n")
    return res


def build_payloads(result: dict, to: str = "") -> tuple[dict, dict]:
    notes = {
        "date": result["date"], "top_themes": result["top_themes"], "weekly_pulse": result["pulse"],
        "identified_fee_issue": result["fee"]["fee"],
        "explanation_bullets": result["explainer"]["bullets"],
        "source_links": [s["url"] for s in result["explainer"]["sources"]],
    }
    email = {
        "to": to, "subject": f"Weekly Product Pulse + Customer Clarification — {result['date']}",
        "body": ("WEEKLY PRODUCT PULSE\n\n" + result["pulse"].replace("**", "") +
                 "\n\n---\nFEE EXPLANATION (reusable support snippet)\n\n" +
                 result["explainer_text"].replace("**", "")),
    }
    return notes, email
