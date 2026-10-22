"""Flask UI: upload/choose CSV -> analyse -> preview -> APPROVE -> MCP actions."""
import os
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, url_for

import mcp_layer as mcp
import pipeline

BASE = Path(__file__).parent
app = Flask(__name__)
RUNS: dict[str, dict] = {}


@app.template_filter("md")
def md(text):
    """Escape, then turn **bold** into <b> (the only markdown the pulse uses)."""
    import re
    from markupsafe import Markup, escape
    return Markup(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(escape(text))))


@app.get("/")
def index():
    return render_template("index.html", has_default=(BASE / "data" / "reviews.csv").exists())


@app.post("/analyze")
def analyze():
    f = request.files.get("csv")
    src = f if f and f.filename else BASE / "data" / "reviews.csv"
    try:
        res = pipeline.run(src, weeks=int(request.form.get("weeks", 12)))
    except Exception as e:  # show the problem instead of a stack trace
        return render_template("index.html", has_default=True, error=str(e)), 400
    notes, email = mcp.build_payloads(res, request.form.get("to", ""))
    gate = mcp.ApprovalGate("Append entry to notes doc + create email draft (no send)")
    RUNS[gate.id] = {"res": res, "notes": notes, "email": email, "gate": gate, "done": None}
    return redirect(url_for("review", rid=gate.id))


@app.get("/review/<rid>")
def review(rid):
    run = RUNS.get(rid) or abort(404)
    return render_template("review.html", rid=rid, **run, tools=mcp.list_tools())


@app.post("/review/<rid>/decide")
def decide(rid):
    run = RUNS.get(rid) or abort(404)
    gate = run["gate"]
    if gate.status != "pending":
        return redirect(url_for("review", rid=rid))
    approve = request.form.get("decision") == "approve"
    gate.decide(approve)                       # <-- the ONLY place approval is granted
    if approve:
        run["done"] = [
            ("append_to_notes", mcp.call_tool("append_to_notes", run["notes"], gate)),
            ("create_email_draft", mcp.call_tool("create_email_draft", run["email"], gate)),
        ]
        # standalone deliverable files (written only after approval)
        out = BASE / "outputs"; out.mkdir(exist_ok=True)
        r = run["res"]
        (out / "weekly_pulse.md").write_text(f"# Weekly Product Pulse - {r['date']}\n\n{r['pulse']}\n", encoding="utf-8")
        (out / "fee_explainer.md").write_text(f"# Fee Explainer - {r['date']}\n\n{r['explainer_text']}\n", encoding="utf-8")
    return redirect(url_for("review", rid=rid))


@app.get("/outputs/<path:name>")
def outputs(name):
    p = (BASE / "outputs" / name).resolve()
    if not str(p).startswith(str(BASE / "outputs")) or not p.is_file():
        abort(404)
    return p.read_text(encoding="utf-8"), 200, {"Content-Type": "text/plain; charset=utf-8"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
