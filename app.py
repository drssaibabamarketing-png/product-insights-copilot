"""Flask UI: upload/choose CSV -> analyse -> preview -> APPROVE -> MCP actions."""
import os
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, url_for

import mcp_layer as mcp
import pipeline

BASE = Path(__file__).parent
from jinja2 import DictLoader

# Page templates are embedded so the app runs from a single file, with no templates/ folder needed.
TEMPLATES = {
    'base.html': r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Product Insights Copilot</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#14213d;--mut:#5b6478;--line:#e3e6ec;--acc:#0b6e4f;--warn:#b45309;--bad:#b42318}
@media(prefers-color-scheme:dark){:root{--bg:#0f1420;--card:#171e2e;--ink:#e8ecf5;--mut:#9aa5bd;--line:#2a3347;--acc:#3ecf9a;--warn:#f0b35a;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.1rem;margin:0 0 .6em}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}
.mut{color:var(--mut);font-size:.9rem}table{width:100%;border-collapse:collapse;font-size:.92rem}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.8rem;border:1px solid var(--line)}.ok{color:var(--acc)}.bad{color:var(--bad)}
pre{white-space:pre-wrap;font:inherit;margin:0}blockquote{margin:.4em 0;padding-left:12px;border-left:3px solid var(--acc)}
button,.btn{font:inherit;padding:10px 20px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.go{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600}button.go:hover{filter:brightness(1.1)}
.gate{border:2px solid var(--warn)}input[type=text],input[type=file],input[type=number]{font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);max-width:100%}
.steps{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}.steps span{padding:4px 12px;border-radius:99px;background:var(--card);border:1px solid var(--line);font-size:.85rem}
</style></head><body><main>{% block c %}{% endblock %}</main></body></html>
''',
    'index.html': r'''{% extends "base.html" %}{% block c %}
<h1>Product Insights Copilot</h1>
<p class="mut">Turns raw app-store reviews into a weekly product pulse and a support-ready fee explainer. Nothing is logged or drafted until you approve.</p>
<div class="steps"><span>1 Analyse reviews</span><span>2 Pulse + fee explainer</span><span>3 You approve</span><span>4 Notes entry + email draft</span></div>
{% if error %}<div class="card bad">{{ error }}</div>{% endif %}
<form class="card" method="post" action="/analyze" enctype="multipart/form-data">
<h2>Reviews CSV</h2>
<p class="mut">Columns: review text + rating (+ date, optional). {% if has_default %}Leave empty to use the bundled <code>data/reviews.csv</code>.{% endif %}</p>
<p><input type="file" name="csv" accept=".csv"></p>
<p>Look-back window (weeks): <input type="number" name="weeks" value="12" min="1" max="52" style="width:80px"></p>
<p>Email draft recipient (optional): <input type="text" name="to" placeholder="team@example.com"></p>
<button class="go">Analyse reviews</button></form>
{% endblock %}
''',
    'review.html': r'''{% extends "base.html" %}{% block c %}
<h1>Review &amp; approve</h1>
<p class="mut">{{ res.product }} · {{ res.info.rows_used }} reviews analysed{% if res.info.window %} · {{ res.info.window[0] }} to {{ res.info.window[1] }}{% else %} · <b class="bad">no date column in CSV, window not applied</b>{% endif %}{% if res.info.dropped_non_english %} · {{ res.info.dropped_non_english }} non-English reviews excluded{% endif %}</p>

<div class="card"><h2>Brief checks</h2>
{% for name, okk, detail in res.checks %}<span class="pill {{ 'ok' if okk else 'bad' }}">{{ '✓' if okk else '✗' }} {{ name }} ({{ detail }})</span> {% endfor %}</div>

<div class="card"><h2>Step 1 · Themes (max 5, top 3 marked)</h2>
<table><tr><th></th><th>Theme</th><th>Reviews</th><th>Share</th><th>Avg ★</th><th>1-2★</th></tr>
{% for t in res.themes %}<tr><td>{{ '★' if t.label in res.top_themes else '' }}</td><td>{{ t.label }}</td><td>{{ t.count }}</td><td>{{ t.share_pct }}%</td><td>{{ t.avg_rating }}</td><td>{{ t.negative_pct }}%</td></tr>{% endfor %}</table>
<h2 style="margin-top:14px">User quotes</h2>
{% for q in res.quotes %}<blockquote>"{{ q.quote }}" <span class="mut">({{ q.rating }}★ · {{ q.theme }})</span></blockquote>{% endfor %}
<h2 style="margin-top:14px">Fee / charge confusion detected</h2>
<p><b>{{ res.fee.fee }}</b>: {{ res.fee.mentions }} reviews ({{ res.fee.pct_of_reviews }}%), {{ res.fee.negative_mentions }} rated 1-2★.</p>
{% for q in res.fee.evidence_quotes %}<blockquote>"{{ q }}"</blockquote>{% endfor %}
<table><tr><th>Fee type scanned</th><th>Mentions</th><th>1-2★</th></tr>{% for c in res.fee.all_candidates %}<tr><td>{{ c.fee }}</td><td>{{ c.mentions }}</td><td>{{ c.negative_mentions }}</td></tr>{% endfor %}</table></div>

<div class="card"><h2>Step 2 · Weekly product pulse</h2><pre>{{ res.pulse | md }}</pre></div>
<div class="card"><h2>Step 3 · Fee explainer</h2><pre>{{ res.explainer_text | md }}</pre></div>

<div class="card gate"><h2>🔒 MCP approval gate</h2>
<p>Pending tool calls: {% for t in tools %}<code>{{ t.name }}</code> {% endfor %}</p>
<ul class="mut"><li><b>append_to_notes</b> payload: date, top_themes, weekly_pulse, identified_fee_issue, explanation_bullets, source_links</li>
<li><b>create_email_draft</b>: "{{ email.subject }}" (draft only, never sent)</li></ul>
<details><summary>Preview payloads</summary><pre>{{ notes | tojson(indent=2) }}</pre><hr><pre>{{ email.subject }}

{{ email.body }}</pre></details>
{% if gate.status == 'pending' %}
<form method="post" action="/review/{{ rid }}/decide" style="margin-top:12px"><button class="go" name="decision" value="approve">Approve &amp; run actions</button> <button name="decision" value="reject">Reject</button></form>
{% elif gate.status == 'approved' %}<p class="ok"><b>Approved {{ gate.decided_at }}.</b> Actions executed:</p>
<ul>{% for n, r in done %}<li><code>{{ n }}</code> → {{ r }}</li>{% endfor %}</ul>
<p><a href="/outputs/notes_log.md">notes_log.md</a> · <a href="/outputs/email_draft.txt">email_draft.txt</a> · <a href="/outputs/audit_log.jsonl">audit_log.jsonl</a></p>
{% else %}<p class="bad"><b>Rejected.</b> No tool was called.</p>{% endif %}</div>
<p><a href="/">← New analysis</a></p>
{% endblock %}
''',
}
app = Flask(__name__)
app.jinja_loader = DictLoader(TEMPLATES)
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
