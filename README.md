# Product Insights Copilot (Groww) - Milestone 2

Turns raw Play Store reviews into (1) a weekly product pulse for the product team and (2) a neutral fee explainer for support, then logs both through **MCP-style tools behind a human approval gate**.

## What fee issue was identified from reviews
**Brokerage & trade charges.** Users call brokerage "very high", compare it with other apps, and complain of "hidden" or "other" charges eating profit. The app counts fee-related mentions per fee type, picks the one with most 1-2★ mentions, and shows the evidence quotes. The explainer is chosen from that result, not hard-coded. Exit load, DP charges and auto square-off are also scanned.

## How to run
```bash
pip install -r requirements.txt
python scrape_reviews.py            # optional: fresh 12-week export with dates -> data/reviews.csv
python app.py                       # open http://localhost:5000
python -m unittest discover -s tests -v
```
`data/reviews.csv` is the default input: public Google Play reviews for Groww. `scrape_reviews.py` (requires `google-play-scraper`) exports the latest 8-12 weeks of dated reviews to that path. Any CSV can also be uploaded in the UI (columns: `review_text`/`content`, `rating`/`score`, optional `date`/`at`).

**Deploy (free):** push to GitHub → Render → New Web Service → pick the repo (`render.yaml` is included).

## Where MCP approval happens
- UI: the orange **MCP approval gate** card on the review page ("Approve & run actions" / "Reject").
- Code: `app.py` → `decide()` is the only place approval is granted. `mcp_layer.call_tool()` raises `ApprovalRequired` unless the gate is approved, so no tool can run without it.
- Tools (`mcp_layer.py`, exposed via `list_tools()` / `call_tool()`): `append_to_notes` (writes `outputs/notes_log.md` + `.jsonl`, with date, top_themes, weekly_pulse, identified_fee_issue, explanation_bullets, source_links) and `create_email_draft` (`outputs/email_draft.eml`, `X-Unsent`, **never sent**). Every executed call is written to `outputs/audit_log.jsonl`.
- This is the simulated approval allowed by the brief. To go live, replace the two function bodies with calls to Google Docs and Gmail MCP servers. The gate stays unchanged.

## How it works
1. **Intelligence:** clean text → TF-IDF → LSA → KMeans (k ≤ 5) → clusters named by lexicon match → top 3 by volume → one verbatim quote per top theme.
2. **Pulse:** template-filled from measured numbers; validated ≤250 words, 3 quotes, 3 actions.
3. **Explainer:** ≤6 facts-only bullets, 2 source links, "Last checked: <date>".
4. No LLM or API key is used, so results are reproducible and nothing can leak.

## Deliverables map
| Brief | Where |
|---|---|
| Weekly pulse (MD) | `outputs/weekly_pulse.md` (also `outputs/fee_explainer.md`) |
| Notes/Doc snippet | `outputs/notes_log.md`, `outputs/notes_log.jsonl` |
| Email draft | `outputs/email_draft.txt` / `.eml` |
| Evidence screenshots | `outputs/shot_*.png` |
| Reviews CSV sample | `data/reviews.csv` |
| Source list | `SOURCES.md` |

## Limitations
- Explainer facts are kept general and carry no rupee rates, since rates change; exact figures are on the linked pricing page. "Last checked" is the date the explainer was generated.
- Themes depend on the CSV; a lexicon names the clusters, so check the labels on new data.
