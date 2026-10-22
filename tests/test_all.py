import io, pathlib, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import app as webapp
import mcp_layer as mcp
import pipeline as p

CSV = ROOT / "data" / "reviews.csv"


class PipelineTests(unittest.TestCase):
    def test_meets_brief(self):
        r = p.run(CSV)
        self.assertTrue(all(ok for _, ok, _ in r["checks"]), r["checks"])
        self.assertLessEqual(len(r["themes"]), 5)
        self.assertEqual(len(r["quotes"]), 3)

    def test_quotes_are_verbatim_reviews(self):
        r = p.run(CSV)
        df, _ = p.load_reviews(CSV)
        corpus = set(df["text"])
        for q in r["quotes"]:
            self.assertIn(q["quote"], corpus)

    def test_fee_comes_from_reviews(self):
        r = p.run(CSV)
        self.assertGreater(r["fee"]["mentions"], 0)
        df, _ = p.load_reviews(CSV)
        for q in r["fee"]["evidence_quotes"]:
            self.assertIn(q, set(df["text"]))

    def test_missing_columns(self):
        with self.assertRaises(ValueError):
            p.load_reviews(io.StringIO("a,b\n1,2\n"))

    def test_date_window(self):
        rows = ["review_text,rating,date"]
        rows += [f"brokerage charges are very high for me {i},1,2026-01-1{i}" for i in range(5)]
        rows += [f"brokerage charges are very high lately {i},1,2026-09-1{i}" for i in range(5)]
        df, info = p.load_reviews(io.StringIO("\n".join(rows)), weeks=12)
        self.assertTrue(info["has_dates"])
        self.assertEqual(len(df), 5)


class McpTests(unittest.TestCase):
    def test_blocked_without_approval(self):
        g = mcp.ApprovalGate("x")
        with self.assertRaises(mcp.ApprovalRequired):
            mcp.call_tool("append_to_notes", {}, g)
        g.decide(False)
        with self.assertRaises(mcp.ApprovalRequired):
            mcp.call_tool("create_email_draft", {"subject": "s", "body": "b"}, g)


class WebFlowTests(unittest.TestCase):
    def test_approve_then_reject(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d)
            mcp.OUT = out
            c = webapp.app.test_client()
            resp = c.post("/analyze", data={"weeks": "12", "to": "a@b.com"})
            self.assertEqual(resp.status_code, 302)
            url = resp.headers["Location"]; rid = url.rsplit("/", 1)[1]
            page = c.get(url).get_data(as_text=True)
            self.assertIn("Approve", page)
            self.assertFalse((out / "notes_log.md").exists())          # nothing before approval
            c.post(f"/review/{rid}/decide", data={"decision": "approve"})
            self.assertTrue((out / "notes_log.md").exists())
            self.assertTrue((out / "email_draft.eml").exists())
            self.assertIn("Weekly Product Pulse + Customer Clarification", (out / "email_draft.txt").read_text())
            rid2 = c.post("/analyze", data={"weeks": "12"}).headers["Location"].rsplit("/", 1)[1]
            before = (out / "audit_log.jsonl").read_text().count("\n")
            c.post(f"/review/{rid2}/decide", data={"decision": "reject"})
            self.assertEqual((out / "audit_log.jsonl").read_text().count("\n"), before)


if __name__ == "__main__":
    unittest.main()
