"""Product Insights Copilot - review intelligence + weekly pulse + fee explainer.

Deterministic and key-free: TF-IDF + KMeans for clustering, rule-based signal
detection, template-based generation with hard validators for the brief's limits.
"""
from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import Normalizer

MAX_THEMES = 5
PULSE_WORD_LIMIT = 250
EXPLAINER_BULLET_LIMIT = 6

# ---------------------------------------------------------------- loading
COLUMN_ALIASES = {
    "text": ["review_text", "content", "review", "text", "body"],
    "rating": ["rating", "score", "stars"],
    "date": ["date", "at", "review_date", "created_at"],
}


def _pick(df: pd.DataFrame, key: str):
    for c in COLUMN_ALIASES[key]:
        if c in df.columns:
            return c
    return None


def load_reviews(path_or_buf, weeks: int = 12) -> tuple[pd.DataFrame, dict]:
    """Load a reviews CSV into columns text/rating/date. Keeps the last `weeks` weeks
    when a date column exists. Returns (df, info)."""
    raw = pd.read_csv(path_or_buf)
    tcol, rcol, dcol = (_pick(raw, k) for k in ("text", "rating", "date"))
    if tcol is None or rcol is None:
        raise ValueError("CSV needs a review text column and a rating column "
                         f"(accepted: {COLUMN_ALIASES['text']} / {COLUMN_ALIASES['rating']}).")
    df = pd.DataFrame({"text": raw[tcol].astype(str), "rating": pd.to_numeric(raw[rcol], errors="coerce")})
    info = {"rows_in": len(raw), "has_dates": dcol is not None, "window": None}
    if dcol is not None:
        df["date"] = pd.to_datetime(raw[dcol], errors="coerce")
        df = df.dropna(subset=["date"])
        if len(df):
            end = df["date"].max()
            start = end - pd.Timedelta(weeks=weeks)
            df = df[df["date"] >= start]
            info["window"] = (start.date().isoformat(), end.date().isoformat())
    df = df.dropna(subset=["rating"])
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.split().str.len() >= 4].drop_duplicates("text").reset_index(drop=True)
    info["rows_used"] = len(df)
    return df, info


def clean_text(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t)                         # placeholder tokens like <URL>/<PERSON>
    t = re.sub(r"https?://\S+|\S+@\S+", " ", t)
    t = re.sub(r"[\u2b50\u2605\u2606]+", " ", t)           # star emoji
    return re.sub(r"\s+", " ", t).strip().strip("\"'“”").strip()


# ---------------------------------------------------------------- themes
# Lexicon used only to NAME clusters (clustering itself is unsupervised).
THEME_LEXICON = {
    "Pricing & charges confusion": r"charge|brokerage|brokrage|fee\b|fees|hidden|deduct|amc|loot|expens",
    "Order execution & trading reliability": r"order|limit|market price|execut|sell|square off|stop ?loss|f&o|option|trade|slippage",
    "App performance & bugs": r"crash|slow|lag|hang|bug|glitch|load|update|server|error|not working|freez",
    "Customer support & trust": r"support|service|customer|response|complain|resolve|scam|fraud|cheat|trust|ticket",
    "Withdrawals, KYC & account issues": r"withdraw|kyc|bank|account|payment|upi|credit|refund|block|login|otp|verify",
    "Mutual funds & SIP": r"mutual|\bsips?\b|\bfunds?\b|\bnav\b|portfolio|ipo",
    "Ease of use & positive experience": r"easy to use|easy|user.?friendly|beginner|simple|smooth|interface|great app|best app|nice|excellent|love",
}
_STOP = list(ENGLISH_STOP_WORDS | {"app", "groww", "aap", "hai", "ko", "ka", "ki", "se", "ho", "nahi"})


def cluster_reviews(df: pd.DataFrame, k: int = MAX_THEMES, seed: int = 42) -> tuple[pd.DataFrame, list[dict]]:
    """TF-IDF -> LSA (SVD) -> L2 norm -> KMeans (k<=5). Clusters are then named by
    assigning each one a distinct label from THEME_LEXICON (Hungarian matching)."""
    k = max(2, min(k, MAX_THEMES, len(df) // 5 or 2))
    vec = TfidfVectorizer(stop_words=_STOP, ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True)
    X = vec.fit_transform(df["text"])
    n_comp = max(2, min(40, X.shape[1] - 1, X.shape[0] - 1))
    Z = Normalizer(copy=False).fit_transform(TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(X))
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    df = df.copy()
    df["cluster"] = km.labels_
    dist = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    df["dist"] = dist
    terms = np.array(vec.get_feature_names_out())
    # top terms per cluster from mean TF-IDF
    top_terms = {}
    for c in range(k):
        m = np.asarray(X[(km.labels_ == c)].mean(axis=0)).ravel()
        top_terms[c] = terms[np.argsort(m)[::-1][:6]].tolist()
    # label assignment: density of lexicon hits per review, distinct label per cluster
    labels = list(THEME_LEXICON)
    S = np.zeros((k, len(labels)))
    for c in range(k):
        blob = df.loc[df["cluster"] == c, "text"].str.lower()
        for j, lab in enumerate(labels):
            S[c, j] = blob.str.contains(THEME_LEXICON[lab], regex=True).mean()
    base = df["text"].str.lower().apply(lambda x: [bool(re.search(THEME_LEXICON[l], x)) for l in labels]).tolist()
    base = np.maximum(np.mean(base, axis=0), 1e-3)   # corpus base rate per label
    rows, cols = linear_sum_assignment(-(S / base))   # lift over base rate, not raw density
    assigned = {int(r): labels[c] for r, c in zip(rows, cols)}
    themes = []
    for c in sorted(range(k), key=lambda c: -(df["cluster"] == c).sum()):
        sub = df[df["cluster"] == c]
        themes.append({
            "cluster": int(c), "label": assigned.get(c, "Other feedback: " + ", ".join(top_terms[c][:3])),
            "count": int(len(sub)), "share_pct": round(100 * len(sub) / len(df), 1),
            "avg_rating": round(float(sub["rating"].mean()), 2),
            "negative_pct": round(100 * float((sub["rating"] <= 2).mean()), 1),
            "top_terms": top_terms[c],
        })
    return df, themes


def rank_top_themes(themes: list[dict], n: int = 3) -> list[dict]:
    """Top themes by volume, so the pulse reflects what most users talk about."""
    return sorted(themes, key=lambda t: -t["count"])[:n]


# ---------------------------------------------------------------- quotes
def extract_quotes(df: pd.DataFrame, top_themes: list[dict], n: int = 3) -> list[dict]:
    """One verbatim review per top theme: must match the theme's vocabulary, be readable
    (50-200 chars), and sit closest to the cluster centre. Issue themes prefer 1-2 star reviews."""
    quotes, seen = [], set()
    for t in top_themes[:n]:
        sub = df[df["cluster"] == t["cluster"]]
        rx = THEME_LEXICON.get(t["label"])
        pool = sub[(sub["text"].str.len().between(50, 200))]
        if rx:
            rel = pool[pool["text"].str.lower().str.contains(rx, regex=True)]
            pool = rel if len(rel) else pool
        if t["negative_pct"] >= 50:
            neg = pool[pool["rating"] <= 2]
            pool = neg if len(neg) else pool
        if pool.empty:
            pool = sub
        for _, r in pool.sort_values("dist").iterrows():
            if r["text"] not in seen:
                seen.add(r["text"])
                quotes.append({"theme": t["label"], "quote": r["text"], "rating": int(r["rating"])})
                break
    return quotes


# ---------------------------------------------------------------- fee detection
FEE_TYPES = {
    "Brokerage & trade charges": r"brokerage|brokrage|broker charge|per order|per trade|rs ?\d+ ?(?:per|/)|\bcharges?\b|other charges|hidden charge|heavy charge|too high.*charge",
    "DP / depository charges on selling": r"\bdp\b|depository|demat charge",
    "Auto square-off / closing charges": r"square ?off|closing charge",
    "Mutual fund exit load / expense ratio": r"exit load|expense ratio|\bamc\b|\bter\b",
}


def detect_fee_issue(df: pd.DataFrame) -> dict:
    """Find the fee/charge users complain about most. Pure counting on the reviews."""
    low = df["text"].str.lower()
    results = []
    for name, rx in FEE_TYPES.items():
        mask = low.str.contains(rx, regex=True)
        sub = df[mask]
        results.append({
            "fee": name, "mentions": int(mask.sum()),
            "negative_mentions": int((sub["rating"] <= 2).sum()),
            "pct_of_reviews": round(100 * float(mask.mean()), 1),
            "_idx": sub.index.tolist(),
        })
    results.sort(key=lambda r: (-r["negative_mentions"], -r["mentions"]))
    top = results[0]
    confusion_rx = r"hidden|other charges|unclear|not transparent|ununderstandable|why|deduct|extra|loot|thief|cutting|negative balance|more than|compar"
    sub = df.loc[top["_idx"]]
    conf = sub[sub["text"].str.lower().str.contains(confusion_rx, regex=True)]
    pool = conf if len(conf) else sub
    pool = pool[(pool["text"].str.len() >= 40) & (pool["text"].str.len() <= 230)].sort_values("rating")
    fee_quotes = pool["text"].head(2).tolist()
    return {
        "fee": top["fee"], "mentions": top["mentions"], "negative_mentions": top["negative_mentions"],
        "pct_of_reviews": top["pct_of_reviews"], "confusion_mentions": int(len(conf)),
        "evidence_quotes": fee_quotes,
        "all_candidates": [{k: v for k, v in r.items() if k != "_idx"} for r in results],
    }


# ---------------------------------------------------------------- pulse
def wc(text: str) -> int:
    return len(re.findall(r"\S+", text))


def generate_pulse(themes_top: list[dict], quotes: list[dict], fee: dict, n_reviews: int,
                   period: str, avg_rating: float) -> str:
    t1, t2, t3 = (themes_top + [None] * 3)[:3]
    lines = [f"**Weekly Product Pulse | {period}** ({n_reviews} reviews, avg {avg_rating}/5)", "",
             "**Top themes**"]
    for t in themes_top:
        lines.append(f"- {t['label']}: {t['share_pct']}% of reviews, {t['negative_pct']}% rated 1-2 stars")
    lines += ["", "**What users say**"]
    for q in quotes:
        lines.append(f"- \"{_trim(q['quote'], 160)}\" ({q['rating']}★)")
    lines += ["", "**Key observation**",
              f"{fee['mentions']} reviews ({fee['pct_of_reviews']}%) mention {fee['fee'].lower()}, "
              f"{fee['negative_mentions']} of them 1-2 stars. Users compare the fee with other apps "
              "and are unsure what \"other\" charges apply on top of brokerage. "
              f"\"{t1['label']}\" is the largest theme.", "",
              "**Action ideas**",
              "1. Show a full charge breakup (brokerage, taxes, statutory charges) on the order screen before confirming.",
              "2. Add a plain-language \"why was I charged?\" link beside every ledger and contract-note line.",
              "3. Give support a standard fee explainer and track charge-related tickets weekly."]
    text = "\n".join(lines)
    return text


def _trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0].rstrip(".,!?") + "…"


# ---------------------------------------------------------------- explainer
# Facts are deliberately general (no rupee figures) because rates change; the
# pricing page link is where exact, current numbers live.
FEE_KB = {
    "Brokerage & trade charges": {
        "bullets": [
            "Brokerage is the fee the broker charges for each executed equity order; Groww publishes its current rates on its pricing page.",
            "Brokerage is not the only cost: statutory charges such as STT, exchange transaction charges, GST, SEBI turnover fees and stamp duty are added by regulation and are not set by the broker.",
            "Selling delivery shares also attracts depository (DP) charges, applied per stock sold per day.",
            "Charges apply per executed order, so splitting one trade into several orders can increase the total fee.",
            "Every charge is itemised in the contract note and ledger for each trade.",
            "Net profit on a trade equals gross profit minus brokerage and all statutory and DP charges.",
        ],
        "sources": [
            ("Groww pricing and charges", "https://groww.in/pricing"),
            ("Groww Help Centre", "https://groww.in/help"),
            ("SEBI (regulator) investor resources", "https://investor.sebi.gov.in/"),
        ],
    },
    "DP / depository charges on selling": {
        "bullets": [
            "DP charges are levied by the depository participant when shares are debited from the demat account on a sale.",
            "They apply per stock (ISIN) sold in a day, not per order.",
            "They are charged only on selling delivery shares, not on buying.",
            "Depositories (NSDL/CDSL) levy part of the amount; the broker adds its own portion.",
            "The charge appears as a separate line in the ledger and contract note.",
            "Rates can change; check the current pricing page.",
        ],
        "sources": [("Groww pricing and charges", "https://groww.in/pricing"),
                    ("Groww Help Centre", "https://groww.in/help"),
                    ("SEBI investor resources", "https://investor.sebi.gov.in/")],
    },
    "Auto square-off / closing charges": {
        "bullets": [
            "Intraday positions not closed by the user are closed automatically by the broker near market close.",
            "Auto square-off carries a charge in addition to brokerage and statutory charges.",
            "Manually closing the position before the cut-off avoids the auto square-off charge.",
            "Charges are itemised in the ledger.",
            "Cut-off times and rates are on the pricing page.",
            "Margin-funded or F&O positions can have additional rules.",
        ],
        "sources": [("Groww pricing and charges", "https://groww.in/pricing"),
                    ("Groww Help Centre", "https://groww.in/help"),
                    ("SEBI investor resources", "https://investor.sebi.gov.in/")],
    },
    "Mutual fund exit load / expense ratio": {
        "bullets": [
            "Exit load is a fee the fund house (AMC) deducts when units are redeemed within a set period.",
            "The exit load rule and period are specific to each scheme.",
            "It is deducted from the redemption amount, not billed separately.",
            "The expense ratio is a yearly running cost already reflected in the NAV.",
            "Both are disclosed in the scheme information document.",
            "Rates are set by the AMC, not the platform.",
        ],
        "sources": [("AMFI investor education", "https://www.amfiindia.com/investor-corner"),
                    ("Groww Help Centre", "https://groww.in/help"),
                    ("SEBI investor resources", "https://investor.sebi.gov.in/")],
    },
}


def generate_explainer(fee: dict, checked_on: date | None = None) -> dict:
    kb = FEE_KB[fee["fee"]]
    checked_on = checked_on or date.today()
    return {
        "fee": fee["fee"],
        "bullets": kb["bullets"][:EXPLAINER_BULLET_LIMIT],
        "sources": [{"name": n, "url": u} for n, u in kb["sources"][:2]],   # brief: 2 official links
        "last_checked": checked_on.strftime("%d %b %Y"),
    }


def explainer_text(exp: dict) -> str:
    out = [f"**{exp['fee']} - what to know**"] + [f"- {b}" for b in exp["bullets"]]
    out += ["", "Sources: " + " | ".join(f"{s['name']} ({s['url']})" for s in exp["sources"]),
            f"Last checked: {exp['last_checked']}"]
    return "\n".join(out)


# ---------------------------------------------------------------- validation
def validate(pulse: str, quotes: list, themes: list, exp: dict) -> list[tuple[str, bool, str]]:
    return [
        ("<= 5 themes", len(themes) <= MAX_THEMES, f"{len(themes)} themes"),
        ("3 real user quotes", len(quotes) == 3, f"{len(quotes)} quotes"),
        (f"Pulse <= {PULSE_WORD_LIMIT} words", wc(pulse) <= PULSE_WORD_LIMIT, f"{wc(pulse)} words"),
        ("3 action ideas", len(re.findall(r"^\d\. ", pulse, re.M)) == 3, "numbered actions"),
        (f"Explainer <= {EXPLAINER_BULLET_LIMIT} bullets", len(exp["bullets"]) <= EXPLAINER_BULLET_LIMIT, f"{len(exp['bullets'])} bullets"),
        ("2 official source links", len(exp["sources"]) == 2, f"{len(exp['sources'])} links"),
        ("'Last checked' present", bool(exp["last_checked"]), exp["last_checked"]),
    ]


def run(path_or_buf, weeks: int = 12, product: str = "Groww") -> dict:
    df, info = load_reviews(path_or_buf, weeks)
    df, themes = cluster_reviews(df)
    top = rank_top_themes(themes)
    quotes = extract_quotes(df, top)
    fee = detect_fee_issue(df)
    period = (f"{info['window'][0]} to {info['window'][1]}" if info["window"] else "date not in CSV")
    pulse = generate_pulse(top, quotes, fee, len(df), period, round(float(df["rating"].mean()), 2))
    exp = generate_explainer(fee)
    return {
        "product": product, "date": date.today().isoformat(), "info": info, "themes": themes,
        "top_themes": [t["label"] for t in top], "quotes": quotes, "fee": fee,
        "pulse": pulse, "explainer": exp, "explainer_text": explainer_text(exp),
        "checks": validate(pulse, quotes, themes, exp),
    }
