"""Export recent Play Store reviews (with dates) to data/reviews.csv.
Usage: python scrape_reviews.py [app_id] [weeks]   (requires: pip install google-play-scraper pandas)
"""
import sys
from datetime import datetime, timedelta

import pandas as pd
from google_play_scraper import Sort, reviews

APP_ID = sys.argv[1] if len(sys.argv) > 1 else "com.nextbillion.groww"   # Groww
WEEKS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
cutoff = datetime.now() - timedelta(weeks=WEEKS)
rows, token = [], None
while True:
    batch, token = reviews(APP_ID, lang="en", country="in", sort=Sort.NEWEST, count=200, continuation_token=token)
    rows += batch
    if not batch or batch[-1]["at"] < cutoff or token is None or len(rows) > 6000:
        break
df = pd.DataFrame(rows)
df = df[df["at"] >= cutoff][["content", "score", "at", "thumbsUpCount", "appVersion"]]
df.columns = ["review_text", "rating", "date", "thumbs_up", "app_version"]
df.to_csv("data/reviews.csv", index=False)
print(f"saved {len(df)} reviews from {df['date'].min()} to {df['date'].max()} -> data/reviews.csv")
