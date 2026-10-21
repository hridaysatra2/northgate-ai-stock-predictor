# News Data Availability Audit — Task 3C (audit ONLY)

No sentiment modelling. No training. No datasets modified. No full-dataset download.
Research period: 2016-01-01..2026-08-31. Universe: AAPL MSFT JPM JNJ PG XOM HD V CAT KO.
Retrieval timestamp (UTC): 2026-09-24T19:49:01Z. Library: yfinance==1.7.0. No news API keys in environment.

## Sources considered

### 1. yfinance `Ticker.news` (Yahoo Finance) — live-tested, NOT suitable
- Access: `yfinance.Ticker(t).news`; verified to be a parameterless property, so historical queries are impossible by construction. Free, no key.
- Sample (small, AAPL only): 10 items, pubDates 2026-09-23T19:50:58Z..2026-09-24T19:13:00Z — ALL after the 2026-08-31 research cutoff. Nothing from the research period.
- Timestamps: second-precision, explicit UTC (`...Z`); convertible to America/New_York, so session mapping would be mechanically possible — but there is no history to map.
- Ticker mapping: WEAK. No per-article ticker/symbol field; association is only "queried ticker", and the AAPL feed contained Qualcomm (QCOM) stories (contamination).
- Title/headline: yes (`title` + `summary`). Body: NO (no body/fulltext field on any item).
- Verdict: unsuitable (recent-only, no pagination, no body, contaminated ticker association).

### 2. GDELT 2.1 DOC API, artlist mode — live-tested (tiny), NOT suitable
- Access: `GET https://api.gdeltproject.org/api/v2/doc/doc?query=..&mode=artlist&maxrecords=..&format=json&startdatetime=..&enddatetime=..`. Free, no key.
- Sample (tiny, one query): `query=Apple`, window 2020-01-01..2020-01-02, maxrecords=3 → 3 articles, fields `url/title/seendate/socialimage/domain/language/sourcecountry`.
- Coverage: DOC 2.1 archive starts 2017, so 2016 is missing; keyword/language bias unquantified. Partial at best.
- Timestamps: `seendate` is minute-precision UTC of when GDELT *saw* the article — NOT the publisher's publication time. Insufficient for before/after-close mapping.
- Ticker mapping: ABSENT. Keyword search only. Sample demonstrated company-vs-commodity ambiguity (Chinese-language hits including an apple-futures blog for query "Apple").
- Body: NO (title + URL only; body would require per-URL scraping — unreliable and out of scope).
- Restrictions: HTTP 429 rate-limit hit on the first attempt; succeeded after a 20 s backoff. Quota is undocumented.
- Verdict: unsuitable (no entities, no pubtime, no body, rate-limited, incomplete period).

### 3. NewsAPI — assessed from public capability, NOT live-tested, NOT suitable here
- No `NEWS_API_KEY` in environment, so no live test and no download was attempted.
- Free plan covers roughly the last month only; full 2016-2026 archive needs a paid plan. Body is truncated content only.
- Verdict: unsuitable in this environment (no credential; free tier cannot cover the period).

### 4. Alpha Vantage News & Sentiment — assessed from public capability, NOT live-tested, NOT suitable here
- No API key in environment; no live test, no download attempted.
- Documented fit is closest in principle (ticker entities, Eastern timestamps, deep history on premium tiers), but free-tier rate limits (~25 req/day) cannot backfill 10 tickers × 10.7 years, and vendor sentiment scores must never be silently substituted for our own modelling.
- Verdict: unverifiable and unsuitable without a credential plus timestamp/entity validation.

## Timestamp quality
| source | precision | timezone | is it pubtime? |
|---|---|---|---|
| yfinance | second | UTC explicit | yes, but recent-only |
| GDELT | minute | UTC explicit | NO (crawl-seen time) |
| NewsAPI / Alpha Vantage | documented only | documented only | unverified |

## Session-mapping investigation (rules defined, applicability assessed)
Market: US equities, regular session 09:30–16:00 America/New_York.
- Before close (day t, ≤16:00 ET) → prediction timestamp t (available by close t).
- After close (day t, >16:00 ET) → prediction timestamp t+1.
- Weekend (Sat/Sun) → next Monday (or next trading day) as pre-open information.
- Exchange holiday → next trading session.
These rules require second-precision publication timestamps with known timezone plus a holiday calendar. yfinance clears the timestamp bar but has no history; GDELT fails it (`seendate` ≠ pubtime); key-gated sources are unverifiable here. No mapping was implemented (audit only).

## Limitations
- Internet used minimally: one 10-item yfinance read + two tiny GDELT queries (first 429'd, one 3-item retry). No bulk download.
- NewsAPI/Alpha Vantage rows rest on public capability knowledge, flagged `live_tested=false`; they were not accessed without credentials.
- Sample proves the negative: yfinance's newest-possible data postdates the research cutoff, and GDELT's keyword search demonstrably mixes entities.

## Suitability for leakage-free Task 3C
No suitable source. Nothing tested covers 2016-01-01..2026-08-31 with publication timestamps AND ticker-level association AND reproducible free access. No synthetic, current-as-history, or unrelated-sentiment data was substituted.

## Recommended next step (evidence-based)
STOP before sentiment modelling. Either (a) procure a licensed, entity-resolved, timestamped feed (or key-gated premium tier) and re-audit timestamps/entities before any modelling, or (b) formally reduce scope (e.g. forward-only collection from today, or a shorter window an accessible archive covers). Any path requires a fresh availability re-audit before Task 3C modelling begins.

## Files
`data/processed/news_availability/news_source_audit.json`, `news_source_audit.csv`, `NEWS_DATA_AVAILABILITY.md` — the only files created. Raw, features, Task 2C/2D/3A/3B outputs untouched.
