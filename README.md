# Product Hunt Launch Traction Analytics

Predicting whether a Product Hunt launch achieves high traction (Top 20% vs. Rest, by upvotes relative to same-day competition) from pre-launch, controllable features using web-scraped leaderboard and product data. Supports data-driven launch timing, Hunter selection, and pre-launch presentation strategy for makers and marketers.
## Project Summary

**Business Problem:** Makers frequently launch on Product Hunt without a data-backed strategy, relying on guesswork for decisions (launch timing, Hunter selection, media presentation) that materially affect visibility and traction.

**Target Variable:** `is_top_20_percent` — binary, computed relative to each day's leaderboard (not a fixed global upvote threshold), since launch-day competition varies.

**Data Source:** Web scraping via Apify actors — a leaderboard scraper for core launch/vote data (name, tagline, votes, comments, topics, launch date) and a maker-focused scraper for Hunter/team information. Collected across ~15–25 daily leaderboards for 300–400+ records.

**Key Features:** launch day/hour, tagline and description length, media asset count, category/topics, Hunter track record, team size, presence of a separate Hunter (vs. self-submission).

**Planned EDA Finding:** Self-submitted launches (no separate Hunter) are expected to underperform hunted launches at comparable feature levels — a structural, non-quality-driven factor worth surfacing as a business insight.

**Limitation:** Upvotes are a popularity/visibility proxy, not a direct business outcome (revenue, retention, funding) — acknowledged explicitly in the final report rather than left implicit.
 