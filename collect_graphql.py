"""
Product Hunt GraphQL API Collector
Fetches 2500+ records via paginated GraphQL requests and saves:
  - data/raw/ph_graphql_raw.json  (flat list of node objects)
  - data/raw/ph_graphql.csv       (normalized flat CSV)

Rate limit: 6,250 credits/window. Each page costs ~50 credits = 125 pages max.
Strategy: auto-pause when credits are low, wait for reset, then resume.

Requires: PH_DEVELOPER_TOKEN in .env file
"""

import os
import sys
import json
import time
import csv
import statistics
from pathlib import Path

import requests

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
GRAPHQL_ENDPOINT = "https://api.producthunt.com/v2/api/graphql"
TARGET_RECORDS = 2500
PAGE_SIZE = 20
DELAY_SECONDS = 2.0         # Slightly more generous delay between pages
PROGRESS_EVERY = 100
SAVE_EVERY = 200            # Checkpoint every 200 records

# Rate limit safety: pause when remaining credits fall below this threshold
CREDIT_SAFETY_BUFFER = 200  # Pause if remaining < 200 credits
QUOTA_RESET_PADDING = 30    # Extra seconds to add to reset_in to be safe

RAW_JSON_PATH = Path("data/raw/ph_graphql_raw.json")
CSV_PATH = Path("data/raw/ph_graphql.csv")
ORIGINAL_CSV = Path("producthunt_final.csv")

# ──────────────────────────────────────────────
# Load token from .env
# ──────────────────────────────────────────────
def load_env():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

load_env()

PH_DEVELOPER_TOKEN = os.environ.get("PH_DEVELOPER_TOKEN", "")
if not PH_DEVELOPER_TOKEN:
    print("ERROR: PH_DEVELOPER_TOKEN is not set in .env")
    sys.exit(1)

# ──────────────────────────────────────────────
# GraphQL query
# ──────────────────────────────────────────────
QUERY = """
query GetPosts($after: String) {
  posts(order: VOTES, first: 20, after: $after) {
    edges {
      node {
        id
        name
        tagline
        description
        votesCount
        commentsCount
        createdAt
        featuredAt
        media { type }
        makers { id name }
        topics { edges { node { name } } }
        website
        reviewsCount
        reviewsRating
      }
    }
    pageInfo {
      endCursor
      hasNextPage
    }
  }
}
"""

# ──────────────────────────────────────────────
# Fetch a single page — returns (nodes, cursor, has_next, credits_remaining, reset_in)
# Never raises; on failure returns (None, ...) 
# ──────────────────────────────────────────────
def fetch_page(after_cursor=None, max_retries=3):
    headers = {
        "Authorization": f"Bearer {PH_DEVELOPER_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "query": QUERY,
        "variables": {"after": after_cursor} if after_cursor else {}
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                GRAPHQL_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30
            )

            # Extract rate limit headers (always present)
            credits_remaining = int(resp.headers.get("x-rate-limit-remaining", 9999))
            reset_in = int(resp.headers.get("x-rate-limit-reset", 60))

            if resp.status_code == 401:
                print(f"\nERROR 401: Unauthorized. Check your PH_DEVELOPER_TOKEN.")
                sys.exit(1)

            if resp.status_code == 429:
                wait = reset_in + QUOTA_RESET_PADDING
                print(f"\n  [Rate limit] Quota exhausted. Waiting {wait}s for reset...")
                time.sleep(wait)
                # After waiting, retry this same request
                continue

            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                print(f"\n  GraphQL errors on attempt {attempt}: {data['errors']}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                return None, None, False, credits_remaining, reset_in

            posts_data = data["data"]["posts"]
            nodes = [edge["node"] for edge in posts_data["edges"]]
            page_info = posts_data["pageInfo"]
            return nodes, page_info["endCursor"], page_info["hasNextPage"], credits_remaining, reset_in

        except requests.exceptions.Timeout:
            print(f"\n  Timeout on attempt {attempt}/{max_retries}")
            if attempt < max_retries:
                time.sleep(10)
        except Exception as e:
            print(f"\n  Error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)

    return None, None, False, 0, 60

# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────
def normalize_node(node):
    media_count = len(node.get("media") or [])
    makers = node.get("makers") or []
    team_size = len(makers)
    topics_list = [
        edge["node"]["name"]
        for edge in (node.get("topics") or {}).get("edges", [])
    ]
    return {
        "id": node.get("id"),
        "name": node.get("name"),
        "tagline": node.get("tagline"),
        "description": node.get("description"),
        "votesCount": node.get("votesCount"),
        "commentsCount": node.get("commentsCount"),
        "createdAt": node.get("createdAt"),
        "featuredAt": node.get("featuredAt"),
        "media_count": media_count,
        "team_size": team_size,
        "topics": ", ".join(topics_list),
        "website": node.get("website"),
        "reviewsCount": node.get("reviewsCount"),
        "reviewsRating": node.get("reviewsRating"),
    }

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
def print_summary(rows):
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total records (deduplicated): {len(rows)}")

    dates = sorted([r["createdAt"] for r in rows if r.get("createdAt")])
    if dates:
        print(f"Date range:  {dates[0]}  →  {dates[-1]}")

    votes = [r["votesCount"] for r in rows if r.get("votesCount") is not None]
    if votes:
        print(f"\nvotesCount:  min={min(votes)}  "
              f"median={statistics.median(votes):.1f}  "
              f"mean={statistics.mean(votes):.2f}  "
              f"max={max(votes)}")

    columns = ["id", "name", "tagline", "description", "votesCount", "commentsCount",
               "createdAt", "featuredAt", "media_count", "team_size", "topics",
               "website", "reviewsCount", "reviewsRating"]
    print(f"\nNull/empty counts per column:")
    for col in columns:
        nulls = sum(1 for r in rows if not r.get(col) and r.get(col) != 0)
        print(f"  {col:<20} {nulls:>5}")
    print("=" * 60)

# ──────────────────────────────────────────────
# Incremental save
# ──────────────────────────────────────────────
def save_raw_json(nodes):
    RAW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    # Pre-run row count check
    original_count_before = None
    if ORIGINAL_CSV.exists():
        with open(ORIGINAL_CSV, "r", encoding="utf-8") as f:
            original_count_before = sum(1 for _ in f) - 1
        print(f"Original scraped CSV ({ORIGINAL_CSV}): {original_count_before} rows (pre-run)")

    RAW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting Product Hunt GraphQL collection...")
    print(f"Target: {TARGET_RECORDS}+ records | Page: {PAGE_SIZE} | Delay: {DELAY_SECONDS}s")
    print(f"Strategy: auto-pause when credits < {CREDIT_SAFETY_BUFFER}, wait for reset\n")

    all_nodes = []
    cursor = None
    page_num = 0
    last_milestone = 0
    quota_pauses = 0

    while len(all_nodes) < TARGET_RECORDS:
        result = fetch_page(after_cursor=cursor)
        nodes, next_cursor, has_next, credits_remaining, reset_in = result

        if nodes is None:
            print(f"\nStopping — API returned no data after all retries.")
            break

        all_nodes.extend(nodes)
        cursor = next_cursor
        page_num += 1
        total = len(all_nodes)

        # Progress reporting
        milestone = (total // PROGRESS_EVERY) * PROGRESS_EVERY
        if milestone > last_milestone:
            print(f"  [{total:>4} records] page {page_num:>3} | credits remaining: {credits_remaining}")
            last_milestone = milestone

        # Incremental save
        if total % SAVE_EVERY == 0:
            save_raw_json(all_nodes)
            print(f"  [Checkpoint] {total} records saved to disk")

        # Proactive quota pause — don't wait for a 429
        if 0 < credits_remaining < CREDIT_SAFETY_BUFFER:
            wait = reset_in + QUOTA_RESET_PADDING
            quota_pauses += 1
            print(f"\n  [Quota pause #{quota_pauses}] Only {credits_remaining} credits left. "
                  f"Pausing {wait}s for reset... (have {total} records so far)")
            time.sleep(wait)
            print(f"  Resuming collection...\n")

        if not has_next:
            print(f"\nAPI has no more pages. Total fetched: {total}")
            break

        time.sleep(DELAY_SECONDS)

    print(f"\nCollection complete. {len(all_nodes)} raw records fetched.")

    # Final save
    print(f"Saving raw JSON → {RAW_JSON_PATH}...")
    save_raw_json(all_nodes)

    # Normalize + dedup
    print(f"Normalizing and deduplicating...")
    normalized = [normalize_node(n) for n in all_nodes]
    seen, deduped = set(), []
    for row in normalized:
        rid = row.get("id")
        if rid not in seen:
            seen.add(rid)
            deduped.append(row)
    print(f"  Before dedup: {len(normalized)} | After dedup: {len(deduped)}")

    # Save CSV
    columns = ["id", "name", "tagline", "description", "votesCount", "commentsCount",
               "createdAt", "featuredAt", "media_count", "team_size", "topics",
               "website", "reviewsCount", "reviewsRating"]
    print(f"Saving CSV → {CSV_PATH}...")
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(deduped)
    print(f"  Saved {len(deduped)} rows.")

    # Post-run guard
    if ORIGINAL_CSV.exists():
        with open(ORIGINAL_CSV, "r", encoding="utf-8") as f:
            original_count_after = sum(1 for _ in f) - 1
        status = "✓ UNCHANGED" if original_count_after == original_count_before else "⚠ CHANGED!"
        print(f"\nOriginal CSV row count after run: {original_count_after} [{status}]")

    print_summary(deduped)

if __name__ == "__main__":
    main()
