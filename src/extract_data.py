"""
src/extract_data.py
===================
Pull EV charging station (POI) data from the Open Charge Map API.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.openchargemap.io/v3"
POI_ENDPOINT = f"{BASE_URL}/poi/"
REF_ENDPOINT = f"{BASE_URL}/referencedata/"

COUNTRY_CODES = [
    "AU", "AT", "BE", "CA", "CN", "DK", "FI", "FR", "DE", "IN", "IE", "IT",
    "JP", "KR", "NL", "NZ", "NO", "PL", "PT", "ES", "SE", "CH", "GB", "US", "ZA"
]

MAX_RESULTS_PER_COUNTRY = 500
INTER_REQUEST_DELAY = 0.6
USER_AGENT = "EV-Charger-Reliability-Analytics/1.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_session(api_key: str) -> requests.Session:
    """Return a requests.Session pre-configured with auth headers."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "X-API-Key": api_key})
    return session


def fetch_reference_data(session: requests.Session) -> dict:
    """Pull referencedata once and return the parsed JSON."""
    logger.info("Fetching reference data...")
    resp = session.get(REF_ENDPOINT, params={"output": "json"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_poi_for_country(
    session: requests.Session,
    country_code: str,
    api_key: str,
    max_results: int = MAX_RESULTS_PER_COUNTRY,
) -> list[dict]:
    """Fetch POI records for a single ISO country code."""
    params = {
        "output": "json",
        "countrycode": country_code,
        "maxresults": max_results,
        "compact": "false",
        "verbose": "true",
        "key": api_key,
    }
    resp = session.get(POI_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    records = resp.json()
    if not isinstance(records, list):
        return []
    return records


def extract(api_key: str) -> None:
    """Run the full extraction pipeline and save data/metadata."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = _build_session(api_key)

    ref_data = fetch_reference_data(session)
    time.sleep(INTER_REQUEST_DELAY)

    ref_path = RAW_DIR / "reference_data.json"
    with open(ref_path, "w", encoding="utf-8") as fh:
        json.dump(ref_data, fh, indent=2, ensure_ascii=False)

    status_types = ref_data.get("StatusTypes", [])
    st_path = RAW_DIR / "reference_status_types.json"
    with open(st_path, "w", encoding="utf-8") as fh:
        json.dump(status_types, fh, indent=2, ensure_ascii=False)

    all_records: list[dict] = []
    per_country_counts: dict[str, int] = {}

    logger.info("Starting POI extraction across %d countries...", len(COUNTRY_CODES))

    for country_code in tqdm(COUNTRY_CODES, desc="Countries", unit="country"):
        try:
            records = fetch_poi_for_country(session, country_code, api_key)
            per_country_counts[country_code] = len(records)

            country_path = RAW_DIR / f"raw_poi_{country_code}_{timestamp}.json"
            with open(country_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, ensure_ascii=False)

            all_records.extend(records)
        except requests.RequestException as exc:
            logger.error("Failed to fetch %s: %s", country_code, exc)

        time.sleep(INTER_REQUEST_DELAY)

    merged_path = RAW_DIR / "raw_poi_all.json"
    with open(merged_path, "w", encoding="utf-8") as fh:
        json.dump(all_records, fh, ensure_ascii=False)

    manifest = {
        "extracted_at": timestamp,
        "total_records": len(all_records),
        "countries": per_country_counts,
        "max_results_per_country": MAX_RESULTS_PER_COUNTRY,
        "files": {
            "reference_data": str(ref_path),
            "reference_status_types": str(st_path),
            "merged_poi": str(merged_path),
        },
    }
    manifest_path = RAW_DIR / f"extraction_manifest_{timestamp}.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Extraction complete. Merged output saved → %s", merged_path)


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OCM_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OCM_API_KEY is not set. "
            "Copy .env.example to .env and add your Open Charge Map API key."
        )
    extract(api_key)
