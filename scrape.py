import os
import time
import requests
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===========================
# CONFIGURATION
# ===========================
API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
ACTOR_ID = "BN0Ukz5f8YV2nviFf"
START_DATE = "2026-07-17"
DAYS = 60
MAX_CONCURRENT_RUNS = 5
# ===========================

os.makedirs("producthunt_data", exist_ok=True)
headers = {
    "Content-Type": "application/json"
}

def scrape_date(date):
    filepath = f"producthunt_data/{date}.json"
    
    # Skip if file already exists and is non-empty
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        print(f"Skipping {date} (already scraped and has data)")
        return True

    print(f"Starting scraping for {date}...")
    body = {
        "mode": "daily",
        "date": date,
        "includeAll": True,
        "maxProducts": 100,
        "proxyConfiguration": {
            "useApifyProxy": True
        }
    }
    
    # 1. Trigger the run
    try:
        url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs?token={API_TOKEN}"
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        run_data = resp.json()["data"]
        run_id = run_data["id"]
        print(f"Run {run_id} started for {date}")
    except Exception as e:
        print(f"Failed to start run for {date}: {e}")
        return False

    # 2. Wait for completion
    status_url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={API_TOKEN}"
    retry_count = 0
    while True:
        try:
            status_resp = requests.get(status_url, timeout=30)
            status_resp.raise_for_status()
            status = status_resp.json()["data"]["status"]
            retry_count = 0  # reset retry count on success
        except Exception as e:
            retry_count += 1
            print(f"Error checking status for {date} (run {run_id}): {e}. Retry {retry_count}/5")
            if retry_count >= 5:
                print(f"Aborting wait for {date} (run {run_id}) due to continuous status check failures")
                return False
            time.sleep(5)
            continue

        if status == "SUCCEEDED":
            break
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            print(f"Run {run_id} for {date} finished with status: {status}")
            return False

        time.sleep(5)

    # 3. Retrieve dataset
    try:
        dataset_url = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={API_TOKEN}"
        dataset_resp = requests.get(dataset_url, timeout=60)
        dataset_resp.raise_for_status()
        
        # Double check data is valid JSON list
        data = dataset_resp.json()
        if not isinstance(data, list):
            print(f"Warning: Expected a JSON list from dataset for {date}, got: {type(data)}")
            
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        print(f"Successfully saved data for {date} ({len(data)} items)")
        return True
    except Exception as e:
        print(f"Failed to retrieve/save dataset for {date} (run {run_id}): {e}")
        return False

def main():
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    dates = [(start - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(DAYS)]
    
    print(f"Starting batch scrape for {DAYS} days from {START_DATE} using {MAX_CONCURRENT_RUNS} concurrent workers")
    
    success_count = 0
    failure_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RUNS) as executor:
        future_to_date = {executor.submit(scrape_date, date): date for date in dates}
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as exc:
                print(f"Date {date} generated an exception: {exc}")
                failure_count += 1

    print("\n====================================")
    print("Batch Scraping Process Completed")
    print(f"Total dates: {DAYS}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print("====================================")

if __name__ == "__main__":
    main()