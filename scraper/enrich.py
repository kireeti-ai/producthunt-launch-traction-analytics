import asyncio
import os
import sys
import logging
import argparse
import random
import pandas as pd
from typing import Optional
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

from .models import EnrichedProduct
from .parser import parse_product_page
from .utils import setup_logger, random_delay, load_data_and_check_progress

async def enrich_process(
    master_csv: str, 
    enriched_csv: str, 
    save_every: int, 
    max_items: Optional[int], 
    delay_min: float, 
    delay_max: float
):
    logger = setup_logger()
    logger.info("Initializing Product Hunt Enrichment Scraper")
    
    # 1. Load data and setup resume support
    try:
        master_df, enriched_df, completed_urls = load_data_and_check_progress(master_csv, enriched_csv)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)
        
    logger.info(f"Loaded master dataset: {len(master_df)} rows")
    logger.info(f"Already completed: {len(completed_urls)} URLs")
    
    # Ensure enriched columns exist in the dataframe
    new_cols = [
        "description", "launch_timestamp", "launch_hour", 
        "hunter_name", "hunter_profile_url", "maker_names", 
        "maker_profile_urls", "team_size", "media_count", 
        "website_url", "product_domain", "is_featured", 
        "self_launched", "description_length", "tagline_length"
    ]
    for col in new_cols:
        if col not in enriched_df.columns:
            # Initialize with None or appropriate defaults
            if col in ["team_size", "media_count", "description_length", "tagline_length"]:
                enriched_df[col] = 0
            elif col in ["is_featured", "self_launched"]:
                enriched_df[col] = False
            else:
                enriched_df[col] = None

    # Identify pending rows
    pending_rows = enriched_df[~enriched_df["url"].isin(completed_urls)]
    total_pending = len(pending_rows)
    logger.info(f"Total URLs remaining to scrape: {total_pending}")
    
    if total_pending == 0:
        logger.info("No pending URLs to scrape. Exiting.")
        return

    # Apply max_items limit if specified
    if max_items is not None and max_items > 0:
        pending_rows = pending_rows.head(max_items)
        logger.info(f"Limited run to next {len(pending_rows)} pending items")

    # Define user data dir for persistent cookies and session state
    user_data_dir = os.path.abspath("playwright_user_data")
    os.makedirs(user_data_dir, exist_ok=True)
    
    # 2. Launch Playwright
    async with Stealth().use_async(async_playwright()) as p:
        logger.info(f"Launching persistent browser context using directory: {user_data_dir}")
        context = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,  # Must be non-headless so user can complete Turnstile if needed
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox"
            ],
            viewport={"width": 1280, "height": 800}
        )
        
        page = context.pages[0] if context.pages else await context.new_page()
        
        scraped_count = 0
        
        try:
            for idx, row in pending_rows.iterrows():
                url = row["url"]
                tagline = row["tagline"]
                logger.info(f"[{scraped_count + 1}/{len(pending_rows)}] Scraping product URL: {url}")
                
                # Navigate to the page
                try:
                    # We wrap goto in a timeout and domcontentloaded check
                    await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                    
                    # Handle potential Cloudflare Turnstile verification page
                    turnstile_detected = False
                    for attempt in range(12):  # Check for 60 seconds max
                        title = await page.title()
                        if "Just a moment" in title or "Verify" in title:
                            if not turnstile_detected:
                                logger.warning("Cloudflare Turnstile challenge page detected!")
                                turnstile_detected = True
                            logger.info(f"Waiting for human verification... Attempt {attempt+1}/12. Please solve the challenge in the browser window.")
                            await page.wait_for_timeout(5000)
                        else:
                            if turnstile_detected:
                                logger.info("Turnstile challenge bypassed successfully!")
                            break
                    
                    # Wait 3 more seconds for hydration and page state to settle
                    await page.wait_for_timeout(3000)
                    
                    # Extract page source
                    html_content = await page.content()
                    
                    # Parse the page
                    enriched_prod = parse_product_page(html_content, url)
                    # Compute derived fields
                    enriched_prod.compute_derived(tagline)
                    
                    # Log scraped results summary
                    logger.info(
                        f"  Scraped: desc_len={enriched_prod.description_length}, "
                        f"makers={len(enriched_prod.maker_names)}, media={enriched_prod.media_count}, "
                        f"website={enriched_prod.website_url}"
                    )
                    
                    # Update enriched dataframe
                    res_dict = enriched_prod.to_dict()
                    for k, v in res_dict.items():
                        # Set using locate row index
                        enriched_df.at[idx, k] = v
                        
                    completed_urls.add(url)
                    scraped_count += 1
                    
                except Exception as page_exc:
                    logger.error(f"Error scraping {url}: {page_exc}")
                    # Save error state by updating description to a placeholder or keeping it None
                    # We continue scraping other pages
                    
                # Periodic Save
                if scraped_count > 0 and scraped_count % save_every == 0:
                    logger.info(f"Saving progress to {enriched_csv}...")
                    enriched_df.to_csv(enriched_csv, index=False)
                    
                # Inter-request delay to mimic human behavior
                if scraped_count < len(pending_rows):
                    await random_delay(delay_min, delay_max)
                    
        except KeyboardInterrupt:
            logger.warning("Scraper process interrupted by user.")
        finally:
            # 3. Final Save and Cleanup
            logger.info("Closing browser context and performing final save...")
            await context.close()
            enriched_df.to_csv(enriched_csv, index=False)
            logger.info(f"Progress saved successfully. Enriched dataset contains {len(completed_urls)} enriched rows.")

def main():
    parser = argparse.ArgumentParser(description="Product Hunt Product Page Data Enrichment Scraper")
    parser.add_argument("--master", default="producthunt_master.csv", help="Path to original master CSV file")
    parser.add_argument("--enriched", default="producthunt_enriched.csv", help="Path to output enriched CSV file")
    parser.add_argument("--save-every", type=int, default=10, help="Save progress to CSV every N pages scraped")
    parser.add_argument("--max-items", type=int, default=None, help="Max number of items to scrape in this run")
    parser.add_argument("--delay-min", type=float, default=2.0, help="Minimum random delay in seconds between page requests")
    parser.add_argument("--delay-max", type=float, default=5.0, help="Maximum random delay in seconds between page requests")
    args = parser.parse_args()

    asyncio.run(
        enrich_process(
            master_csv=args.master,
            enriched_csv=args.enriched,
            save_every=args.save_every,
            max_items=args.max_items,
            delay_min=args.delay_min,
            delay_max=args.delay_max
        )
    )

if __name__ == "__main__":
    main()
