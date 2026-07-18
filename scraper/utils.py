import logging
import os
import random
import asyncio
import pandas as pd
from typing import Tuple

def setup_logger():
    """
    Sets up a logger that outputs to both console and a log file 'scraper.log'.
    """
    logger = logging.getLogger("scraper")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler("scraper.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

async def random_delay(min_s: float = 2.0, max_s: float = 5.0):
    """
    Sleeps for a random duration between min_s and max_s.
    """
    delay = random.uniform(min_s, max_s)
    logging.getLogger("scraper").debug(f"Sleeping for {delay:.2f} seconds...")
    await asyncio.sleep(delay)

def load_data_and_check_progress(master_csv: str, enriched_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    """
    Loads the master dataframe and the enriched dataframe (if it exists) to enable resume support.
    Returns:
        master_df: The original dataframe.
        enriched_df: The enriched dataframe (either new or loaded).
        scraped_urls: A set of URLs that have already been enriched.
    """
    logger = logging.getLogger("scraper")
    
    # 1. Load original master CSV
    if not os.path.exists(master_csv):
        raise FileNotFoundError(f"Original master CSV not found at: {master_csv}")
    master_df = pd.read_csv(master_csv)
    
    # 2. Setup or Load enriched CSV
    if os.path.exists(enriched_csv):
        logger.info(f"Existing enriched file found at {enriched_csv}. Loading to resume progress.")
        try:
            enriched_df = pd.read_csv(enriched_csv)
            # Find URLs that have non-null descriptions (which indicates successful scraping)
            # or simply all URLs in the enriched file if it has the enriched columns
            if "description" in enriched_df.columns:
                completed_rows = enriched_df[enriched_df["description"].notna()]
                scraped_urls = set(completed_rows["url"].tolist())
            else:
                scraped_urls = set()
        except Exception as e:
            logger.warning(f"Error loading {enriched_csv}: {e}. Starting fresh.")
            enriched_df = master_df.copy()
            scraped_urls = set()
    else:
        logger.info(f"No existing enriched file found. Starting fresh.")
        enriched_df = master_df.copy()
        scraped_urls = set()
        
    return master_df, enriched_df, scraped_urls
