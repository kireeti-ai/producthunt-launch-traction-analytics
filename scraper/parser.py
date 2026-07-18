import json
import re
import logging
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Optional, Dict, Any, List
from .models import EnrichedProduct

logger = logging.getLogger("scraper")

def parse_iso_hour(iso_str: Optional[str]) -> Optional[int]:
    """
    Parses launch hour from ISO timestamp string.
    Supports ISO formats like '2026-02-23T00:01:00-08:00' or '2026-02-23T00:01:00Z'.
    """
    if not iso_str:
        return None
    try:
        # Standard fromisoformat in Python 3.11 handles offset suffix like -08:00
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.hour
    except Exception as e:
        logger.debug(f"Failed to parse launch hour from {iso_str}: {e}")
        # Try manual regex parsing as fallback
        match = re.search(r'T(\d{2}):', iso_str)
        if match:
            return int(match.group(1))
    return None

def extract_json_from_apollo_js(js_content: str) -> Optional[dict]:
    """
    Extracts and parses JSON from ApolloSSRDataTransport JS push script content.
    Replaces ':undefined' or ': undefined' with ':null' to validate JSON.
    """
    start_idx = js_content.find('{')
    end_idx = js_content.rfind('}') + 1
    if start_idx != -1 and end_idx != -1:
        json_str = js_content[start_idx:end_idx]
        json_str = json_str.replace(":undefined", ":null").replace(": undefined", ": null")
        try:
            return json.loads(json_str)
        except Exception as e:
            logger.debug(f"Apollo JS JSON parsing failed: {e}")
    return None

def find_key_recursively(obj: Any, target_key: str) -> List[Any]:
    """
    Helper to search a nested dictionary/list for a specific key.
    Returns list of all values matching target_key.
    """
    results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target_key:
                results.append(v)
            results.extend(find_key_recursively(v, target_key))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(find_key_recursively(item, target_key))
    return results

def extract_product_from_apollo_state(merged_rehydrate: dict) -> Optional[Dict[str, Any]]:
    """
    Searches the merged Apollo state for the main Product dictionary.
    We look for a dict where __typename is Product and which contains key fields.
    """
    candidate_products = []
    
    def collect_products(obj):
        if isinstance(obj, dict):
            if obj.get("__typename") == "Product" and "id" in obj:
                candidate_products.append(obj)
            for v in obj.values():
                collect_products(v)
        elif isinstance(obj, list):
            for item in obj:
                collect_products(item)

    collect_products(merged_rehydrate)
    
    if not candidate_products:
        return None
        
    # Sort candidate products by number of keys to get the most detailed representation
    candidate_products.sort(key=len, reverse=True)
    return candidate_products[0]

def parse_product_page(html_content: str, url: str) -> EnrichedProduct:
    """
    Main entry point for parsing the HTML page. 
    Implements hybrid parsing: JSON-LD first, Apollo SSR second, DOM selectors third.
    """
    soup = BeautifulSoup(html_content, "lxml")
    
    # Initialize enriched product
    product = EnrichedProduct(url=url)
    
    # Track which methods successfully extract fields
    extracted_makers = False
    
    # ====================================================
    # 1. PARSE JSON-LD (Extremely stable for metadata)
    # ====================================================
    json_ld_tags = soup.find_all("script", type="application/ld+json")
    for tag in json_ld_tags:
        content = tag.string or ""
        try:
            data = json.loads(content)
            
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                types = item.get("@type", [])
                types = [types] if isinstance(types, str) else types
                
                if "Product" in types or "MobileApplication" in types:
                    # 1. Description
                    if not product.description and item.get("description"):
                        product.description = item.get("description")
                        
                    # 2. Launch Timestamp (datePublished)
                    if not product.launch_timestamp and item.get("datePublished"):
                        product.launch_timestamp = item.get("datePublished")
                        product.launch_hour = parse_iso_hour(product.launch_timestamp)
                        
                    # 3. Makers (author field)
                    authors = item.get("author", [])
                    authors = [authors] if isinstance(authors, dict) else authors
                    
                    maker_names = []
                    maker_urls = []
                    for author in authors:
                        if isinstance(author, dict) and author.get("name"):
                            maker_names.append(author.get("name"))
                            if author.get("url"):
                                maker_urls.append(author.get("url"))
                                
                    if maker_names:
                        product.maker_names = maker_names
                        product.maker_profile_urls = maker_urls
                        extracted_makers = True
        except Exception as e:
            logger.debug(f"Error parsing JSON-LD schema block: {e}")

    # ====================================================
    # 2. PARSE APOLLO SSR STATE (For rich dynamic elements)
    # ====================================================
    try:
        # Merge all Apollo rehydrate states
        merged_rehydrate = {}
        scripts = soup.find_all("script")
        for s in scripts:
            content = s.string or ""
            if "ApolloSSRDataTransport" in content and "rehydrate" in content:
                data = extract_json_from_apollo_js(content)
                if data and "rehydrate" in data:
                    merged_rehydrate.update(data["rehydrate"])

        if merged_rehydrate:
            apollo_prod = extract_product_from_apollo_state(merged_rehydrate)
            if apollo_prod:
                logger.debug("Successfully located Product object in Apollo SSR state.")
                
                # Fallback Description
                if not product.description and apollo_prod.get("description"):
                    product.description = apollo_prod.get("description")
                    
                # Fallback Website URL
                if not product.website_url and apollo_prod.get("websiteUrl"):
                    product.website_url = apollo_prod.get("websiteUrl")
                    
                # Domain
                if apollo_prod.get("cleanUrl"):
                    product.product_domain = apollo_prod.get("cleanUrl")
                    
                # Fallback Launch Timestamp & Featured status from latestLaunch Post node
                latest_launch = apollo_prod.get("latestLaunch")
                if latest_launch and isinstance(latest_launch, dict):
                    if not product.launch_timestamp and latest_launch.get("createdAt"):
                        product.launch_timestamp = latest_launch.get("createdAt")
                        product.launch_hour = parse_iso_hour(product.launch_timestamp)
                    
                    launch_state = latest_launch.get("launchState")
                    featured_at = latest_launch.get("featuredAt")
                    if launch_state == "featured" or featured_at:
                        product.is_featured = True

            # Recursive scan for media lists in entire state (handles normalized cache splits)
            media_lists = find_key_recursively(merged_rehydrate, "media")
            for ml in media_lists:
                if isinstance(ml, list) and ml:
                    # Validate it's a media list containing Media typename
                    if all(isinstance(x, dict) and x.get("__typename") == "Media" for x in ml):
                        product.media_count = len(ml)
                        break

            # Recursive scan for posts to check if any launch was featured
            post_nodes = find_key_recursively(merged_rehydrate, "featuredAt")
            if any(x is not None for x in post_nodes):
                product.is_featured = True
                        
            # If JSON-LD maker list was empty, try recursive search in Apollo state
            if not extracted_makers:
                makers_list = find_key_recursively(merged_rehydrate, "makers")
                for makers in makers_list:
                    if isinstance(makers, list) and makers:
                        names = []
                        urls = []
                        for maker in makers:
                            if isinstance(maker, dict) and maker.get("name"):
                                names.append(maker.get("name"))
                                if maker.get("username"):
                                    urls.append(f"https://www.producthunt.com/@{maker.get('username')}")
                        if names:
                            product.maker_names = names
                            product.maker_profile_urls = urls
                            extracted_makers = True
                            break
    except Exception as e:
        logger.warning(f"Error parsing Apollo state: {e}", exc_info=True)

    # ====================================================
    # 3. DOM SELECTOR FALLBACKS (If script parsing missed fields)
    # ====================================================
    
    # Fallback Description
    if not product.description:
        meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta_desc and meta_desc.get("content"):
            product.description = meta_desc.get("content")

    # Fallback Website URL
    if not product.website_url:
        visit_link = soup.find("a", href=True, string=re.compile("Visit website", re.I))
        if visit_link:
            product.website_url = visit_link["href"]

    # Fallback Media Count from DOM
    if product.media_count == 0:
        slide_imgs = soup.find_all("img", alt=re.compile("Screenshot", re.I))
        if slide_imgs:
            product.media_count = len(slide_imgs)
        else:
            dots = soup.find_all("button", attrs={"aria-label": re.compile(r"slide|dot", re.I)})
            if dots:
                product.media_count = len(dots)

    return product
