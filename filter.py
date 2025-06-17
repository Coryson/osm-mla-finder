import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pandas as pd
from urllib.parse import urlparse
from typing import List
import logging

# Set up logging
logger = logging.getLogger("filter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Keywords for MLA relevance werden nun aus externer Datei geladen
def load_mla_keywords(filepath: str) -> list:
    with open(filepath, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

MLA_KEYWORDS = load_mla_keywords(os.path.join(os.path.dirname(__file__), 'mla_keywords.txt'))

# Helper: check if a website is reachable and HTML
def is_website_reachable(url: str) -> bool:
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and 'text/html' in resp.headers.get('Content-Type', ''):
            return True
    except Exception as e:
        logger.debug(f"Website check failed for {url}: {e}")
    return False

# Helper: check for MLA keywords in website content
def is_mla_relevant(url: str, keywords: List[str]) -> bool:
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and 'text/html' in resp.headers.get('Content-Type', ''):
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True).lower()
            return any(kw in text for kw in keywords)
    except Exception as e:
        logger.debug(f"MLA keyword check failed for {url}: {e}")
    return False

# Helper: use SerpAPI to find a website
def serpapi_find_website(name: str, city: str, serpapi_key: str) -> str:
    import requests
    params = {
        "engine": "google",
        "q": f"{name} {city} labor medizin website",
        "api_key": serpapi_key,
        "num": 3
    }
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        data = resp.json()
        for result in data.get("organic_results", []):
            url = result.get("link", "")
            if url and is_website_reachable(url):
                return url
    except Exception as e:
        logger.warning(f"SerpAPI search failed for {name}, {city}: {e}")
    return ""

# Main enrichment pipeline
def enrich_and_filter(input_csv: str, output_csv: str, serpapi_key: str):
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    is_reachable_list = []
    is_mla_relevant_list = []
    website_list = []

    for idx, row in df.iterrows():
        website = str(row.get('website', '')).strip()
        name = str(row.get('name', '')).strip()
        city = str(row.get('city', '')).strip()
        valid_url = website if website and is_website_reachable(website) else ""
        # If no valid website, try SerpAPI
        if not valid_url:
            valid_url = serpapi_find_website(name, city, serpapi_key)
        is_reachable = bool(valid_url)
        is_relevant = is_mla_relevant(valid_url, MLA_KEYWORDS) if is_reachable else False
        is_reachable_list.append(is_reachable)
        is_mla_relevant_list.append(is_relevant)
        website_list.append(valid_url)
        logger.info(f"{name} ({city}): reachable={is_reachable}, mla_relevant={is_relevant}, url={valid_url}")

    df['website'] = website_list
    df['is_reachable'] = is_reachable_list
    df['is_mla_relevant'] = is_mla_relevant_list

    # Filter: only keep rows where both are True
    df_final = df[(df['is_reachable']) & (df['is_mla_relevant'])].copy()
    df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"Saved verified MLA facilities to {output_csv}: {len(df_final)} rows")

if __name__ == "__main__":
    load_dotenv()
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
    if not SERPAPI_KEY:
        logger.error("SERPAPI_KEY is required in .env file or environment.")
        exit(1)
    import argparse
    parser = argparse.ArgumentParser(description="Enrich and verify MLA facilities with website and relevance checks.")
    parser.add_argument('--input', '-i', required=True, help='Input CSV file (from OSM parsing)')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file (verified)')
    args = parser.parse_args()
    enrich_and_filter(args.input, args.output, SERPAPI_KEY)
