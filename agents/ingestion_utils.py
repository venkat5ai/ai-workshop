import os
import requests
import json
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from datetime import datetime

# LangChain specific imports for document loading and processing
from langchain_community.document_loaders import WebBaseLoader # For basic web loading
from langchain_community.document_loaders import UnstructuredURLLoader # For more robust URL loading
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Initialize logger
logger = logging.getLogger(__name__)

# Global configuration for ingestion, set by agent.py
WEB_SCRAPER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SCRAPE_DELAY_SECONDS = 1.0 # Default delay between requests
MAX_DOCUMENTS_TO_SCRAPE = 5 # Default max documents for deep scraping

def set_ingestion_config(user_agent: str, scrape_delay: float, max_docs: int):
    """
    Sets the global configuration variables for ingestion.
    Called from agent.py during initialization.
    """
    global WEB_SCRAPER_USER_AGENT, SCRAPE_DELAY_SECONDS, MAX_DOCUMENTS_TO_SCRAPE
    WEB_SCRAPER_USER_AGENT = user_agent
    SCRAPE_DELAY_SECONDS = scrape_delay
    MAX_DOCUMENTS_TO_SCRAPE = max_docs
    logger.info(f"Ingestion config set: User-Agent='{WEB_SCRAPER_USER_AGENT}', Scrape Delay={SCRAPE_DELAY_SECONDS}s, Max Docs to Scrape={MAX_DOCUMENTS_TO_SCRAPE}")


def _fetch_url_content(url):
    """Fetches content from a URL with custom headers and error handling."""
    headers = {
        'User-Agent': WEB_SCRAPER_USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    try:
        logger.debug(f"Attempting to fetch content from: {url}")
        response = requests.get(url, headers=headers, timeout=10) # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        logger.debug(f"Successfully fetched content from: {url}")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch content from {url}: {e}")
        return None


def _extract_links(html_content, base_url):
    """Extracts unique, valid internal links from HTML content."""
    if not html_content:
        return set()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        full_url = urljoin(base_url, href)
        parsed_base = urlparse(base_url)
        parsed_full = urlparse(full_url)

        # Only add links that are:
        # 1. Valid URLs
        # 2. On the same domain as the base_url
        # 3. Not mailto or phone links
        # 4. Not fragment identifiers for the same page
        if parsed_full.scheme in ['http', 'https'] and \
           parsed_full.netloc == parsed_base.netloc and \
           not full_url.startswith('mailto:') and \
           not full_url.startswith('tel:') and \
           (parsed_full.path != parsed_base.path or parsed_full.fragment == ''): # Avoid self-page fragments
            links.add(full_url)
    return links


def load_web_documents_deep_scrape(base_document_directory: str):
    """
    Performs a deep scrape of URLs listed in web.conf, extracts text content,
    and returns them as LangChain Document objects.
    Attempts to discover and scrape child links up to MAX_DOCUMENTS_TO_SCRAPE.
    """
    web_conf_path = os.path.join(base_document_directory, 'web.conf')
    if not os.path.exists(web_conf_path):
        logger.info(f"web.conf not found at {web_conf_path}. Skipping web document ingestion.")
        return []

    with open(web_conf_path, 'r') as f:
        urls_to_scrape = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    if not urls_to_scrape:
        logger.info("web.conf found but is empty. No URLs to scrape.")
        return []

    scraped_documents = []
    visited_urls = set()
    urls_to_visit = list(urls_to_scrape) # Use a list to maintain order for BFS-like behavior

    logger.info(f"Starting deep web scraping for {len(urls_to_scrape)} initial URLs from web.conf...")

    while urls_to_visit and len(scraped_documents) < MAX_DOCUMENTS_TO_SCRAPE:
        current_url = urls_to_visit.pop(0)

        if current_url in visited_urls:
            continue

        logger.info(f"Scraping: {current_url} ({len(scraped_documents)}/{MAX_DOCUMENTS_TO_SCRAPE} documents scraped)")
        
        # Add to visited set immediately to prevent re-queuing
        visited_urls.add(current_url) 

        html_content = _fetch_url_content(current_url)
        if html_content:
            try:
                # Use UnstructuredURLLoader for robust HTML parsing
                # It handles extracting cleaner text than just BeautifulSoup often
                loader = UnstructuredURLLoader(urls=[current_url])
                # Note: UnstructuredURLLoader.load() returns a list of documents for each URL
                docs = loader.load()
                
                if docs:
                    for doc in docs:
                        # Add metadata for source URL and timestamp
                        doc.metadata["source"] = current_url
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                        scraped_documents.append(doc)
                    logger.debug(f"Added document from {current_url}. Total scraped: {len(scraped_documents)}")
                else:
                    logger.warning(f"No documents extracted by UnstructuredURLLoader from {current_url}.")

                # Extract new links for deeper scraping (if not at max documents)
                if len(scraped_documents) < MAX_DOCUMENTS_TO_SCRAPE:
                    new_links = _extract_links(html_content, current_url)
                    for link in new_links:
                        if link not in visited_urls and len(urls_to_visit) + len(scraped_documents) < MAX_DOCUMENTS_TO_SCRAPE * 2: # Heuristic to prevent massive queue bloat
                            urls_to_visit.append(link)
            except Exception as e:
                logger.error(f"Error processing URL {current_url} with UnstructuredURLLoader: {e}")
        
        # Implement a delay to be polite to websites and avoid IP blocking
        time.sleep(SCRAPE_DELAY_SECONDS)

    logger.info(f"Finished web scraping. Total documents scraped: {len(scraped_documents)}.")
    return scraped_documents

# Example of how this might be used (for testing ingestion_utils in isolation)
if __name__ == '__main__':
    # This block is for local testing of this utility script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a dummy web.conf for testing
    test_dir = "test_data_for_ingestion"
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "web.conf"), "w") as f:
        f.write("https://www.google.com/\n") # Example URL
        f.write("https://www.bbc.com/news\n") # Another example

    set_ingestion_config(
        user_agent="MyCustomScraper/1.0",
        scrape_delay=0.5,
        max_docs=3
    )

    print(f"\n--- Running ingestion_utils.py standalone test ---")
    scraped_docs = load_web_documents_deep_scrape(test_dir)

    print(f"\nScraped {len(scraped_docs)} documents.")
    for i, doc in enumerate(scraped_docs):
        print(f"--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Content (first 200 chars): {doc.page_content[:200]}...")
        print("-" * 20)
    
    # Clean up test directory
    os.remove(os.path.join(test_dir, "web.conf"))
    os.rmdir(test_dir)