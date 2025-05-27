# ingestion_utils.py
import requests
from bs4 import BeautifulSoup
import logging
import os
import time
from urllib.parse import urljoin, urlparse
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# These global variables will be set by agent.py from config.py
WEB_SCRAPER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
SCRAPE_DELAY_SECONDS = 1.0
_MAX_DOCUMENTS_TO_SCRAPE = 200 # Default fallback, will be updated by set_ingestion_config


def set_ingestion_config(user_agent: str, scrape_delay: float, max_docs_to_scrape: int):
    """Sets configuration parameters for ingestion functions."""
    global WEB_SCRAPER_USER_AGENT, SCRAPE_DELAY_SECONDS, _MAX_DOCUMENTS_TO_SCRAPE
    WEB_SCRAPER_USER_AGENT = user_agent
    SCRAPE_DELAY_SECONDS = scrape_delay
    _MAX_DOCUMENTS_TO_SCRAPE = max_docs_to_scrape
    logger.info(f"Ingestion config set: User-Agent='{WEB_SCRAPER_USER_AGENT}', Scrape Delay={SCRAPE_DELAY_SECONDS}s, Max Docs={_MAX_DOCUMENTS_TO_SCRAPE}")


def load_web_documents_deep_scrape(config_dir: str) -> list[Document]:
    """
    Loads text content from URLs listed in a web.conf file
    and performs deep scraping (follows internal links on the same domain)
    to convert content into LangChain Document objects.
    """
    web_documents = []
    websites_conf_path = os.path.join(config_dir, "web.conf")
    
    if not os.path.exists(websites_conf_path):
        logger.warning(f"web.conf not found at {websites_conf_path}. No web documents will be loaded.")
        return []

    logger.info(f"Initiating deep scraping from URLs listed in {websites_conf_path}...")
    
    try:
        with open(websites_conf_path, 'r') as f:
            start_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Use a set to keep track of visited URLs to avoid infinite loops and re-scraping
        visited_urls = set()
        # Use a list as a queue for URLs to visit
        urls_to_visit = list(start_urls)
        
        # Headers to mimic a browser for web requests
        headers = {
            'User-Agent': WEB_SCRAPER_USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
        }

        while urls_to_visit and len(web_documents) < _MAX_DOCUMENTS_TO_SCRAPE: 
            current_url = urls_to_visit.pop(0) # Get the next URL from the queue

            # Normalize URL to prevent duplicates from slight variations (e.g., trailing slashes)
            parsed_current_url = urlparse(current_url)
            normalized_current_url = parsed_current_url.scheme + "://" + parsed_current_url.netloc + parsed_current_url.path
            
            if normalized_current_url in visited_urls:
                logger.debug(f"Skipping already visited URL: {normalized_current_url}")
                continue
            
            visited_urls.add(normalized_current_url)
            
            # Extract base domain for scope control
            base_domain = parsed_current_url.netloc

            try:
                logger.info(f"Deep scraping URL: {current_url}")
                response = requests.get(current_url, timeout=20, headers=headers)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract links for deep scraping BEFORE modifying soup for text extraction
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    parsed_full_url = urlparse(full_url)
                    
                    # Ensure it's an HTTP/HTTPS link and within the same domain
                    # Also check if it's already visited or in queue to avoid adding duplicates
                    if parsed_full_url.scheme in ['http', 'https'] and \
                       parsed_full_url.netloc == base_domain and \
                       full_url not in visited_urls and \
                       full_url not in urls_to_visit: 
                        urls_to_visit.append(full_url)
                        logger.debug(f"Found new link: {full_url}")

                # Now, clean the HTML for text extraction for the current page
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'form', 'aside', 'img', 'noscript']):
                    tag.extract()
                
                text_content = soup.get_text(separator=' ', strip=True)
                
                if text_content:
                    # Create a LangChain Document from the scraped text
                    web_documents.append(Document(page_content=text_content, metadata={"source": current_url, "scraped_from_domain": base_domain}))
                    logger.info(f"Successfully scraped {current_url}. Text length: {len(text_content)} chars. Total scraped docs: {len(web_documents)}")
                else:
                    logger.warning(f"No meaningful text extracted from {current_url}.")

            except requests.exceptions.Timeout:
                logger.error(f"Deep scrape timed out for {current_url}.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching {current_url}: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing {current_url}: {e}")
            
            # Introduce a delay to be polite and avoid being blocked
            time.sleep(SCRAPE_DELAY_SECONDS) 

    except Exception as e:
        logger.error(f"Error reading web.conf or during deep scraping: {e}")
    
    logger.info(f"Finished deep scraping. Loaded {len(web_documents)} web documents.")
    return web_documents