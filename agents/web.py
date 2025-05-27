# web.py
import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool
import logging
from playwright.sync_api import sync_playwright, Playwright
import os

logger = logging.getLogger(__name__)

# --- Static Web Scraper Tool ---
def _static_web_scraper(url: str) -> str:
    """
    Scrapes static HTML content from a given URL using requests and BeautifulSoup4.
    Returns the cleaned text content of the page.
    """
    try:
        # Basic headers to mimic a browser and avoid immediate blocking
        headers = {
            'User-Agent': os.getenv("WEB_SCRAPER_USER_AGENT", 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/', # Mimic a referrer
        }
        logger.info(f"Attempting static scrape for URL: {url}")
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, and nav tags to get cleaner text for LLM processing
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.extract()

        text_content = soup.get_text(separator=' ', strip=True)
        logger.info(f"Successfully scraped static content from {url}, length: {len(text_content)} chars.")
        # Return a manageable chunk for the LLM. You might need to adjust this based on LLM context window.
        return text_content[:10000] # Limit to first 10,000 characters to avoid overwhelming LLM

    except requests.exceptions.Timeout:
        logger.error(f"Static scrape timed out for {url}.")
        return f"Failed to retrieve content from {url}: Request timed out."
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch static content from {url}: {e}")
        return f"Failed to retrieve content from {url}: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during static scraping of {url}: {e}")
        return f"An error occurred while processing content from {url}: {e}"

# Define the static web scraper tool
static_web_scraper_tool = Tool(
    name="StaticWebScraper",
    func=_static_web_scraper,
    description="Useful for getting the full text content of a static webpage from a URL. Input should be a URL (string). Use this for simple blogs, news articles, or documentation that does not load content dynamically with JavaScript. Returns the main textual content of the page."
)

# --- Dynamic Web Browser Tool ---
# Global playwright instance (managed via context manager)
# Note: For production, consider managing Playwright browser instances more robustly (e.g., a pool)
# rather than launching/closing per request for performance, but this is simpler for initial setup.
_playwright_instance: Playwright = None

def _get_playwright_instance():
    global _playwright_instance
    if _playwright_instance is None:
        _playwright_instance = sync_playwright().start()
        logger.info("Initialized Playwright instance.")
    return _playwright_instance

def _close_playwright_instance():
    global _playwright_instance
    if _playwright_instance:
        _playwright_instance.stop()
        _playwright_instance = None
        logger.info("Closed Playwright instance.")

def _dynamic_web_browser(url: str, task_description: str = "") -> str:
    """
    Navigates a dynamic webpage using Playwright and extracts content based on a task description.
    This tool can handle JavaScript-rendered content, clicks, and waits for elements.
    Returns the cleaned text content after performing the task.
    
    Args:
        url (str): The URL of the dynamic webpage.
        task_description (str): A clear description of the task to perform
                                (e.g., "click the 'Load More' button", "wait for product prices to load",
                                "get the text from the element with ID 'main-content'",
                                "find the price of the first item after scrolling down").
                                The tool will try to interpret this to interact with the page.
    Returns:
        str: The extracted content or an error message.
    """
    browser = None
    try:
        pw = _get_playwright_instance()
        browser = pw.chromium.launch() # Launch a new browser instance
        page = browser.new_page()

        logger.info(f"Attempting dynamic scrape for URL: {url} with task: '{task_description}'")
        page.goto(url, wait_until="domcontentloaded", timeout=30000) # Increased timeout

        # Basic interaction based on task_description (can be greatly expanded)
        # LLM will need to guide this more precisely in advanced scenarios
        lower_task = task_description.lower()
        if "scroll down" in lower_task:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000) # Wait for content to load after scroll
        elif "click" in lower_task:
            # Example: "click the 'Load More' button" -> try to find a button with that text
            if "load more" in lower_task:
                try:
                    page.click("button:has-text('Load More')", timeout=5000)
                    page.wait_for_load_state("networkidle") # Wait for network to settle after click
                except Exception as e:
                    logger.warning(f"Could not click 'Load More' button: {e}")
        elif "wait for" in lower_task:
            # Attempt to find a selector mentioned in the description (e.g., "wait for #product-list")
            if "selector" in lower_task:
                try:
                    # Very basic parsing to get a selector from a phrase like "wait for selector #my-element"
                    parts = lower_task.split("selector")
                    if len(parts) > 1:
                        potential_selector = parts[1].strip().split(' ')[0]
                        if potential_selector:
                            page.wait_for_selector(potential_selector, timeout=10000)
                            logger.info(f"Waited for selector: {potential_selector}")
                except Exception as wait_e:
                    logger.warning(f"Failed to wait for specified selector: {wait_e}. Proceeding.")
            else: # Default wait for common dynamic content scenarios
                 page.wait_for_load_state("networkidle") # Wait for network activity to cease
                 page.wait_for_timeout(2000) # Small additional wait

        # After interactions, get the page content
        html_content = page.content()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.extract()
        text_content = soup.get_text(separator=' ', strip=True)

        logger.info(f"Successfully scraped dynamic content from {url}, length: {len(text_content)} chars.")
        return text_content[:10000] # Limit to first 10,000 characters

    except Exception as e:
        logger.error(f"Failed to perform dynamic scraping on {url} for task '{task_description}': {e}")
        return f"Failed to retrieve dynamic content from {url}. Error: {e}. Please ensure the URL is correct and the task description is clear. This might require manual inspection or more specific instructions."
    finally:
        if browser:
            browser.close() # Ensure browser is closed

# Define the dynamic web browser tool
dynamic_web_browser_tool = Tool(
    name="DynamicWebBrowser",
    func=lambda args: _dynamic_web_browser(args["url"], args.get("task_description", "")),
    description="Advanced tool for interacting with dynamic or interactive webpages that load content with JavaScript. Use this when `StaticWebScraper` fails or when you need to perform actions like clicking buttons, logging in, navigating multiple pages, or waiting for specific content to appear. Input must be a JSON dictionary with 'url' (string) and an optional 'task_description' (string) detailing the actions to take (e.g., 'scroll down', 'click load more button', 'wait for prices to load'). Returns the main textual content after performing the task."
)