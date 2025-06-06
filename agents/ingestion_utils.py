# ingestion_utils.py

import os
import logging
# Removed json, requests, bs4, urllib.parse, time, datetime as web scraping is removed.

# LangChain specific imports for document loading and processing
# from langchain_community.document_loaders import WebBaseLoader # Removed
# from langchain_community.document_loaders import UnstructuredURLLoader # Removed
# from langchain.text_splitter import RecursiveCharacterTextSplitter # Used by agent directly
# from langchain_core.documents import Document # Used by agent directly

# Initialize logger
logger = logging.getLogger(__name__)

# Global configuration for ingestion (reduced to only what's still relevant if any)
# Removed WEB_SCRAPER_USER_AGENT, SCRAPE_DELAY_SECONDS, MAX_DOCUMENTS_TO_SCRAPE as web scraping is removed.

def set_ingestion_config(user_agent: str = None, scrape_delay: float = None, max_docs: int = None):
    """
    Sets the global configuration variables for ingestion.
    Now primarily a placeholder as web scraping config is removed.
    """
    logger.info("Ingestion config set (web scraping specific settings removed).")

# Removed _fetch_url_content, _extract_links as web scraping is removed.

def load_web_documents_deep_scrape(base_document_directory: str):
    """
    This function is now a stub, as web scraping has been removed.
    It will always return an empty list.
    """
    logger.info("Web scraping functionality has been removed. Returning no web documents.")
    return []

# Removed example usage block (if __name__ == '__main__':)