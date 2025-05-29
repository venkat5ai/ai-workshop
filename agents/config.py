import os

# --- Application Configuration ---
# Use environment variables with fallbacks to defaults for flexibility

# Flask App Configuration
FLASK_PORT = int(os.getenv("FLASK_PORT", 3010))

# Document and ChromaDB Configuration
# This is the directory INSIDE the container where documents will be accessed.
# It should match the target of your Docker volume mount.
DOCUMENT_STORAGE_DIRECTORY = os.getenv("DOCUMENT_STORAGE_DIRECTORY", "/app/data")

# This is the directory INSIDE the container where ChromaDB will be stored.
# This should ALSO be volume-mounted if you want the ChromaDB to persist on the host.
CHROMA_DB_DIRECTORY = os.getenv("CHROMA_DB_DIRECTORY", "/app/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents_collection")

# --- API Tool Configuration ---
# Store API base URLs and API keys here for easy management

# For OpenWeatherMap:
# API Key should be set as an environment variable (OPENWEATHER_API_KEY)
OPENWEATHER_BASE_URL = os.getenv("OPENWEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5/weather")

# Google Gemini LLM Configuration
# GOOGLE_API_KEY is expected to be set as an environment variable directly
GEMINI_MODEL_TO_USE = os.getenv("GEMINI_MODEL_TO_USE", "models/gemini-2.5-flash-preview-05-20")

# Ollama LLM Configuration
OLLAMA_MODEL_TO_USE = os.getenv("OLLAMA_MODEL_TO_USE", "gemma3:latest")

# --- Google Search Configuration ---
# GOOGLE_SEARCH_API_KEY is expected to be set as an environment variable (e.g., GOOGLE_SEARCH_API_KEY)
# GOOGLE_CSE_ID is expected to be set as an environment variable (e.g., GOOGLE_CSE_ID)
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY") # No default, must be provided via env var
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID") # No default, must be provided via env var
GOOGLE_SEARCH_TOP_K = int(os.getenv("GOOGLE_SEARCH_TOP_K", 5)) # Max number of top results from GOOGLE SEARCH

# RAG Configuration
RETRIEVER_SEARCH_K = int(os.getenv("RETRIEVER_SEARCH_K", 3)) # Number of documents to retrieve initially from vector store

# --- NEW: Reranker Configuration ---
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2") # A good general-purpose reranker
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", 3)) # Number of top documents to keep AFTER reranking (this is your final context size)
# Note: RETRIEVER_SEARCH_K should typically be higher than RERANKER_TOP_K,
# as you retrieve more initially and then filter down.

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # INFO, DEBUG, WARNING, ERROR, CRITICAL

# Web Scraper User Agent (Optional: can be configured via environment variable)
WEB_SCRAPER_USER_AGENT = os.getenv("WEB_SCRAPER_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Deep Scraping Configuration
SCRAPE_DELAY_SECONDS = float(os.getenv("SCRAPE_DELAY_SECONDS", 1.0)) # Delay between requests during deep scraping
MAX_DOCUMENTS_TO_SCRAPE = int(os.getenv("MAX_DOCUMENTS_TO_SCRAPE", 5)) # Max number of web documents to scrape

# GitHub Configuration
GITHUB_PAT = os.getenv("GITHUB_PAT")