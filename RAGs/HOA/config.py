import os

# --- Application Configuration ---
# Use environment variables with fallbacks to defaults for flexibility

# Flask App Configuration
FLASK_PORT = int(os.getenv("FLASK_PORT", 3010))

# Document and ChromaDB Configuration
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./data")
CHROMA_DB_DIRECTORY = os.getenv("CHROMA_DB_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hoa_documents_collection")

# Google Gemini LLM Configuration
# GOOGLE_API_KEY is expected to be set as an environment variable directly
# For consistency, you could also define it here as os.getenv("GOOGLE_API_KEY")
# but LangChain automatically picks it up from the environment.
GEMINI_MODEL_TO_USE = os.getenv("GEMINI_MODEL_TO_USE", "models/gemini-2.5-flash-preview-05-20")

# RAG Configuration
RETRIEVER_SEARCH_K = int(os.getenv("RETRIEVER_SEARCH_K", 3))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # INFO, DEBUG, WARNING, ERROR, CRITICAL