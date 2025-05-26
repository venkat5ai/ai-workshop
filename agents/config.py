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
CHROMA_DB_DIRECTORY = os.getenv("CHROMA_DB_DIRECTORY", "/app/chroma_db") # Set to absolute path inside container for consistency
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents_collection")

# Google Gemini LLM Configuration
# GOOGLE_API_KEY is expected to be set as an environment variable directly
GEMINI_MODEL_TO_USE = os.getenv("GEMINI_MODEL_TO_USE", "models/gemini-2.5-flash-preview-05-20")

# Ollama LLM Configuration
OLLAMA_MODEL_TO_USE = os.getenv("OLLAMA_MODEL_TO_USE", "gemma3:latest")

# RAG Configuration
RETRIEVER_SEARCH_K = int(os.getenv("RETRIEVER_SEARCH_K", 3))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # INFO, DEBUG, WARNING, ERROR, CRITICAL