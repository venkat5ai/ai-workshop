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

# Google Gemini LLM Configuration
# GOOGLE_API_KEY is expected to be set as an environment variable directly
GEMINI_MODEL_TO_USE = os.getenv("GEMINI_MODEL_TO_USE", "models/gemini-2.5-flash-preview-05-20")

# Ollama LLM Configuration
OLLAMA_MODEL_TO_USE = os.getenv("OLLAMA_MODEL_TO_USE", "gemma3:latest")

# RAG Configuration
RETRIEVER_SEARCH_K = int(os.getenv("RETRIEVER_SEARCH_K", 3)) # Number of documents to retrieve initially from vector store

# --- Reranker Configuration ---
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2") # A good general-purpose reranker
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", 3)) # Number of top documents to keep AFTER reranking (this is your final context size)
# Note: RETRIEVER_SEARCH_K should typically be higher than RERANKER_TOP_K,
# as you retrieve more initially and then filter down.

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # INFO, DEBUG, WARNING, ERROR, CRITICAL