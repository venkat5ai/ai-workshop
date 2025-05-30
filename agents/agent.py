# First, ensure you have the necessary libraries installed.
# You can run 'pip install -r requirements.txt' where requirements.txt is provided.

"""
This script implements a Retrieval-Augmented Generation (RAG) assistant
using Flask for the API, LangChain for orchestration, and various
libraries for document processing and LLM interaction. It supports
both local (Ollama) and cloud (Google Gemini) LLM mode.
"""

# --- IMPORTANT: Patch sqlite3 for ChromaDB compatibility ---
# This must be the very first lines, before any other imports that might
# indirectly trigger chromadb's import.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules['pysqlite3']
# --- END sqlite3 patch ---


# --- Standard Library Imports ---
import uuid
import os
import logging
import json
import requests
import google.generativeai as genai
from werkzeug.utils import secure_filename # For secure file naming
import time # For exponential backoff in Google Search Tool
from datetime import datetime # For web scraping metadata
import yaml # To load OpenAPI spec files
from functools import partial # For dynamic tool creation

# --- Flask and related imports ---
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Core LangChain and Data Processing Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
# Temporarily alias UnstructuredFileLoader from langchain_community
# The deprecation warning indicates this should move to 'langchain-unstructured' package
from langchain_community.document_loaders import UnstructuredFileLoader as LangChainUnstructuredFileLoader
from langchain_community.document_loaders import WebBaseLoader # For basic web loading in ingestion_utils
from langchain_community.document_loaders import UnstructuredURLLoader # For more robust URL loading in ingestion_utils

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Imports for LangChain Agents and Tools
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_community.utilities.requests import RequestsWrapper # For making HTTP requests


# --- NEW: Imports for Google Search Tool ---
from langchain_google_community import GoogleSearchAPIWrapper

# --- NEW: Imports for Reranking ---
from sentence_transformers import CrossEncoder # For reranking
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

# --- NEW: Import ingestion utilities and openapi tools ---
import ingestion_utils 
import openapi

# --- Application Configuration ---
import config

# --- Logging Setup ---
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables for the Flask App --
llm = None
retriever = None
agent_executors = {}
db = None # Make db a global variable so it can be accessed for updates
all_agent_tools = [] # NEW: Global list to hold all tools (RAG, Weather, Google, OpenAPI)

# --- Flask App Instance ---
app = Flask(__name__)
CORS(app)

# --- Define allowed extensions for uploads ---
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'csv', 'xlsx', 'pptx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Tool Definitions (Global) ---

# --- Google Search Tool Definition ---
# Initialize Google Search API Wrapper
# It picks up GOOGLE_API_KEY and GOOGLE_CSE_ID from environment variables
try:
    Google_Search_wrapper = GoogleSearchAPIWrapper()
    Google_Search_tool = Tool(
        name="Google_Search",
        func=lambda query: run_Google_Search(query), # Use a lambda to call the function
        description="A powerful search tool for general knowledge, current events, and information not found in the documents. Input should be a concise search query string."
    )
    logger.info("GoogleSearchAPIWrapper and Google_Search_tool initialized.")
except Exception as e:
    logger.error(f"Failed to initialize GoogleSearchAPIWrapper: {e}")
    logger.error("Please ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are set in your environment for Google Search functionality.")
    Google_Search_wrapper = None
    Google_Search_tool = None # Set to None to prevent errors if not initialized

def run_Google_Search(query: str) -> str:
    """
    Performs a Google search for the given query and returns a summary of the results.
    Includes basic error handling and exponential backoff for rate limits.
    """
    if Google_Search_wrapper is None:
        logger.error("Google Search tool is not initialized. Cannot perform search.")
        return "Google Search tool is not available. Please check server configuration."

    max_retries = 3
    for attempt in range(max_retries):
        try:
            results = Google_Search_wrapper.results(query, config.GOOGLE_SEARCH_TOP_K)
            
            if not results:
                return "No relevant results found for the query."
            
            formatted_results = []
            for i, res in enumerate(results):
                formatted_results.append(
                    f"Result {i+1}: "
                    f"Title: {res.get('title', 'N/A')}\n"
                    f"Snippet: {res.get('snippet', 'N/A')}\n"
                    f"Link: {res.get('link', 'N/A')}"
                )
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "Quota exceeded" in error_message:
                logger.warning(f"Google Search API quota exceeded or rate limited (attempt {attempt+1}/{max_retries}). Retrying with backoff...")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Error during Google Search (attempt {attempt+1}/{max_retries}) for query '{query}': {e}")
                if attempt == max_retries - 1:
                    return f"Failed to perform Google search due to an error: {e}. Please try again later."
    return "Failed to perform Google search after multiple retries."


# --- Weather Tool Definition ---
def get_current_weather(location: str) -> str:
    """
    Fetches current weather data for a given location (zip code, city, city,state, or city,country)
    using OpenWeatherMap.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logger.error("OPENWEATHER_API_KEY environment variable is not set for weather tool.")
        return "Weather API key not configured."

    base_url = config.OPENWEATHER_BASE_URL
    params = {
        "appid": api_key,
        "units": "imperial"
    }

    location_attempts = []
    original_parts = [p.strip() for p in location.split(',')]

    if location:
        if location.isdigit() and len(location) == 5:
            location_attempts.append({"zip": f"{location},us"})
        else:
            location_attempts.append({"q": location})
    
    if len(original_parts) == 2:
        city, state_or_country = original_parts
        location_attempts.append({"q": f"{city},{state_or_country}"})
        location_attempts.append({"q": city})
    
    if len(original_parts) == 1 and not (original_parts[0].isdigit() and len(original_parts[0]) == 5):
        if {"q": original_parts[0]} not in location_attempts:
            location_attempts.append({"q": original_parts[0]})

    final_attempts = []
    seen = set()
    for d in location_attempts:
        t = tuple(sorted(d.items()))
        if t not in seen:
            final_attempts.append(d)
            seen.add(t)
    
    for attempt_params in final_attempts:
        current_params = {**params, **attempt_params}
        query_identifier = attempt_params.get("q") or attempt_params.get("zip")
        logger.info(f"Attempting weather API call for: {query_identifier} (Original: {location})")

        try:
            response = requests.get(base_url, params=current_params)
            response.raise_for_status() 
            weather_data = response.json()
            
            if weather_data.get("main") and weather_data.get("weather"):
                main_info = weather_data["main"]
                weather_desc = weather_data["weather"][0]["description"]
                city_name = weather_data.get("name", query_identifier.split(',')[0])
                
                summary = (
                    f"Current weather in {city_name}: "
                    f"{main_info.get('temp')}°F, feels like {main_info.get('feels_like')}°F. "
                    f"Conditions: {weather_desc}. Humidity: {main_info.get('humidity')}%. "
                    f"Wind speed: {weather_data.get('wind', {}).get('speed')} mph."
                )
                logger.info(f"Successfully retrieved weather for '{query_identifier}'.")
                return summary
            else:
                logger.warning(f"Incomplete weather data for '{query_identifier}'. Trying next option if available.")
                continue

        except requests.exceptions.RequestException as e:
            logger.warning(f"OpenWeatherMap API call failed for '{query_identifier}': {e}. Trying next option if available.")
            continue
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decoding failed for '{query_identifier}' from OpenWeatherMap: {e}. Trying next next option if available.")
            continue
        except Exception as e:
            logger.error(f"Unexpected error in get_current_weather tool for '{query_identifier}': {e}. Trying next option if available.")
            continue

    return (
        f"I was unable to retrieve weather data for '{location}' using the weather tool. "
        f"This could be due to an invalid or ambiguous location, or a temporary issue with the weather service. "
        f"Please try again with a more precise location, such as a zip code (e.g., '90210') or a city and state/country (e.g., 'Ashburn, VA' or 'Paris, FR')."
    )

# --- CrossEncoderCompressor Class ---
class CrossEncoderCompressor(BaseDocumentCompressor):
    _model: CrossEncoder = None
    top_n: int = 3

    def __init__(self, model: CrossEncoder, top_n: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self.top_n = top_n
        logger.info(f"CrossEncoderCompressor initialized with top_n={self.top_n}")

    @property
    def model(self) -> CrossEncoder:
        return self._model

    def compress_documents(
        self, documents: list[Document], query: str, callbacks=None
    ) -> list[Document]:
        if not documents:
            return []
        
        passages = [doc.page_content for doc in documents]
        sentence_pairs = [[query, passage] for passage in passages]
        
        if self.model is None:
            logger.error("CrossEncoder model is not initialized in compressor. Skipping reranking.")
            return documents
        
        scores = self.model.predict(sentence_pairs)
        
        scored_documents = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )
        
        compressed_documents = [doc for score, doc in scored_documents[:self.top_n]]
        logger.debug(f"Reranked {len(documents)} documents to top {len(compressed_documents)} for query: '{query}'")
        return compressed_documents


# --- Function to check available Gemini models (Optional, useful for initial setup) ---
def check_gemini_models():
    """
    Checks and logs available Gemini models if GOOGLE_API_KEY is set.
    It specifically checks for the configured model and logs available ones if not found.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set. Please set it to proceed for Gemini model features.")
        return []

    genai.configure(api_key=api_key)
    
    logger.info("--- Checking available Gemini models ---")
    available_model_names = []
    try:
        all_listed_models = list(genai.list_models())
        for m in all_listed_models:
            if 'generateContent' in m.supported_generation_methods:
                available_model_names.append(m.name)
        
        configured_model = config.GEMINI_MODEL_TO_USE

        if configured_model in available_model_names:
            logger.info(f"Configured Gemini model '{configured_model}' is available and ready for use.")
        else:
            logger.error(f"Configured Gemini model '{configured_model}' is NOT available for your API key.")
            logger.error("Please ensure the model name is correct, the Gemini API is enabled in your Google Cloud Project,")
            logger.error("and your API key has the necessary permissions.")
            logger.info("--- Full list of available Gemini models supporting 'generateContent' ---")
            if not available_model_names:
                logger.info("No models supporting 'generateContent' were found at all with the provided API key.")
            else:
                for model_name in sorted(available_model_names):
                    logger.info(f"  - {model_name}")
            logger.info("--- End of available models list ---")
        
        logger.info(f"--- End of available models check ---")

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        logger.error("Please double-check your GOOGLE_API_KEY, internet connection, and API enablement if using Gemini.")
    return available_model_names

# --- RAG Tool Definition Helper Function (now defined at module level) ---
# CRITICAL FIX: Move this function definition BEFORE initialize_rag_components
def _run_document_qa_retriever_tool_for_session(query: str, session_memory: ConversationBufferMemory) -> str:
    """
    Runs the RAG chain with the given query and the current session's chat history.
    This function is used as the 'func' for the document_qa_retriever tool.
    It takes the session's memory as an explicit argument.
    """
    # Access the memory object passed for the current session.
    input_dict = {"input": query, "chat_history": session_memory.load_memory_variables({})["chat_history"]}
    
    try:
        # Assuming rag_chain is accessible globally or passed in
        # For simplicity, we'll re-create the basic RAG chain here each time the tool is called
        # This is less efficient but ensures the tool is self-contained.
        # A more advanced approach would make rag_chain a global or cached object.
        history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Standalone question:"),
        ])
        retrieval_chain = create_history_aware_retriever(llm, retriever, history_aware_retriever_prompt)
        
        document_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a diligent and accurate Document Assistant. Provide information clearly and directly.
            Based on the chat history and the provided context documents, answer the user's question. If the question is outside the context, use your general knowledge.
            If you refer to specific information from the documents, you may briefly mention 'based on the documents' or similar, but avoid phrases like 'I cannot answer from the provided context'.

            Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        document_qa_chain = create_stuff_documents_chain(llm, document_qa_prompt)
        rag_chain = create_retrieval_chain(retrieval_chain, document_qa_chain)

        result = rag_chain.invoke(input_dict)
        return result.get('answer', 'No answer found from documents.')
    except Exception as e:
        logger.error(f"Error in document_qa_retriever tool for query '{query}': {e}", exc_info=True)
        return f"Error retrieving document answer: An issue occurred while processing the document context."


# --- Function to initialize the RAG components ---
def initialize_rag_components(model_mode: str):
    """
    Initializes the RAG pipeline components: document loader, text splitter,
    embedding model, vector store (ChromaDB), and language model.
    """
    global llm, retriever, db, all_agent_tools

    # Ensure necessary directories exist.
    for directory in [config.DOCUMENT_STORAGE_DIRECTORY, config.AGENT_CONFIG_DIRECTORY, config.CHROMA_DB_DIRECTORY]:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory '{directory}': {e}")
            logger.error("Application cannot proceed without a writable directory.")
            sys.exit(1)

    all_documents = []

    # --- 1. Load Documents from Various Types (Local files and Web) ---
    logger.info(f"Attempting to load local documents from: {config.DOCUMENT_STORAGE_DIRECTORY}")
    
    if not os.path.isdir(config.DOCUMENT_STORAGE_DIRECTORY):
        logger.error(f"'{config.DOCUMENT_STORAGE_DIRECTORY}' is not a valid directory. Skipping local document loading.")
        local_documents = []
    else:
        try:
            local_loader = DirectoryLoader(
                config.DOCUMENT_STORAGE_DIRECTORY,
                loader_cls=LangChainUnstructuredFileLoader,
                recursive=True,
                show_progress=True,
            )
            local_documents = local_loader.load()
            logger.info(f"Loaded {len(local_documents)} local document(s).")
        except Exception as e:
            logger.exception(f"Detailed error during local document loading from '{config.DOCUMENT_STORAGE_DIRECTORY}': {e}")
            logger.error("Error loading local documents. Application will proceed without them.")
            local_documents = []

    all_documents.extend(local_documents)

    ingestion_utils.set_ingestion_config(
        config.WEB_SCRAPER_USER_AGENT, 
        config.SCRAPE_DELAY_SECONDS, 
        config.MAX_DOCUMENTS_TO_SCRAPE
    )
    web_docs = ingestion_utils.load_web_documents_deep_scrape(config.DOCUMENT_STORAGE_DIRECTORY)
    all_documents.extend(web_docs)
    logger.info(f"Loaded {len(web_docs)} web document(s) via deep scraping. Total documents for RAG: {len(all_documents)}")


    # --- 2. Split Documents into Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    logger.info(f"Split documents into {len(texts)} chunks.")

    # --- 3. Create Embeddings and Store in ChromaDB ---
    logger.info("Creating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(config.CHROMA_DB_DIRECTORY) and os.listdir(config.CHROMA_DB_DIRECTORY):
        logger.info("Loading existing ChromaDB vector store...")
        db = Chroma(persist_directory=config.CHROMA_DB_DIRECTORY, embedding_function=embeddings, collection_name=config.COLLECTION_NAME)
    elif texts:
        logger.info("Creating new ChromaDB vector store from loaded documents...")
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=config.CHROMA_DB_DIRECTORY,
            collection_name=config.COLLECTION_NAME
        )
        logger.info("ChromaDB vector store created and persisted.")
    else:
        logger.info("No existing ChromaDB and no documents to load. Initializing an empty ChromaDB vector store.")
        db = Chroma(embedding_function=embeddings, persist_directory=config.CHROMA_DB_DIRECTORY, collection_name=config.COLLECTION_NAME)
        logger.info("Empty ChromaDB vector store initialized.")


    # --- 4. Set up the Language Model (Conditional based on mode) ---
    if model_mode == "local":
        llm = ChatOllama(model=config.OLLAMA_MODEL_TO_USE)
        logger.info(f"Using Ollama local model: {config.OLLAMA_MODEL_TO_USE}")
    elif model_mode == "cloud":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable is not set. Cannot use Gemini model.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_TO_USE, temperature=0.2) 
        logger.info(f"Using Google Gemini cloud model: {config.GEMINI_MODEL_TO_USE}")
    else:
        logger.error(f"Invalid model mode: {model_mode}. Please use 'local' or 'cloud'.")
        sys.exit(1)

    # --- 5. Create the Retriever with Contextual Compression and Reranking ---
    base_retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVER_SEARCH_K})
    
    logger.info(f"Loading reranker model: {config.RERANKER_MODEL_NAME}...")
    try:
        reranker_model = CrossEncoder(config.RERANKER_MODEL_NAME)
        logger.info(f"CrossEncoder model '{config.RERANKER_MODEL_NAME}' loaded successfully.")
        
        if reranker_model:
            compressor = CrossEncoderCompressor(model=reranker_model, top_n=config.RERANKER_TOP_K)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            )
            logger.info("ContextualCompressionRetriever with Reranker initialized.")
        else:
            logger.warning("Reranker model failed to load (returned None). Proceeding without reranking.")
            retriever = base_retriever 

    except Exception as e:
        logger.error(f"Error loading or initializing reranker model '{config.RERANKER_MODEL_NAME}': {e}", exc_info=True)
        logger.error("Proceeding without reranking. Retrieval quality might be lower.")
        retriever = base_retriever 
        
    logger.info("\nRAG components initialized.")

    # --- 6. Initialize Global Tools for the Agent ---
    logger.info("Initializing global tools for the agent...")
    
    # Tool 1: Weather Tool
    weather_tool = Tool(
        name="get_current_weather",
        func=get_current_weather,
        description="Useful for fetching current weather conditions for a specified location. Input must be a single string representing the location, such as a zip code (e.g., '90210'), a precise city name (e.g., 'London'), or a combination of city and state/country (e.g., 'Ashburn, VA' or 'Paris, FR'). The tool will try its best to resolve the location given."
    )
    all_agent_tools.append(weather_tool)
    logger.info("Added 'get_current_weather' tool.")

    # Tool 2: Google Search Tool
    if Google_Search_tool:
        all_agent_tools.append(Google_Search_tool) 
        logger.info("Added 'Google Search' tool.")
    else:
        logger.warning("Google Search tool was not initialized. Skipping addition to agent tools.")
    
    # Tool 3: Document QA Retriever Tool (its func is defined within api_bot route)
    all_agent_tools.append(Tool(
        name="document_qa_retriever",
        func=lambda q: "This tool needs to be properly initialized with session memory.", # Placeholder func
        description="Useful for answering questions about the uploaded documents and pre-scraped web content. Input should be the user's question, ideally rephrased if it's a follow-up."
    ))
    logger.info("Added 'document_qa_retriever' tool placeholder (will be bound with session memory).")


    # Tool 4: GitHub API Tools (dynamically loaded from OpenAPI spec via openapi.py)
    github_spec_path = os.path.join(config.AGENT_CONFIG_DIRECTORY, config.GITHUB_SPEC_FILENAME) # Use new config directory
    logger.info(f"Attempting to load GitHub API tools from: {github_spec_path}")
    
    github_headers = {}
    if config.GITHUB_TOKEN: # Use GITHUB_TOKEN from config.py
        github_headers["Authorization"] = f"token {config.GITHUB_TOKEN}" # GitHub uses 'token' for PATs
        github_headers["Accept"] = "application/vnd.github.v3+json" # Recommended for GitHub API
        logger.info("GitHub Token found. Will use for authentication with GitHub OpenAPI tools.")
    else:
        logger.warning("GITHUB_TOKEN environment variable not set. GitHub API requests via OpenAPI tools may be rate-limited or fail for private resources.")

    github_requests_wrapper = RequestsWrapper(headers=github_headers)

    # Use the function from the new openapi module
    github_openapi_tools = openapi.load_and_create_tools_from_openapi_spec(github_spec_path, github_requests_wrapper)
    
    if not github_openapi_tools:
        logger.error("No GitHub tools were loaded from OpenAPI spec. Skipping GitHub API integration.")
    else:
        all_agent_tools.extend(github_openapi_tools)
        logger.info(f"GitHub API tools initialized successfully from OpenAPI spec. Added {len(github_openapi_tools)} GitHub operations as tools. Total tools: {len(all_agent_tools)}")


# --- Route for the Web UI (e.g., for local testing and debugging) ---
@app.route('/')
def index():
    """Renders the main index page for the web UI."""
    return render_template('index.html')

# --- API Endpoint for Document Upload ---
@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    """
    Handles document uploads, processes them, and updates the ChromaDB vector store.
    """
    global db, retriever, agent_executors # Need to re-initialize retriever if db changes

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(config.DOCUMENT_STORAGE_DIRECTORY, filename)
            
            os.makedirs(config.DOCUMENT_STORAGE_DIRECTORY, exist_ok=True)
            
            file.save(upload_path)
            logger.info(f"File '{filename}' saved to '{upload_path}'.")

            logger.info(f"Processing uploaded document: '{filename}'...")
            loader = LangChainUnstructuredFileLoader(upload_path)
            new_documents = loader.load()
            
            if not new_documents:
                logger.warning(f"No content extracted from uploaded file: '{filename}'.")
                os.remove(upload_path)
                return jsonify({"status": "warning", "message": f"File '{filename}' uploaded but no text content could be extracted. It was not added to the knowledge base."}), 200

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            new_texts = text_splitter.split_documents(new_documents)
            logger.info(f"Split uploaded document into {len(new_texts)} chunks.")

            if db:
                db.add_documents(new_texts)
                logger.info(f"Added {len(new_texts)} chunks from '{filename}' to existing ChromaDB.")
            else:
                logger.warning("ChromaDB not initialized, re-initializing with new documents.")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                db = Chroma.from_documents(
                    new_texts,
                    embeddings,
                    persist_directory=config.CHROMA_DB_DIRECTORY,
                    collection_name=config.COLLECTION_NAME
                )
            
            # Re-initialize retriever after db update
            base_retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVER_SEARCH_K})
            try:
                reranker_model = CrossEncoder(config.RERANKER_MODEL_NAME)
                compressor = CrossEncoderCompressor(model=reranker_model, top_n=config.RERANKER_TOP_K)
                retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
            except Exception as e:
                logger.error(f"Could not re-initialize reranker after DB re-initialization: {e}")
                retriever = base_retriever

            agent_executors.clear()
            logger.info("Cleared existing agent sessions. New sessions will use updated knowledge base.")

            return jsonify({"status": "success", "message": f"File '{filename}' processed and added to knowledge base. Knowledge base updated successfully."}), 200
        except Exception as e:
            logger.exception(f"Error processing uploaded file '{file.filename}': {e}")
            if 'upload_path' in locals() and os.path.exists(upload_path):
                os.remove(upload_path)
                logger.info(f"Cleaned up partially processed file: '{filename}'.")
            return jsonify({"status": "error", "message": f"Failed to process file '{file.filename}': An internal error occurred. See server logs for details."}), 500
    else:
        return jsonify({"status": "error", "message": "Invalid file type. Allowed types are: " + ', '.join(ALLOWED_EXTENSIONS)}), 400

# --- API Endpoint for Chat (for external systems / programmatic access) ---
@app.route('/api/bot', methods=['POST'])
def api_bot():
    """
    Handles chat requests, processes questions, and returns answers using the Agent Executor.
    Manages conversation history using session IDs.
    """
    if not request.is_json:
        logger.warning("Received non-JSON request to /api/bot")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id')

    if not question:
        logger.warning(f"Received request with no 'question' for session {session_id}")
        return jsonify({"error": "No 'question' provided in request"}), 400

    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"New session created: {session_id}")
    else:
        logger.info(f"Processing for session: {session_id}")

    # Check if an AgentExecutor is already set up for this session
    if session_id not in agent_executors:
        logger.info(f"Setting up new agent for session {session_id}")
        
        # 1. Create a new ConversationBufferMemory for the session
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # 2. Create the RAG chain for document retrieval
        history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Standalone question:"),
        ])
        if retriever is None:
            logger.error("RAG retriever is not initialized. Cannot create agent executor.")
            return jsonify({"error": "RAG system not fully initialized. Please wait or check server logs."}), 200

        retrieval_chain = create_history_aware_retriever(llm, retriever, history_aware_retriever_prompt)
        
        document_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a diligent and accurate Document Assistant. Provide information clearly and directly.
            Based on the chat history and the provided context documents, answer the user's question. If the question is outside the context, use your general knowledge.
            If you refer to specific information from the documents, you may briefly mention 'based on the documents' or similar, but avoid phrases like 'I cannot answer from the provided context'.

            Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        document_qa_chain = create_stuff_documents_chain(llm, document_qa_prompt)
        
        rag_chain = create_retrieval_chain(retrieval_chain, document_qa_chain)

        # Helper function for the document_qa_retriever tool's 'func'
        # Now uses partial to bind the session's memory
        _run_document_qa_retriever_tool = partial(_run_document_qa_retriever_tool_for_session, session_memory=memory)

        # 3. Define the tools the agent can use for this session
        session_tools = []
        for tool in all_agent_tools:
            if tool.name == "document_qa_retriever":
                session_tools.append(Tool(
                    name="document_qa_retriever",
                    func=_run_document_qa_retriever_tool,
                    description=tool.description
                ))
            else:
                session_tools.append(tool)

        # 4. Create the Agent
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and versatile AI assistant. Your primary goal is to answer questions accurately and directly.
            
            **Tool Usage Priority and Guidelines:**
            - **Document Retrieval:** Use `document_qa_retriever` to answer questions by looking through your uploaded documents AND the pre-scraped web content. This is your primary source for internal knowledge.
            - **Google Search:** Use `Google Search` for general knowledge questions, current events, or anything that requires up-to-date information not found in the provided documents. If a question is clearly about recent events or a broad fact not likely in your specific documents, use this tool.
            - **OpenAPI Tools (GitHub, Weather):** You have access to various specialized tools for specific external services like GitHub and weather. These tools are dynamically loaded from OpenAPI specifications.
                - When asked about GitHub, use the appropriate GitHub tool. Carefully extract necessary parameters like `username`, `owner`, `repo`, etc.
                - When asked about weather, use `get_current_weather`. If a location is vague, ask for clarification (e.g., zip code, city and state/country). Do NOT call the weather tool without a precise location.
            
            **General Knowledge Fallback:**
            - If a question **cannot be answered by any of your tools** and is a common fact, use your inherent general knowledge to provide a concise and relevant answer directly.
            
            **Response Style:**
            - Always provide a direct and factual answer.
            - Think step-by-step in your `agent_scratchpad` to determine the best approach (which tool to use, how to clarify, or when to use general knowledge)."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, session_tools, agent_prompt)
        
        # 5. Create the Agent Executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=session_tools, # Use the session-specific tools
            verbose=True,
            handle_parsing_errors=True,
            memory=memory # Pass the ConversationBufferMemory instance directly
        )
        
        agent_executors[session_id] = agent_executor
        
    else:
        agent_executor = agent_executors[session_id]

    try:
        result = agent_executor.invoke({"input": question})
        
        answer = result.get('output', 'No answer found.')

        logger.info(f"Session {session_id}: Q: '{question}' A: '{answer}'")
        return jsonify({"answer": answer, "session_id": session_id}), 200
    except Exception as e:
        from google.api_core import exceptions as google_exceptions

        if isinstance(e, google_exceptions.InternalServerError) or (
            hasattr(e, 'args') and len(e.args) > 0 and 'InternalServerError' in str(e.args[0])
        ):
            user_message = (
                "I'm sorry, I encountered a temporary issue with the AI service while processing your request (Google API Internal Error). "
                "Please try again in a moment, or rephrase your question."
            )
            logger.error(f"Google Gemini InternalServerError for session {session_id} with question '{question}': {e}")
            return jsonify({"error": user_message, "details": str(e)}), 200
        elif isinstance(e, google_exceptions.ResourceExhausted): # Handle the 429 ResourceExhausted specifically
            user_message = (
                "I'm sorry, the request to the AI model was too large, exceeding its capacity. "
                "This typically happens when too many tools are available or the conversation history is very long. "
                "Please try a simpler query or start a new conversation."
            )
            logger.error(f"Google Gemini ResourceExhausted for session {session_id} with question '{question}': {e}")
            return jsonify({"error": user_message, "details": str(e)}), 200 # Changed to 200
        elif isinstance(e, requests.exceptions.RequestException):
            user_message = (
                "I'm experiencing network issues and cannot connect to external services. "
                "Please check your internet connection and try again."
            )
            logger.error(f"Network/API connection error for session {session_id} with question '{question}': {e}")
            return jsonify({"error": user_message, "details": str(e)}), 200
        else:
            logger.exception(f"An unexpected error occurred during chat for session {session_id} with question '{question}':")
            user_message = (
                "I'm sorry, an unexpected error occurred while processing your request. "
                "Please try again or contact support if the issue persists."
            )
            return jsonify({"error": user_message, "details": str(e)}), 200


# --- Health Check Endpoint (Optional but Recommended) ---
@app.route('/health', methods=['GET'])
def health_check():
    """Provides a simple health check endpoint for the Flask application."""
    logger.info("Health check requested.")
    return jsonify({"status": "healthy", "message": "Document Assistant is running"}), 200

# --- Main execution block (Eager Loading) ---
if __name__ == '__main__':
    logger.info(f"Starting Document Assistant Flask app on http://127.0.0.1:{config.FLASK_PORT}")
    logger.info("Waiting for incoming requests...")

    if len(sys.argv) > 1:
        model_mode = sys.argv[1].lower()
    else:
        model_mode = "cloud"
        logger.info("No model mode provided. Defaulting to 'cloud' mode.")

    ingestion_utils.set_ingestion_config(
        config.WEB_SCRAPER_USER_AGENT, 
        config.SCRAPE_DELAY_SECONDS,
        config.MAX_DOCUMENTS_TO_SCRAPE
    )
    
    initialize_rag_components(model_mode)
    logger.info("RAG components ready for requests!")

    if model_mode == "cloud":
        check_gemini_models()

    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False)