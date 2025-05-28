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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Imports for LangChain Agents and Tools
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool

# --- NEW: Imports for Reranking ---
from sentence_transformers import CrossEncoder # For reranking
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor # <--- NEW IMPORT

# --- NEW: Import ingestion utilities ---
import ingestion_utils # Import the new file

# --- Application Configuration ---
import config

# --- Logging Setup ---
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables for the Flask App ---
llm = None
retriever = None
agent_executors = {}
db = None # Make db a global variable so it can be accessed for updates

# --- Flask App Instance ---
app = Flask(__name__)
CORS(app)

# --- Define allowed extensions for uploads ---
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'csv', 'xlsx', 'pptx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Tool Definitions ---
def get_current_weather(location: str) -> str:
    """
    Fetches current weather data for a given location (zip code, city, city,state, or city,country)
    using OpenWeatherMap.

    Args:
        location (str): The location string (e.g., "28227", "London", "Ashburn, VA", "Paris, FR").

    Returns:
        str: A summarized string of the weather data, or an informative error message.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logger.error("OPENWEATHER_API_KEY environment variable is not set for weather tool.")
        return "Weather API key not configured."

    base_url = config.OPENWEATHER_BASE_URL
    params = {
        "appid": api_key,
        "units": "imperial" # or "metric" for Celsius
    }

    # Attempt to parse location more robustly and try multiple formats if needed
    # Prioritize precise formats first, then fall back to less precise.
    location_attempts = []
    original_parts = [p.strip() for p in location.split(',')]

    # Attempt 1: Exact input as provided
    if location:
        if location.isdigit() and len(location) == 5:
            location_attempts.append({"zip": f"{location},us"})
        else:
            location_attempts.append({"q": location})
    
    # Attempt 2: If input is "City, State_Code", try parsing as "City,State_Code" and then "City"
    if len(original_parts) == 2:
        city, state_or_country = original_parts
        location_attempts.append({"q": f"{city},{state_or_country}"})
        location_attempts.append({"q": city}) # Also try just the city
    
    # Attempt 3: If input is just a city name, ensure it's in the list
    if len(original_parts) == 1 and not (original_parts[0].isdigit() and len(original_parts[0]) == 5):
        if {"q": original_parts[0]} not in location_attempts:
            location_attempts.append({"q": original_parts[0]})


    # Remove duplicates from attempts (e.g., if "London" was added twice)
    seen = set()
    unique_attempts = []
    for d in location_attempts:
        t = tuple(sorted(d.items()))
        if t not in seen:
            unique_attempts.append(d)
            seen.add(t)

    # Specific workaround for known tricky locations if LLM still passes them
    lower_location = location.lower()
    if "ashburn" in lower_location and "nc" in lower_location:
        # If the LLM insists on Ashburn, NC, let's explicitly try Ashburn, VA as a common correction
        if {"q": "Ashburn,VA"} not in unique_attempts:
            unique_attempts.insert(0, {"q": "Ashburn,VA"}) # Try VA first for Ashburn issues

    final_attempts = unique_attempts
    
    for attempt_params in final_attempts:
        current_params = {**params, **attempt_params} # Combine base params with attempt-specific params
        query_identifier = attempt_params.get("q") or attempt_params.get("zip")
        logger.info(f"Attempting weather API call for: {query_identifier} (Original: {location})")

        try:
            response = requests.get(base_url, params=current_params)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            weather_data = response.json()
            
            if weather_data.get("main") and weather_data.get("weather"):
                main_info = weather_data["main"]
                weather_desc = weather_data["weather"][0]["description"]
                city_name = weather_data.get("name", query_identifier.split(',')[0]) # Use name from response, fallback to part of query
                
                summary = (
                    f"Current weather in {city_name}: "
                    f"{main_info.get('temp')}°F, feels like {main_info.get('feels_like')}°F. "
                    f"Conditions: {weather_desc}. Humidity: {main_info.get('humidity')}%. "
                    f"Wind speed: {weather_data.get('wind', {}).get('speed')} mph."
                )
                logger.info(f"Successfully retrieved weather for '{query_identifier}'.")
                return summary # Return immediately on success
            else:
                logger.warning(f"Incomplete weather data for '{query_identifier}'. Trying next option if available.")
                continue # Try next attempt if data is incomplete

        except requests.exceptions.RequestException as e:
            logger.warning(f"OpenWeatherMap API call failed for '{query_identifier}': {e}. Trying next option if available.")
            continue # Try next attempt on API error
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decoding failed for '{query_identifier}' from OpenWeatherMap: {e}. Trying next option if available.")
            continue # Try next attempt on JSON error
        except Exception as e:
            logger.error(f"Unexpected error in get_current_weather tool for '{query_identifier}': {e}. Trying next option if available.")
            continue # Try next attempt on unexpected error

    # If all attempts fail, return a consolidated and actionable error message for the LLM
    return (
        f"I was unable to retrieve weather data for '{location}' using the weather tool. "
        f"This could be due to an invalid or ambiguous location, or a temporary issue with the weather service. "
        f"Please try again with a more precise location, such as a zip code (e.g., '90210') or a city and state/country (e.g., 'Ashburn, VA' or 'Paris, FR')."
    )

# Define the weather tool (this can be global as it doesn't need session-specific state)
weather_tool = Tool(
    name="get_current_weather",
    func=get_current_weather,
    description="Useful for fetching current weather conditions for a specified location. Input must be a single string representing the location, such as a zip code (e.g., '90210'), a precise city name (e.g., 'London'), or a combination of city and state/country (e.g., 'Ashburn, VA' or 'Paris, FR'). The tool will try its best to resolve the location given."
)

# --- CrossEncoderCompressor Class (MOVED TO GLOBAL SCOPE AND NOW IMPLEMENTS abstract method) ---
class CrossEncoderCompressor(BaseDocumentCompressor): # Inherit from BaseDocumentCompressor
    def __init__(self, model: CrossEncoder, top_n: int = 3):
        # IMPORTANT: Set model and top_n BEFORE calling super().__init__()
        # This ensures they are available when BaseDocumentCompressor's __init__
        # potentially runs validation or other logic that might expect them.
        self.model = model
        self.top_n = top_n
        super().__init__() # Call the constructor of the parent class
        logger.info(f"CrossEncoderCompressor initialized with top_n={top_n}")

    # This is the concrete implementation of the abstract method from BaseDocumentCompressor
    def compress_documents(
        self, documents: list[Document], query: str, callbacks=None
    ) -> list[Document]:
        if not documents:
            return []
        
        passages = [doc.page_content for doc in documents]
        sentence_pairs = [[query, passage] for passage in passages]
        # self.model should now be guaranteed to be initialized here
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
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set. Please set it to proceed for Gemini model features.")
        return []

    genai.configure(api_key=api_key)
    
    logger.info("--- Checking available Gemini models ---")
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        logger.info("--- End of available models ---")
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        logger.error("Please double-check your GOOGLE_API_KEY and internet connection if using Gemini.")
    return available_models


# --- Function to initialize the RAG components ---
def initialize_rag_components(model_mode: str):
    """
    Initializes the RAG pipeline components: document loader, text splitter,
    embedding model, vector store (ChromaDB), and language model.

    Args:
        model_mode (str): The mode for the language model ('local' for Ollama, 'cloud' for Gemini).
    """
    global llm, retriever, db # Added db to global

    # --- 1. Load Documents from Various Types (Local files and Web) ---
    all_documents = []

    # Load local documents
    logger.info(f"Loading local documents from: {config.DOCUMENT_STORAGE_DIRECTORY} (supporting multiple file types via LangChainUnstructuredFileLoader)")
    try:
        local_loader = DirectoryLoader(
            config.DOCUMENT_STORAGE_DIRECTORY,
            loader_cls=LangChainUnstructuredFileLoader, # Use the aliased class
            recursive=True,
            show_progress=True,
        )
        local_documents = local_loader.load()
        all_documents.extend(local_documents)
        logger.info(f"Loaded {len(local_documents)} local document(s).")
    except Exception as e:
        logger.exception(f"Detailed error during local document loading: {e}")
        logger.error(f"Error loading local documents: {e}")
        logger.error(f"Please ensure the directory '{config.DOCUMENT_STORAGE_DIRECTORY}' exists and contains document files.")
        sys.exit(1)

    # Load web documents from web.conf via deep scraping
    # Set the ingestion config for user agent, scrape delay, and max documents
    ingestion_utils.set_ingestion_config(
        config.WEB_SCRAPER_USER_AGENT, 
        config.SCRAPE_DELAY_SECONDS, 
        config.MAX_DOCUMENTS_TO_SCRAPE
    )
    web_docs = ingestion_utils.load_web_documents_deep_scrape(config.DOCUMENT_STORAGE_DIRECTORY) # Pass the document storage directory for web.conf
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
    else:
        logger.info("Creating new ChromaDB vector store...")
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=config.CHROMA_DB_DIRECTORY,
            collection_name=config.COLLECTION_NAME
        )
        logger.info("ChromaDB vector store created and persisted.")

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
    
    # Initialize the reranker model
    logger.info(f"Loading reranker model: {config.RERANKER_MODEL_NAME}...")
    try:
        reranker_model = CrossEncoder(config.RERANKER_MODEL_NAME)
        # Instantiate the globally defined CrossEncoderCompressor, now correctly implementing abstract method
        compressor = CrossEncoderCompressor(reranker_model, top_n=config.RERANKER_TOP_K)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        logger.info("ContextualCompressionRetriever with Reranker initialized.")

    except Exception as e:
        logger.error(f"Error loading or initializing reranker model '{config.RERANKER_MODEL_NAME}': {e}")
        logger.error("Proceeding without reranking. Retrieval quality might be lower.")
        retriever = base_retriever # Fallback to base retriever if reranker fails
        
    logger.info("\nRAG components initialized.")


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
    global db, retriever # Need to access and potentially re-initialize retriever if db changes

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(config.DOCUMENT_STORAGE_DIRECTORY, filename)
            
            # Ensure the directory exists
            os.makedirs(config.DOCUMENT_STORAGE_DIRECTORY, exist_ok=True)
            
            file.save(upload_path)
            logger.info(f"File '{filename}' saved to '{upload_path}'.")

            # --- Process the new document and update ChromaDB ---
            logger.info(f"Processing uploaded document: '{filename}'...")
            loader = LangChainUnstructuredFileLoader(upload_path) # Using the aliased class
            new_documents = loader.load()
            
            if not new_documents:
                logger.warning(f"No content extracted from uploaded file: '{filename}'.")
                os.remove(upload_path) # Clean up empty file
                return jsonify({"status": "warning", "message": f"File '{filename}' uploaded but no text content could be extracted. It was not added to the knowledge base."}), 200

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            new_texts = text_splitter.split_documents(new_documents)
            logger.info(f"Split uploaded document into {len(new_texts)} chunks.")

            if db:
                # Add new documents to the existing ChromaDB collection
                db.add_documents(new_texts)
                logger.info(f"Added {len(new_texts)} chunks from '{filename}' to existing ChromaDB.")
            else:
                # If db was not initialized for some reason, re-initialize with the new docs
                logger.warning("ChromaDB not initialized, re-initializing with new documents.")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                db = Chroma.from_documents(
                    new_texts,
                    embeddings,
                    persist_directory=config.CHROMA_DB_DIRECTORY,
                    collection_name=config.COLLECTION_NAME
                )
                # Re-initialize retriever if db object changed
                base_retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVER_SEARCH_K})
                try: # Re-add reranking if it was active
                    reranker_model = CrossEncoder(config.RERANKER_MODEL_NAME)
                    # Use globally defined class, now correctly implementing abstract method
                    compressor = CrossEncoderCompressor(reranker_model, top_n=config.RERANKER_TOP_K)
                    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
                except Exception as e:
                    logger.error(f"Could not re-initialize reranker after DB re-initialization: {e}")
                    retriever = base_retriever # Fallback

            # Clear existing agent executors so they pick up the new retriever on next session
            agent_executors.clear()
            logger.info("Cleared existing agent sessions. New sessions will use updated knowledge base.")

            return jsonify({"status": "success", "message": f"File '{filename}' processed and added to knowledge base. Knowledge base updated successfully."}), 200
        except Exception as e:
            logger.exception(f"Error processing uploaded file '{file.filename}': {e}")
            # Attempt to clean up the partially processed file if an error occurs
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
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            ("user", "Standalone question:"),
        ])
        # IMPORTANT: Ensure 'retriever' is globally initialized and valid here
        if retriever is None:
            logger.error("RAG retriever is not initialized. Cannot create agent executor.")
            # Do not return 500, return 200 with an error message in body
            return jsonify({"error": "RAG system not fully initialized. Please wait or check server logs."}), 200

        retrieval_chain = create_history_aware_retriever(llm, retriever, history_aware_retriever_prompt)
        
        document_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a diligent and accurate Document Assistant. Provide information clearly and directly.
            Based on the chat history and the provided context documents, answer the user's question. If the question is outside the context, use your general knowledge.
            If you refer to specific information from the documents, you may briefly mention 'based on the documents' or similar, but avoid phrases like 'I cannot answer from the provided context'.

            Context: {context}"""),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ])
        
        document_qa_chain = create_stuff_documents_chain(llm, document_qa_prompt)
        
        # This is the RAG chain for answering questions based on documents
        rag_chain = create_retrieval_chain(retrieval_chain, document_qa_chain)

        # Helper function for the document_qa_retriever tool's 'func'
        # This function needs to access the session's memory
        def _run_document_qa_retriever_tool_for_session(query: str) -> str:
            """
            Runs the RAG chain with the given query and the current session's chat history.
            This function is used as the 'func' for the document_qa_retriever tool.
            It captures the `memory` object from the enclosing scope.
            """
            # Access the memory object for the current session.
            # This is safe because this function is defined within the scope where `memory` is available.
            
            # The agent typically passes just the question string.
            # We need to construct the dictionary expected by rag_chain.invoke.
            input_dict = {"input": query, "chat_history": memory.load_memory_variables({})["chat_history"]}
            
            try:
                result = rag_chain.invoke(input_dict)
                return result.get('answer', 'No answer found from documents.')
            except Exception as e:
                logger.error(f"Error in document_qa_retriever tool for query '{query}': {e}")
                return f"Error retrieving document answer: An issue occurred while processing the document context."

        # 3. Define the tools the agent can use
        tools = [
            weather_tool, # Our existing weather tool
            Tool(
                name="document_qa_retriever",
                func=_run_document_qa_retriever_tool_for_session, # Use the dynamic helper function
                description="Useful for answering questions about the uploaded documents and pre-scraped web content. Input should be the user's question, ideally rephrased if it's a follow-up."
            ),
        ]
        
        # 4. Create the Agent
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and versatile AI assistant. Your primary goal is to answer questions accurately and directly.
            
            **Tool Usage Priority:**
            - **Document Retrieval:** Use `document_qa_retriever` to answer questions by looking through your uploaded documents AND the pre-scraped web content.
            - **Weather Information:** Use `get_current_weather` for real-time weather.
                - **Clarification First:** If a location is vague (e.g., just "Ashburn") or seems incorrect (e.g., "Ashburn NC"), *always ask the user for clarification (e.g., "Which Ashburn are you referring to?")* or **suggest a more precise format** (e.g., "Please provide a zip code or a city and state/country, like 'Ashburn, VA' or 'Paris, FR'"). **Do not call the weather tool until you have a clear, precise location.**
            
            **General Knowledge Fallback:**
            - If a question **cannot be answered by any of your tools** (e.g., "how far is Earth from the Moon?", "who won FIFA last?", general facts), use your inherent general knowledge to provide a concise and relevant answer directly. Do not apologize for not using a tool if you can answer directly.
            
            **Response Style:**
            - Always provide a direct and factual answer.
            - Think step-by-step in your `agent_scratchpad` to determine the best approach (which tool to use, how to clarify, or when to use general knowledge)."""),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, agent_prompt)
        
        # 5. Create the Agent Executor
        # Pass the memory directly to the AgentExecutor.
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory # Pass the ConversationBufferMemory instance directly
        )
        
        # Store the AgentExecutor for the session
        agent_executors[session_id] = agent_executor
        
    else:
        # Retrieve the AgentExecutor for an existing session
        agent_executor = agent_executors[session_id]
        # Memory is managed by the AgentExecutor, no explicit retrieval needed here.

    try:
        # The AgentExecutor's `invoke` method automatically uses its associated `memory`
        # if initialized with it.
        result = agent_executor.invoke({"input": question})
        
        answer = result.get('output', 'No answer found.')


        logger.info(f"Session {session_id}: Q: '{question}' A: '{answer}'")
        return jsonify({"answer": answer, "session_id": session_id}), 200
    except Exception as e:
        # Catch specific Google API errors for better user feedback
        from google.api_core import exceptions as google_exceptions # <--- Moved import here for consistency

        if isinstance(e, google_exceptions.InternalServerError) or (
            hasattr(e, 'args') and len(e.args) > 0 and 'InternalServerError' in str(e.args[0])
        ):
            user_message = (
                "I'm sorry, I encountered a temporary issue with the AI service while processing your request (Google API Internal Error). "
                "Please try again in a moment, or rephrase your question."
            )
            logger.error(f"Google Gemini InternalServerError for session {session_id} with question '{question}': {e}")
            return jsonify({"error": user_message, "details": str(e)}), 200 # Changed to 200
        elif isinstance(e, requests.exceptions.RequestException):
            user_message = (
                "I'm experiencing network issues and cannot connect to external services (like weather). "
                "Please check your internet connection and try again."
            )
            logger.error(f"Network/API connection error for session {session_id} with question '{question}': {e}")
            return jsonify({"error": user_message, "details": str(e)}), 200 # Changed to 200
        else:
            # For any other unexpected errors
            logger.exception(f"An unexpected error occurred during chat for session {session_id} with question '{question}':")
            user_message = (
                "I'm sorry, an unexpected error occurred while processing your request. "
                "Please try again or contact support if the issue persists."
            )
            return jsonify({"error": user_message, "details": str(e)}), 200 # Changed to 200


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

    # Set ingestion config before loading docs
    ingestion_utils.set_ingestion_config(
        config.WEB_SCRAPER_USER_AGENT, 
        config.SCRAPE_DELAY_SECONDS,
        config.MAX_DOCUMENTS_TO_SCRAPE # Pass the new configurable value
    )
    
    initialize_rag_components(model_mode)
    logger.info("RAG components ready for requests!")

    if model_mode == "cloud":
        check_gemini_models()

    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False)