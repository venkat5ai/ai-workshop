# First, ensure you have the necessary libraries installed.
# You can install them by running the following in your terminal:
# pip install Flask langchain chromadb pypdf sentence-transformers langchain-community langchain-huggingface langchain-chroma Flask-CORS unstructured[pdf,docx,csv,pptx] python-magic python-docx openpyxl

"""
This script implements a Retrieval-Augmented Generation (RAG) assistant
using Flask for the API, LangChain for orchestration, and various
libraries for document processing and LLM interaction. It supports
both local (Ollama) and cloud (Google Gemini) LLM modes.
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

# --- Flask and related imports ---
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Core LangChain and Data Processing Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# NEW: Imports for LangChain Agents and Tools
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool


# --- Application Configuration ---
import config

# --- Logging Setup ---
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables for the Flask App ---
llm = None
retriever = None
# This dictionary will store AgentExecutor instances, keyed by session_id.
# Each AgentExecutor will manage its own memory.
agent_executors = {} 

# --- Flask App Instance ---
app = Flask(__name__)
CORS(app)

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

    params = {
        "appid": api_key,
        "units": "imperial" # or "metric" for Celsius
    }

    # Attempt to parse location more robustly
    parts = [p.strip() for p in location.split(',')]
    
    # Try to infer location based on common patterns
    if len(parts) == 1:
        if parts[0].isdigit() and len(parts[0]) == 5:
            params["zip"] = f"{parts[0]},us" # Assume US for 5-digit zip
        else:
            params["q"] = parts[0] # Assume it's a city name
    elif len(parts) == 2:
        params["q"] = f"{parts[0]},{parts[1]}" # Assume city, state/country code
    elif len(parts) >= 3:
        params["q"] = f"{parts[0]},{parts[1]},{parts[2]}" # Assume city, state, country
    else:
        return f"Invalid location format: '{location}'. Please provide a city, zip code, or city,state/country."

    try:
        response = requests.get(config.OPENWEATHER_BASE_URL, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        weather_data = response.json()
        
        if weather_data.get("main") and weather_data.get("weather"):
            main_info = weather_data["main"]
            weather_desc = weather_data["weather"][0]["description"]
            city_name = weather_data.get("name", location)
            
            # --- IMPORTANT: SUMMARIZE TOOL OUTPUT FOR LLM ---
            summary = (
                f"Current weather in {city_name}: "
                f"{main_info.get('temp')}°F, feels like {main_info.get('feels_like')}°F. "
                f"Conditions: {weather_desc}. Humidity: {main_info.get('humidity')}%. "
                f"Wind speed: {weather_data.get('wind', {}).get('speed')} mph."
            )
            return summary
        else:
            return f"Could not retrieve full weather data for '{location}'. API response was incomplete."

    except requests.exceptions.RequestException as e:
        logger.error(f"Error making API call to OpenWeatherMap for '{location}': {e}")
        if "404 Client Error" in str(e):
             return f"Weather data not found for the location '{location}'. Please ensure the location is valid and spelled correctly. If it's a city, try adding a state (e.g., 'Ashburn, VA') or country (e.g., 'Paris, FR')."
        return f"Failed to retrieve weather data due to API error: {e}. Check network connection or API key."
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from OpenWeatherMap response for '{location}': {e}")
        return f"Failed to parse weather data response from API."
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_current_weather for '{location}': {e}")
        return "An unexpected error occurred while fetching weather data."

# Define the weather tool (this can be global as it doesn't need session-specific state)
weather_tool = Tool(
    name="get_current_weather",
    func=get_current_weather,
    description="Useful for fetching current weather conditions for a specified location. Input must be a single string representing the location, such as a zip code (e.g., '90210'), a precise city name (e.g., 'London'), or a combination of city and state/country (e.g., 'Ashburn, VA' or 'Paris, FR')."
)


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
    global llm, retriever

    # --- 1. Load Documents from Various Types ---
    logger.info(f"Loading documents from: {config.DOCUMENT_STORAGE_DIRECTORY} (supporting multiple file types via UnstructuredFileLoader)")
    try:
        loader = DirectoryLoader(
            config.DOCUMENT_STORAGE_DIRECTORY,
            loader_cls=UnstructuredFileLoader,
            recursive=True,
            show_progress=True,
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s) of various types.")
    except Exception as e:
        logger.exception(f"Detailed error during document loading: {e}")
        logger.error(f"Error loading documents: {e}")
        logger.error(f"Please ensure the directory '{config.DOCUMENT_STORAGE_DIRECTORY}' exists and contains document files.")
        logger.error("Also, check file permissions for the directory and that unstructured dependencies are installed.")
        sys.exit(1)

    # --- 2. Split Documents into Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
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
        # Using a slightly higher temperature (e.g., 0.2) can sometimes help with agent reasoning
        # and natural language generation, reducing InternalServerErrors.
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_TO_USE, temperature=0.2) 
        logger.info(f"Using Google Gemini cloud model: {config.GEMINI_MODEL_TO_USE}")
    else:
        logger.error(f"Invalid model mode: {model_mode}. Please use 'local' or 'cloud'.")
        sys.exit(1)

    # --- 5. Create the Retriever ---
    retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVER_SEARCH_K})
    
    logger.info("\nRAG components initialized.")


# --- Route for the Web UI (e.g., for local testing and debugging) ---
@app.route('/')
def index():
    """Renders the main index page for the web UI."""
    return render_template('index.html')


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
                return f"Error retrieving document answer: {e}"

        # 3. Define the tools the agent can use
        tools = [
            weather_tool, # Our new weather tool (globally defined)
            # Wrap the RAG chain using the helper function
            Tool(
                name="document_qa_retriever",
                func=_run_document_qa_retriever_tool_for_session, # Use the dynamic helper function
                description="Useful for answering questions about the uploaded documents. Input should be the user's question, ideally rephrased if it's a follow-up."
            ),
        ]
        
        # 4. Create the Agent
        # --- MODIFIED AGENT PROMPT ---
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and versatile AI assistant. Your primary goal is to answer questions accurately and directly.
            
            **Prioritize tool usage when appropriate:**
            - **Document Questions:** Use `document_qa_retriever` to find answers in the uploaded documents.
            - **Weather Questions:** Use `get_current_weather` for current weather conditions.
                - **Important:** If the location is vague (e.g., "Ashburn") or appears incorrect (e.g., "Ashburn NC"), try to infer the most likely precise location (e.g., "Ashburn, VA" if common) or, if unsure, **ask the user for clarification (e.g., "Which Ashburn are you referring to?")** before calling the tool. Provide the most precise location possible to the tool (e.g., "London", "90210", "Ashburn, VA", "Paris, FR").
            
            **For general knowledge questions that do not require tools** (e.g., "how far is Earth from the Moon?", "who won FIFA last?"), use your inherent knowledge to provide a concise and relevant answer. Do not apologize for not using a tool if you can answer directly.
            
            Always provide a direct answer. Think step-by-step to determine the best approach (tool or general knowledge)."""),
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

        # AgentExecutor's memory will handle saving context if initialized with it.
        # This manual save_context might be redundant if memory is working perfectly
        # with AgentExecutor, but doesn't hurt. Keeping for explicit control.
        # Access memory from the executor itself:
        # agent_executor.memory.save_context({"input": question}, {"output": answer})


        logger.info(f"Session {session_id}: Q: '{question}' A: '{answer}'")
        return jsonify({"answer": answer, "session_id": session_id}), 200
    except Exception as e:
        logger.exception(f"An error occurred during chat for session {session_id} with question '{question}':")
        return jsonify({"error": "An internal error occurred while processing your request.", "details": str(e)}), 500

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

    initialize_rag_components(model_mode)
    logger.info("RAG components ready for requests!")

    if model_mode == "cloud":
        check_gemini_models()

    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False)