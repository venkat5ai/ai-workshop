# First, ensure you have the necessary libraries installed.
# You can install them by running the following in your terminal:
# pip install Flask langchain chromadb pypdf sentence-transformers langchain-community langchain-huggingface langchain-chroma Flask-CORS unstructured[pdf,docx,csv,pptx] python-magic python-docx openpyxl


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
import sys
import google.generativeai as genai

# --- Flask and related imports ---
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Core LangChain and Data Processing Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Corrected imports for handling diverse document types
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader # This is the loader class for individual unstructured files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- Application Configuration ---
import config

# --- Logging Setup ---
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global variables for the Flask App ---
llm = None
retriever = None
qa_chain_configs = {}

# --- Function to check available Gemini models (Optional, useful for initial setup) ---
def check_gemini_models():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set. Please set it to proceed for Gemini model.")
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
def initialize_rag_components(model_mode):
    global llm, retriever

    # --- 1. Load Documents from Various Types ---
    logger.info(f"Loading documents from: {config.DOCUMENT_STORAGE_DIRECTORY} (supporting multiple file types via UnstructuredFileLoader)")
    try:
        # Use DirectoryLoader with UnstructuredFileLoader as the class for each file
        # This is the recommended way in newer LangChain versions to use unstructured on directories.
        loader = DirectoryLoader(
            config.DOCUMENT_STORAGE_DIRECTORY,
            loader_cls=UnstructuredFileLoader, # Specify UnstructuredFileLoader for each document
            recursive=True,
            show_progress=True,
            # You can also use loader_kwargs for specific UnstructuredFileLoader settings, e.g.:
            # loader_kwargs={"mode": "elements", "strategy": "hi_res"}
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s) of various types.")
    except Exception as e:
        logger.exception(f"Detailed error during document loading: {e}") # ADDED/MODIFIED LINE FOR DETAILED ERROR
        logger.error(f"Error loading documents: {e}")
        logger.error(f"Please ensure the directory '{config.DOCUMENT_STORAGE_DIRECTORY}' exists and contains document files.")
        logger.error("Also, check file permissions for the directory and that unstructured dependencies are installed.")
        exit()

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
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_TO_USE)
        logger.info(f"Using Google Gemini cloud model: {config.GEMINI_MODEL_TO_USE}")
    else:
        logger.error(f"Invalid model mode: {model_mode}. Please use 'local' or 'cloud'.")
        sys.exit(1)

    # --- 5. Create the Retriever ---
    retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVER_SEARCH_K})
    
    logger.info("\nRAG components initialized.")

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- Route for the Web UI (e.g., for local testing and debugging) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API Endpoint for Chat (for external systems / programmatic access) ---
@app.route('/api/bot', methods=['POST'])
def api_bot():
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

    if session_id not in qa_chain_configs:
        logger.info(f"Setting up new chain and memory for session {session_id}")
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language."),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            ("user", "Standalone question:"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, history_aware_retriever_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a diligent and accurate Document Assistant. Provide information clearly and directly.
Based on the chat history and the provided context documents (if relevant), or your general knowledge (if context is not relevant), answer the user's question. Provide a direct answer without explicitly stating whether the answer came from the provided documents or your general knowledge.
If you refer to specific information from the documents, you may briefly mention 'based on the documents' or similar, but avoid phrases like 'I cannot answer from the provided context'.

Context: {context}"""),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ])
        
        Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

        session_qa_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)
        
        qa_chain_configs[session_id] = {"chain": session_qa_chain, "memory": memory}
    else:
        session_qa_chain = qa_chain_configs[session_id]["chain"]
        memory = qa_chain_configs[session_id]["memory"]

    try:
        chat_history_messages = memory.load_memory_variables({})["chat_history"]
        invoke_input = {"input": question, "chat_history": chat_history_messages}
        result = session_qa_chain.invoke(invoke_input)
        answer = result.get('answer', 'No answer found.')
        memory.save_context({"input": question}, {"output": answer})


        logger.info(f"Session {session_id}: Q: '{question}' A: '{answer}'")
        return jsonify({"answer": answer, "session_id": session_id}), 200
    except Exception as e:
        logger.exception(f"An error occurred during chat for session {session_id} with question '{question}':")
        return jsonify({"error": "An internal error occurred while processing your request.", "details": str(e)}), 500

# --- Health Check Endpoint (Optional but Recommended) ---
@app.route('/health', methods=['GET'])
def health_check():
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