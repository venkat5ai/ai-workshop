# First, ensure you have the necessary libraries installed.
# You can install them by running the following in your terminal:
# pip install Flask langchain chromadb pypdf sentence-transformers transformers accelerate bitsandbytes langchain_google_genai langchain-community langchain-huggingface langchain-chroma Flask-CORS

# --- Standard Library Imports ---
import uuid
import os
import logging
import google.generativeai as genai
import shutil
import time # NEW: Import time module for sleep
import gc   # NEW: Import gc module for garbage collection

# --- Flask and related imports ---
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Core LangChain and Data Processing Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain

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
db = None # NEW: Declare db as global, will be initialized in rebuild_knowledge_base

# --- Function to check available Gemini models ---
def check_gemini_models():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set. Please set it to proceed.")
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
        logger.error("Please double-check your GOOGLE_API_KEY and internet connection.")
    return available_models

# --- Function to load documents and rebuild ChromaDB (without clearing data/ directory) ---
def rebuild_knowledge_base():
    global llm, retriever, qa_chain_configs, db # Added db to global here

    logger.info(f"Starting knowledge base rebuild from: {config.PDF_DIRECTORY}")
    documents = []

    try:
        # Load PDFs
        pdf_loader = PyPDFDirectoryLoader(config.PDF_DIRECTORY)
        documents.extend(pdf_loader.load())
        logger.info(f"Loaded PDFs from '{config.PDF_DIRECTORY}'. Current documents count: {len(documents)}")

        # Load TXT files
        for filename in os.listdir(config.PDF_DIRECTORY):
            filepath = os.path.join(config.PDF_DIRECTORY, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(".txt"):
                txt_loader = TextLoader(filepath)
                documents.extend(txt_loader.load())
                logger.info(f"Loaded text file: {filename}")
            elif os.path.isfile(filepath) and not filename.lower().endswith((".pdf", ".txt")):
                logger.info(f"Skipping unsupported file type: {filename}")
                
    except Exception as e:
        logger.error(f"Error during document loading from '{config.PDF_DIRECTORY}': {e}")
        logger.error("Check your PDF/TXT files and directory structure.")
        raise # Re-raise the exception to be caught by the calling function

    if not documents:
        logger.warning(f"No supported documents found in '{config.PDF_DIRECTORY}'. The RAG will rely purely on LLM general knowledge.")
        retriever = None # Ensure retriever is None if no documents
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_TO_USE) # Still initialize LLM
        logger.info("Only LLM initialized as no documents found for RAG.")
        qa_chain_configs.clear() # Clear all existing session memories if KB is purged
        return

    logger.info(f"Total pages/files processed for knowledge base: {len(documents)}.")

    # Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(texts)} chunks.")

    # Create Embeddings and Store in ChromaDB
    logger.info("Creating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- FIX START: Release ALL relevant references and retry deletion ---
    # Explicitly clear global references to potentially held ChromaDB objects and chains
    retriever = None
    llm = None
    qa_chain_configs.clear()
    db = None # NEW: Explicitly clear the db object reference

    gc.collect() # NEW: Force garbage collection to release handles

    # Clear previous ChromaDB instance (if any) before creating a new one
    if os.path.exists(config.CHROMA_DB_DIRECTORY):
        max_retries = 5
        retry_delay = 0.5 # seconds
        for i in range(max_retries):
            try:
                shutil.rmtree(config.CHROMA_DB_DIRECTORY)
                logger.info(f"Deleted existing ChromaDB at '{config.CHROMA_DB_DIRECTORY}'.")
                break # Success, exit retry loop
            except OSError as e:
                if i < max_retries - 1:
                    logger.warning(f"Attempt {i+1}/{max_retries}: Error deleting ChromaDB directory: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to delete ChromaDB after {max_retries} attempts: {e}. Another process might be holding files.")
                    raise # Re-raise if still failing after retries
    # --- FIX END ---

    # Re-instantiate db, llm, retriever after successful deletion/creation
    db = Chroma.from_documents( # This creates the new db instance
        texts,
        embeddings,
        persist_directory=config.CHROMA_DB_DIRECTORY,
        collection_name=config.COLLECTION_NAME
    )
    logger.info("ChromaDB vector store created and persisted from all current documents.")

    # Re-Set up the Language Model (Gemini)
    llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_TO_USE)
    logger.info(f"Using Gemini model: {config.GEMINI_MODEL_TO_USE}")

    # Re-Create the Retriever
    retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVER_SEARCH_K})
    
    logger.info("RAG components rebuilt successfully.")


# --- Main Initialization on App Startup ---
# This function will now be responsible for the initial cleanup AND rebuilding.
def initialize_app_on_startup():
    global db # NEW: Declare db as global here too for initial cleanup
    logger.info(f"Preparing application environment: Cleaning '{config.PDF_DIRECTORY}' and '{config.CHROMA_DB_DIRECTORY}'.")
    try:
        # Clear all files and subdirectories from the PDF_DIRECTORY
        if os.path.exists(config.PDF_DIRECTORY):
            for filename in os.listdir(config.PDF_DIRECTORY):
                file_path = os.path.join(config.PDF_DIRECTORY, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path) # Delete file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) # Delete sub-directories
            logger.info(f"Cleared all content from '{config.PDF_DIRECTORY}'.")
        else:
            os.makedirs(config.PDF_DIRECTORY, exist_ok=True) # Ensure it exists if it was deleted
            logger.info(f"Created '{config.PDF_DIRECTORY}' as it did not exist.")

        # Delete the ChromaDB directory (this will be rebuilt by rebuild_knowledge_base)
        if os.path.exists(config.CHROMA_DB_DIRECTORY):
            # Ensure any references are released before deletion during initial startup too
            db = None # Clear global db reference
            gc.collect() # Force garbage collection
            shutil.rmtree(config.CHROMA_DB_DIRECTORY)
            logger.info(f"Deleted existing ChromaDB at '{config.CHROMA_DB_DIRECTORY}'.")
        
    except Exception as e:
        logger.error(f"Critical error during initial cleanup: {e}. Please check file permissions.")
        exit() # Cannot proceed without clean environment

    # Now, rebuild the knowledge base (which will be empty if no initial docs are copied to data/ after cleanup)
    rebuild_knowledge_base() # Call the new function for initial setup
    logger.info("RAG components ready for requests!")


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Route for the Web UI (e.g., for local testing and debugging) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API Endpoint for Document Upload ---
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        logger.warning("No 'file' part in the upload request.")
        return jsonify({"error": "No file part"}), 400

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        logger.warning("No selected file for upload.")
        return jsonify({"error": "No selected file"}), 400

    filename = uploaded_file.filename
    # Basic filename sanitation to prevent path traversal
    if '/' in filename or '\\' in filename:
        filename = os.path.basename(filename)

    filepath = os.path.join(config.PDF_DIRECTORY, filename) # Saving to PDF_DIRECTORY
    
    # Check for allowed file extensions (case-insensitive)
    allowed_extensions = ('.pdf', '.txt')
    if not filename.lower().endswith(allowed_extensions):
        logger.warning(f"Attempted upload of unsupported file type: {filename}")
        return jsonify({"error": "Unsupported file type. Only PDF and TXT are allowed."}), 400

    try:
        uploaded_file.save(filepath)
        logger.info(f"File '{filename}' uploaded successfully to {filepath}")

        # Rebuild RAG components to include the new document
        logger.info("Triggering full ChromaDB rebuild to include new document(s). This might take a moment...")
        rebuild_knowledge_base() # Call the new function for rebuilding
        logger.info("ChromaDB rebuilt successfully with new document(s).")
        
        return jsonify({"message": f"File '{filename}' uploaded and knowledge base updated successfully!"}), 200

    except Exception as e:
        logger.exception(f"Error during file upload or ChromaDB rebuild for '{filename}':")
        return jsonify({"error": f"Failed to upload or process file: {str(e)}"}), 500
    
    logger.error("An unexpected error occurred during /api/upload processing.")
    return jsonify({"error": "An unexpected error occurred during upload."}), 500

# --- API Endpoint for Chat (for external systems / programmatic access) ---
@app.route('/api/bot', methods=['POST'])
def api_bot():
    # Ensure request is JSON
    if not request.is_json:
        logger.warning("Received non-JSON request to /api/bot")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id') # Get session_id from client

    if not question:
        logger.warning(f"Received request with no 'question' for session {session_id}")
        return jsonify({"error": "No 'question' provided in request"}), 400

    # If no session_id provided by client, generate a new one
    if not session_id:
        session_id = str(uuid.uuid4()) # Generate a UUID for the session
        logger.info(f"New session created: {session_id}")
    else:
        logger.info(f"Processing for session: {session_id}")

    # Get or create the qa_chain for this session
    if session_id not in qa_chain_configs:
        logger.info(f"Setting up new qa_chain for session {session_id}")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define the custom prompt template for the final answer generation
        custom_template = """
You are a diligent and accurate HOA assistant. Provide information clearly and directly.

Chat History:
{chat_history}
Context:
{context}
Question:
{question}

Based on the chat history and the provided context documents (if relevant), or your general knowledge (if context is not relevant), answer the user's question. Provide a direct answer without explicitly stating whether the answer came from the provided documents or your general knowledge.
If you refer to specific information from the documents, you may briefly mention 'based on the documents' or similar, but avoid phrases like 'I cannot answer from the provided context'.
"""
        # Create the prompt from the template
        combine_docs_prompt = ChatPromptTemplate.from_template(custom_template)

        # Define the Question Generator Prompt and Chain
        question_generator_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
        question_generator_prompt = ChatPromptTemplate.from_template(question_generator_template)
        question_generator_chain = LLMChain(llm=llm, prompt=question_generator_prompt)

        # Define the Combine Documents Chain
        combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=combine_docs_prompt)

        # Initialize the ConversationalRetrievalChain explicitly
        session_qa_chain = ConversationalRetrievalChain(
            retriever=retriever,
            question_generator=question_generator_chain,
            combine_docs_chain=combine_docs_chain,
            memory=memory,
        )
        qa_chain_configs[session_id] = session_qa_chain
    else:
        session_qa_chain = qa_chain_configs[session_id]

    # Process the question
    try:
        result = session_qa_chain({"question": question})
        answer = result['answer']
        logger.info(f"Session {session_id}: Q: '{question}' A: '{answer}'")
        return jsonify({"answer": answer, "session_id": session_id}), 200
    except Exception as e:
        logger.exception(f"An error occurred during chat for session {session_id} with question '{question}':")
        return jsonify({"error": "An internal error occurred while processing your request.", "details": str(e)}), 500

# --- Health Check Endpoint (Optional but Recommended) ---
@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested.")
    return jsonify({"status": "healthy", "message": "HOA Assistant is running"}), 200


# --- Main execution block (Eager Loading) ---
if __name__ == '__main__':
    logger.info(f"Starting HOA Assistant Flask app on http://127.0.0.1:{config.FLASK_PORT}")
    logger.info("Waiting for incoming requests...")

    # --- EAGER LOADING OF RAG COMPONENTS ---
    # This will now happen once when the application first starts.
    # This function now also handles cleanup of old docs and chroma_db
    initialize_app_on_startup()
    logger.info("Initial RAG setup complete. App ready.")

    # Call the function to check models (can be commented out after initial setup)
    check_gemini_models()

    # Run the Flask app
    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False)