# First, ensure you have the necessary libraries installed.
# You can install them by running the following in your terminal:
# pip install Flask langchain chromadb pypdf sentence-transformers transformers accelerate bitsandbytes langchain_google_genai langchain-community langchain-huggingface langchain-chroma Flask-CORS

# --- Standard Library Imports ---
import uuid
import os
import logging
import google.generativeai as genai

# --- Flask and related imports ---
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Core LangChain and Data Processing Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
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
qa_chain_configs = {} # Dictionary to store qa_chain instances per session_id

# --- Function to check available Gemini models (Optional, but useful for initial setup) ---
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

# --- Function to initialize the RAG components ---
def initialize_rag_components():
    global llm, retriever

    # --- 1. Load PDF Documents ---
    logger.info(f"Loading documents from: {config.PDF_DIRECTORY}")
    try:
        loader = PyPDFDirectoryLoader(config.PDF_DIRECTORY)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s).")
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        logger.error(f"Please ensure the directory '{config.PDF_DIRECTORY}' exists and contains PDF files.")
        logger.error("Also, check file permissions for the directory.")
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

    # --- 4. Set up the Language Model (Gemini) ---
    llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_TO_USE)
    logger.info(f"Using Gemini model: {config.GEMINI_MODEL_TO_USE}")

    # --- 5. Create the Retriever ---
    retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVER_SEARCH_K})
    
    logger.info("\nRAG components initialized.")

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Route for the Web UI (e.g., for local testing and debugging) ---
@app.route('/')
def index():
    return render_template('index.html')

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
    initialize_rag_components()
    logger.info("RAG components ready for requests!")

    # Call the function to check models (can be commented out after initial setup)
    check_gemini_models()

    # Run the Flask app
    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False)