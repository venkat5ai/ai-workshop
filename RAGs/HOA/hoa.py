# First, ensure you have the necessary libraries installed.
# You can install them by running the following in your terminal:
# pip install langchain chromadb pypdf sentence-transformers transformers accelerate bitsandbytes langchain_google_genai
# pip install -U langchain-community

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain # For memory-aware RAG
from langchain.memory import ConversationBufferMemory # For storing conversation history
from langchain_core.prompts import ChatPromptTemplate # For custom prompt template
# Removed: from langchain.chains.Youtubeing import load_qa_chain (not strictly needed this way)

import os
import google.generativeai as genai

# --- Configuration ---
# export GOOGLE_API_KEY as an environment variable, LangChain will automatically pick it up.
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # uncomment below line if you want to set it here

# Define the directory where your PDF documents are located
PDF_DIRECTORY = "./data"

# Define the directory where your ChromaDB vector store will be saved
CHROMA_DB_DIRECTORY = "./chroma_db"

# Define the directory where your ChromaDB vector store will be saved
CHROMA_DB_DIRECTORY = "./chroma_db"
# Define a meaningful name for your ChromaDB collection
COLLECTION_NAME = "hoa_docs_collection"

# --- Function to check available Gemini models (Optional, but useful for initial setup) ---
# You can comment out the call to this function once you've confirmed your MODEL_TO_USE.
def check_gemini_models():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable is not set. Please set it to proceed.")
        return []

    genai.configure(api_key=api_key)
    
    print("\n--- Checking available Gemini models ---")
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
                # print(f"  Available model: {m.name}") # Uncomment to see all available models
        print("--- End of available models ---")
    except Exception as e:
        print(f"Error listing models: {e}")
        print("Please double-check your GOOGLE_API_KEY and internet connection.")
    return available_models

# Call the function to check models (can be commented out after initial setup)
# available_gemini_models = check_gemini_models() # Keep this line if you want the check

# --- 1. Load PDF Documents ---
print(f"Loading documents from: {PDF_DIRECTORY}")
try:
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
except Exception as e:
    print(f"Error loading documents: {e}")
    print(f"Please ensure the directory '{PDF_DIRECTORY}' exists and contains PDF files.")
    print("Also, check file permissions for the directory.")
    exit()

# --- 2. Split Documents into Chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Split documents into {len(texts)} chunks.")

# --- 3. Create Embeddings and Store in ChromaDB ---
print("Creating embeddings and storing in ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(CHROMA_DB_DIRECTORY) and os.listdir(CHROMA_DB_DIRECTORY):
    print("Loading existing ChromaDB vector store...")
    # Load with the specified collection name
    db = Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings, collection_name=COLLECTION_NAME) # UPDATED
else:
    print("Creating new ChromaDB vector store...")
    # Create with the specified collection name
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=CHROMA_DB_DIRECTORY,
        collection_name=COLLECTION_NAME # UPDATED
    )    
    print("ChromaDB vector store created and persisted.")

# --- 4. Set up the Language Model (Gemini) ---
# The GOOGLE_API_KEY will be automatically picked up from your environment variables.
# Using the model you selected.
MODEL_TO_USE = "models/gemini-2.5-flash-preview-05-20" # Ensure this is the correct, available model

llm = ChatGoogleGenerativeAI(model=MODEL_TO_USE)
print(f"Using Gemini model: {MODEL_TO_USE}")

# --- 5. Create the Retriever ---
retriever = db.as_retriever(search_kwargs={"k": 3}) # k=3 means it will retrieve the top 3 most relevant chunks

# --- 6. Set up the Retrieval-Augmented Generation (RAG) Chain with Memory ---

# Initialize memory for the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define a custom prompt template for seamless hybrid behavior WITH chat history
# This prompt will be used for the final answer generation
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


# Initialize the ConversationalRetrievalChain
# We explicitly set the `combine_docs_chain_kwargs` to pass our custom prompt
# This avoids the "multiple values for keyword argument 'combine_docs_chain'" error
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory, # Pass the memory object here
    # This automatically uses the LLM to rephrase the current question
    # given the chat history, making it suitable for retrieval.
    # No changes needed for `condense_question_llm` unless you want a different LLM
    # for that specific task.
    # We pass the custom prompt for the `combine_docs_chain` via its kwargs
    combine_docs_chain_kwargs={"prompt": combine_docs_prompt},
    # `return_source_documents` is False by default, matching your preference.
)

print("\nSetup complete! You can now start asking questions.")
print("The assistant will remember the conversation within this session.")

# --- 7. Interactive Q&A Loop ---
while True:
    question = input("\nEnter your question (type 'exit' to quit): ")
    if question.lower() == 'exit':
        print("Exiting the Q&A session. Goodbye!")
        break

    print("\nThinking...")
    try:
        # Pass the question to the ConversationalRetrievalChain
        # It handles chat history automatically via the `memory` object
        result = qa_chain({"question": question}) # Note: key is "question", not "query" for this chain

        answer = result['answer'] # ConversationalRetrievalChain returns 'answer' not 'result'
        
        print("\n--- Answer ---")
        print(answer)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your Google Gemini API key is correct and you have an active internet connection.")