# First, ensure you have the necessary libraries installed.
# You can install them by running the following in your terminal:
# pip install langchain chromadb pypdf sentence-transformers transformers accelerate bitsandbytes langchain_google_genai
# pip install -U langchain-community

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate # For custom prompt template
from langchain.chains.question_answering import load_qa_chain

import os
import google.generativeai as genai

# --- Configuration ---  
# export GOOGLE_API_KEY as an environment variable, LangChain will automatically pick it up.
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # uncomment below line if you want to set it here

# Define the directory where your PDF documents are located
PDF_DIRECTORY = "./data"

# Define the directory where your ChromaDB vector store will be saved
CHROMA_DB_DIRECTORY = "./chroma_db"

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
    db = Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings)
else:
    print("Creating new ChromaDB vector store...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_DIRECTORY)
    db.persist()
    print("ChromaDB vector store created and persisted.")

# --- 4. Set up the Language Model (Gemini) ---
# The GOOGLE_API_KEY will be automatically picked up from your environment variables.
# Using the model you selected.
MODEL_TO_USE = "models/gemini-2.5-flash-preview-05-20" # Ensure this is the correct, available model

llm = ChatGoogleGenerativeAI(model=MODEL_TO_USE)
print(f"Using Gemini model: {MODEL_TO_USE}")

# --- 5. Create the Retriever ---
retriever = db.as_retriever(search_kwargs={"k": 3}) # k=3 means it will retrieve the top 3 most relevant chunks

# --- 6. Set up the Retrieval-Augmented Generation (RAG) Chain with Hybrid Behavior ---
# Define a custom prompt template for seamless hybrid behavior
custom_template = """
You are a diligent and accurate HOA assistant. Provide information clearly and directly.

If the user's question can be fully answered using the provided 'Context' documents, prioritize those documents and provide a precise answer based solely on them.
If the 'Context' documents do not contain relevant information or sufficient detail to answer the question, then use your general knowledge to answer the question.
In all cases, provide a direct answer without explicitly stating whether the answer came from the provided documents or your general knowledge.
If you refer to specific information from the documents, you may briefly mention 'based on the documents' or similar, but avoid phrases like 'I cannot answer from the provided context'.

Context:
{context}

Question: {question}
"""
custom_prompt = ChatPromptTemplate.from_template(custom_template)

# Build the internal QA chain with the custom prompt
# This replaces the simple chain_type="stuff" usage in RetrievalQA.from_chain_type
combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=custom_prompt)

# Create the RetrievalQA chain using the custom combine_docs_chain
qa_chain = RetrievalQA(
    combine_documents_chain=combine_docs_chain,
    retriever=retriever,
    return_source_documents=True # Still useful for debugging or optional display
)

print("\nSetup complete! You can now start asking questions.")

# --- 7. Interactive Q&A Loop ---
while True:
    question = input("\nEnter your question (type 'exit' to quit): ")
    if question.lower() == 'exit':
        print("Exiting the Q&A session. Goodbye!")
        break

    print("\nThinking...")
    try:
        result = qa_chain({"query": question})
        answer = result['result']
        source_documents = result['source_documents']

        print("\n--- Answer ---")
        print(answer)

        # You can still conditionally show sources if you want, or just always say
        # "Answer provided." based on whether source_documents exist.
        print("\n--- Sources (Indicates if documents were highly relevant) ---")
        if source_documents:
            print("  Relevant documents found:")
            for i, doc in enumerate(source_documents):
                print(f"  Source {i+1}: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
        else:
            print("  No highly relevant documents were directly cited for this answer.")
            # Note: This doesn't mean it *didn't* try to look; it means the retriever
            # didn't return documents that the LLM deemed directly useful for the final answer.

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your Google Gemini API key is correct and you have an active internet connection.")

        