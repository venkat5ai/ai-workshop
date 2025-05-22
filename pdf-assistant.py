import argparse
import os
import json
import requests
import fitz # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # For finding relevant chunks

# --- Configuration for Local LLM and Embedding Model ---
# This is now the true base API URL for Ollama
OLLAMA_API_BASE_URL = "http://localhost:11434/api"
# IMPORTANT: Ensure these models are downloaded via 'ollama run <model_name>'
OLLAMA_LLM_MODEL_NAME = "gemma3"       # Main LLM for answering questions
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text" # Embedding model

# --- Global Data Store for PDF Chunks and Embeddings ---
# This will hold the processed content of your PDFs
vector_store = [] # Stores (embedding, text_chunk) tuples

# --- Helper Functions ---

def get_embedding(text):
    """Calls Ollama's embedding API to get a vector embedding for the given text."""
    try:
        response = requests.post(
            # Corrected endpoint for embeddings
            f"{OLLAMA_API_BASE_URL}/embeddings",
            json={
                "model": OLLAMA_EMBEDDING_MODEL_NAME,
                "prompt": text
            }
        )
        response.raise_for_status()
        return np.array(response.json()['embedding'])
    except requests.exceptions.ConnectionError:
        print(f"[Error] Could not connect to Ollama. Is Ollama running and model '{OLLAMA_EMBEDDING_MODEL_NAME}' downloaded? Check http://localhost:11434")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[Error] Failed to get embedding from local LLM: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        print(f"[Agent] Extracted text from: {os.path.basename(pdf_path)}")
        return text
    except Exception as e:
        print(f"[Error] Could not extract text from {pdf_path}: {e}")
        return None

def chunk_text(text, chunk_size=1000, overlap=100):
    """Splits text into overlapping chunks."""
    chunks = []
    if not text:
        return chunks

    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else []
            current_length = sum(len(w) + 1 for w in current_chunk) -1 if current_chunk else 0 # -1 for space

        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def load_and_process_pdfs(directory_path):
    """Loads PDFs, extracts text, chunks it, and generates embeddings."""
    global vector_store
    vector_store = [] # Clear previous store

    if not os.path.isdir(directory_path):
        print(f"[Agent] Error: Directory '{directory_path}' not found.")
        return

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"[Agent] No PDF files found in '{directory_path}'.")
        return

    print(f"[Agent] Processing {len(pdf_files)} PDF files from '{directory_path}'...")
    all_chunks = []
    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        text = extract_text_from_pdf(file_path)
        if text:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

    if not all_chunks:
        print("[Agent] No text extracted or chunks generated from PDFs.")
        return

    print(f"[Agent] Generating embeddings for {len(all_chunks)} text chunks (this may take a while)...")
    for i, chunk in enumerate(all_chunks):
        embedding = get_embedding(chunk)
        if embedding is not None:
            vector_store.append((embedding, chunk))
        print(f"\r[Agent] Processed {i+1}/{len(all_chunks)} chunks...", end="")
    print("\n[Agent] PDF processing complete. Vector store built.")


def retrieve_relevant_chunks(query, top_k=5):
    """
    Retrieves the most relevant text chunks from the vector store based on the query.
    """
    if not vector_store:
        return []

    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []

    similarities = []
    for i, (chunk_embedding, chunk_text) in enumerate(vector_store):
        # Reshape for sklearn's cosine_similarity
        similarity = cosine_similarity(query_embedding.reshape(1, -1), chunk_embedding.reshape(1, -1))[0][0]
        similarities.append((similarity, chunk_text))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Return top_k chunks
    return [chunk for sim, chunk in similarities[:top_k]]

def answer_question_with_context(user_question, context_chunks):
    """
    Uses the local LLM (via Ollama) to answer a question based on provided context.
    """
    if not context_chunks:
        return "I don't have enough relevant information in the loaded PDFs to answer that question."

    context_str = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the following question based ONLY on the provided context.
If the answer cannot be found in the context, state that you don't have enough information.

Context:
{context_str}

Question: "{user_question}"
Answer:
"""
    print(f"\n[Agent] Sending question and context to local LLM ({OLLAMA_LLM_MODEL_NAME})...")
    try:
        response = requests.post(
            # Corrected endpoint for text generation
            f"{OLLAMA_API_BASE_URL}/generate",
            json={
                "model": OLLAMA_LLM_MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        llm_response_data = response.json()
        llm_response_text = llm_response_data['response'].strip()
        print(f"[Agent] Local LLM Raw Response:\n{llm_response_text}")
        return llm_response_text
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to Ollama. Is Ollama running and model '{OLLAMA_LLM_MODEL_NAME}' downloaded? Check http://localhost:11434"
    except requests.exceptions.RequestException as e:
        return f"Error calling local LLM API: {e}"
    except Exception as e:
        return f"An unexpected error occurred with local LLM: {e}"

# --- Main Application Logic ---
def main():
    parser = argparse.ArgumentParser(description="Local AI Assistant for PDF Analysis (Ollama).")
    args = parser.parse_args()

    print(f"\n[Agent] Starting Local PDF Assistant. Ensure Ollama is running and models '{OLLAMA_LLM_MODEL_NAME}' and '{OLLAMA_EMBEDDING_MODEL_NAME}' are downloaded.")
    print("Commands:")
    print("  - 'load_pdfs <directory_path>': Load and analyze PDFs from a local directory.")
    print("  - '<your_question>': Ask a question based on the loaded PDFs.")
    print("  - 'exit' or 'quit': End the session.")

    while True:
        user_input = input("\n[You] Enter command or question: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("[Agent] Exiting session. Goodbye!")
            break

        if not user_input:
            print("[Agent] Please enter a command or question.")
            continue

        if user_input.lower().startswith("load_pdfs "):
            pdf_dir_path = user_input[len("load_pdfs "):].strip()
            load_and_process_pdfs(pdf_dir_path)
        else:
            # Treat as a natural language question
            if not vector_store:
                print("[Agent] Please load PDFs first using 'load_pdfs <directory_path>' before asking questions.")
                continue

            # Retrieve relevant chunks
            relevant_chunks = retrieve_relevant_chunks(user_input)

            if not relevant_chunks:
                print("[Agent] I couldn't find any relevant information in the loaded PDFs for your question.")
                continue

            # Answer question with context
            answer = answer_question_with_context(user_input, relevant_chunks)
            print("\n[Agent] Answer:")
            print(answer)

if __name__ == "__main__":
    main()
