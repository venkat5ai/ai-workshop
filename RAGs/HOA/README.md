# 🏡 HOA Assistant - AI-Powered Document Q&A (Flask Web Service)

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Features](#2-features)
3.  [Architecture](#3-architecture)
    * [Phase 1 (Current Implementation)](#phase-1-current-implementation)
    * [Future Phases (Conceptual)](#future-phases-conceptual)
4.  [Prerequisites](#4-prerequisites)
5.  [Setup Instructions](#5-setup-instructions)
    * [1. Clone the Repository](#1-clone-the-repository)
    * [2. Configure Google Gemini API Key](#2-configure-google-gemini-api-key)
    * [3. Place Your HOA Documents](#3-place-your-hoa-documents)
    * [4. Install Dependencies](#4-install-dependencies)
    * [5. Initialize the ChromaDB Vector Store](#5-initialize-the-chromadb-vector-store)
6.  [Running the Application](#6-running-the-application)
    * [Running the Flask Server](#running-the-flask-server)
    * [Accessing the Web UI](#accessing-the-web-ui)
    * [Testing the API (using curl)](#testing-the-api-using-curl)
    * [Health Check](#health-check)
7.  [Technical Details](#7-technical-details)
    * [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    * [Embedding Model](#embedding-model)
    * [Vector Database (ChromaDB)](#vector-database-chromadb)
    * [Large Language Model (LLM)](#large-language-model-llm)
    * [Conversational Memory](#conversational-memory)
    * [Hybrid RAG & Persona Prompting](#hybrid-rag--persona-prompting)
    * [CORS (Cross-Origin Resource Sharing)](#cors-cross-origin-resource-sharing)
    * [Eager Loading](#eager-loading)
8.  [Project Structure](#8-project-structure)
9.  [Troubleshooting](#9-troubleshooting)
10. [Future Enhancements](#10-future-enhancements)
11. [License](#11-license)

---

## 1. Introduction

The HOA Assistant is an AI-powered question-answering system designed to help HOA residents, board members, or property managers quickly find information within their community's documents. Leveraging Retrieval-Augmented Generation (RAG), the assistant can answer questions based on specific HOA bylaws, declarations, financial reports, and pool rules.

This project implements the core RAG logic as a Flask web service, listening for HTTP requests and providing conversational, context-aware answers.

## 2. Features

* **Document-Grounded Answers:** Provides responses directly from your uploaded HOA PDF documents.
* **Hybrid RAG:** Seamlessly falls back to the Large Language Model's (LLM) general knowledge if information is not found in the provided documents, without explicit user notification.
* **Conversational Memory:** Remembers the context of previous questions within a single user session for more natural interactions.
* **Web API (`/api/bot`):** Exposes a dedicated HTTP API endpoint for integration with a web frontend or other backend applications.
* **Simple Web UI (`/`):** Provides a basic browser-based chat interface for easy local testing and interaction.
* **Eager Loading:** RAG components (ChromaDB, LLM, Embedding Model) are initialized at application startup, eliminating "cold start" delays for the first request.
* **Extensible:** Easily add more PDF documents to expand its knowledge base.
* **Open-Source Technologies:** Built using popular Python libraries like LangChain, Flask, and ChromaDB.

## 3. Architecture

### Phase 1 (Current Implementation)

The current architecture provides a clear separation between the UI and the API layer.

* **Client (Web Browser):** Loads the `index.html` UI from the Flask app's root (`/`). Its JavaScript code then makes HTTP POST requests to the `/api/bot` endpoint.
* **Client (External System/`curl`):** Initiates HTTP POST requests directly to the `/api/bot` endpoint.
* **Python Application (Flask `hoa.py`):**
    * Runs as a web server, listening on port `3010`.
    * **Serves the Web UI:** The root endpoint (`/`) serves the `templates/index.html` file.
    * **API Endpoint:** The `/api/bot` endpoint handles all JSON-based question-answering requests.
    * **Eager Initialization:** Initializes and manages the RAG components (Embedding Model, Vector Database, LLM) once per application startup, before listening for requests.
    * **Session Management:** Maintains an in-memory dictionary (`qa_chain_configs`) to store individual `ConversationalRetrievalChain` instances (each with its own `ConversationBufferMemory`) for each active `session_id`.
    * When a request comes in, it retrieves or creates the appropriate session's memory and RAG chain.
    * Processes the question using the RAG pipeline.
    * Returns the answer as a JSON HTTP response, along with the `session_id`.
* **ChromaDB Vector Store (`./chroma_db`):** Stores the vectorized (embedded) chunks of your HOA documents locally on the filesystem. This acts as the knowledge base for the RAG system.

+----------------+       HTTP GET (HTML/JS)          +-------------------+
|                | &lt;-------------------------------> |  Python App (Flask) |
|   Web Browser  |                                   | - hoa.py            |
| (Web UI on /)  |                                   | - Listens on 3010   |
+----------------+                                   | - Serves UI on /    |
^                                            | - Handles API on /api/bot |
| HTTP POST (question, session_id)           | - Manages in-memory |
| (from JS)                                  |   session_id -> RAG Chain (with Memory) |
v                                            +-------------------+
+----------------+                                            |
| External System|                                            v
|    (e.g., curl) | -------------------------------------->  +-------------------+
+----------------+   HTTP POST (question, session_id)        |   ChromaDB        |
&lt;---------------------------------------   |  (chroma_db folder)|
HTTP JSON Response (answer, session_id)   +-------------------+

### Future Phases (Conceptual)

As discussed, future enhancements could involve:

* **Production WSGI Server:** Using Gunicorn or Uvicorn to run the Flask app for better concurrency and stability in production.
* **Containerization (Docker):** Packaging the Flask app into a Docker container for portable and consistent deployment.
* **Orchestration (Kubernetes):** Managing Docker containers in a Kubernetes cluster for scalability, high availability, and automated deployments.
* **Externalized Session Memory:** Migrating `ConversationBufferMemory` to a shared, external store like Redis to support horizontal scaling of the Flask application (where multiple instances can share session state).
* **Externalized Vector Database:** For very large datasets or high concurrency, using a standalone, network-accessible vector database service instead of a local filesystem-based ChromaDB.

## 4. Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
* **`pip` (Python package installer)**: Usually comes with Python.
* **Google Gemini API Key**: You'll need an API key from Google AI Studio. You can get one here: [Google AI Studio](https://aistudio.google.com/app/apikey). This key should be set as an environment variable.
* **Access to your HOA PDF documents**.

## 5. Setup Instructions

### 1. Clone the Repository

If this project is part of a larger Git repository, ensure you have it cloned. Assuming your project root is `ai-workshop` and this specific project is under `RAGs/HOA`:

```bash
git clone <your-repository-url>
cd ai-workshop/RAGs/HOA
2. Configure Google Gemini API Key
Set your Google Gemini API key as an environment variable. Replace YOUR_API_KEY_HERE with your actual key.

For Linux/macOS:
Bash

export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
For Windows (Command Prompt):
Bash

set GOOGLE_API_KEY="YOUR_API_KEY_HERE"
For Windows (PowerShell):
PowerShell

$env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
IMPORTANT: This setting is usually temporary for the current terminal session. For persistent setting, refer to your OS documentation.
3. Place Your HOA Documents
Create a directory named data within your HOA project folder if it doesn't exist. Place all your HOA PDF documents inside this data folder.

ai-workshop/
└── RAGs/
    └── HOA/
        ├── data/                  <-- Place your PDFs here
        │   ├── bylaws.pdf
        │   ├── Declaration-incl-Supplements.pdf
        │   └── pool-rules.pdf
        ├── templates/             <-- Contains index.html for Web UI
        │   └── index.html
        ├── hoa.py
        └── requirements.txt
4. Install Dependencies
It's recommended to use a Python virtual environment.

Bash

python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
requirements.txt content:

Flask
Flask-Cors
langchain
chromadb
pypdf
sentence-transformers
transformers
accelerate
bitsandbytes
langchain_google_genai
langchain-community
langchain-huggingface
langchain-chroma
5. Initialize the ChromaDB Vector Store
The first time you run hoa.py, it will automatically create the chroma_db directory and build the vector store from your PDFs. If you update your documents or want to rebuild the vector store, you can delete the chroma_db folder before running the application again.

Bash

rm -rf chroma_db # (Use 'rmdir /s /q chroma_db' on Windows)
6. Running the Application
Running the Flask Server
Navigate to your HOA directory in the terminal and run:

Bash

python hoa.py
You should see output similar to this, indicating that the Flask app is starting and listening on port 3010:

Starting HOA Assistant Flask app on [http://127.0.0.1:3010](http://127.0.0.1:3010)
Waiting for incoming requests...
Loading documents from: ./data
Loaded XXX document(s).
Split documents into YYY chunks.
Creating embeddings and storing in ChromaDB...
Creating new ChromaDB vector store... OR Loading existing ChromaDB vector store...
Using Gemini model: models/gemini-2.5-flash-preview-05-20
RAG components initialized.
RAG components ready for requests!
--- Checking available Gemini models ---
--- End of available models ---
 * Debugger is active!
 * Debugger PIN: XXX-XXX-XXX
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
 * Running on [http://0.0.0.0:3010](http://0.0.0.0:3010) (Press CTRL+C to quit)
Accessing the Web UI
Once the server is running, open your web browser and navigate to:

[http://127.0.0.1:3010/](http://127.0.0.1:3010/)
You will see the chat interface. You can type your questions and interact with the HOA Assistant. The UI will automatically manage the session ID.

Testing the API (using curl)
You can directly interact with the API endpoint using curl commands.

First Request (to start a new session):
A session_id will be generated if not provided. Copy this session_id from the response.

Bash

curl -X POST -H "Content-Type: application/json" -d "{\"question\": \"What is the annual HOA due amount?\"}" [http://127.0.0.1:3010/api/bot](http://127.0.0.1:3010/api/bot)
Example Response:

JSON

{"answer": "The annual HOA due amount is $X,000, payable on [date]...", "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"}
Subsequent Requests (in the same session):
Use the session_id obtained from the first request to continue the conversation.

Bash

curl -X POST -H "Content-Type: application/json" -d "{\"question\": \"When is the annual meeting?\", \"session_id\": \"a1b2c3d4-e5f6-7890-1234-567890abcdef\"}" [http://127.0.0.1:3010/api/bot](http://127.0.0.1:3010/api/bot)
Health Check
You can check if the Flask application is running and responsive by hitting the /health endpoint:

Bash

curl [http://127.0.0.1:3010/health](http://127.0.0.1:3010/health)
Expected Response:

JSON

{"message": "HOA Assistant is running", "status": "healthy"}
7. Technical Details
Retrieval-Augmented Generation (RAG)
The core of the system is RAG, which combines the power of a large language model with a retrieval mechanism. When a question is asked, the system first retrieves relevant chunks of information from the HOA documents (based on semantic similarity) and then uses the LLM to generate an answer grounded in that retrieved context.

Embedding Model
The all-MiniLM-L6-v2 Sentence Transformer model is used to convert text (document chunks and user questions) into numerical vector embeddings. These embeddings allow for efficient semantic search within the vector database.

Vector Database (ChromaDB)
ChromaDB is used as a lightweight, local vector store. It persists the document embeddings on the filesystem in the ./chroma_db directory, allowing for fast retrieval of relevant document chunks during the RAG process without re-embedding documents on every application start (unless the chroma_db folder is deleted).

Large Language Model (LLM)
The system utilizes the models/gemini-2.5-flash-preview-05-20 model from Google Gemini through the langchain_google_genai integration. This powerful LLM is responsible for understanding the question, synthesizing information from retrieved documents, and generating coherent, conversational answers.

Conversational Memory
ConversationBufferMemory from LangChain is used to store the history of the current conversation within each user session. This allows the LLM to consider previous turns in the dialogue when generating responses, leading to more natural and contextually aware interactions. Each session_id (managed by the client, e.g., the web UI's JavaScript or explicitly by curl) is tied to its own memory instance.

Hybrid RAG & Persona Prompting
The system is designed to provide answers primarily from the HOA documents. However, if the retrieved context is insufficient, the LLM's general knowledge will be leveraged, providing a seamless fallback without explicitly informing the user that the answer came from outside the documents. The LLM is also prompted to act as a "diligent and accurate HOA assistant."

CORS (Cross-Origin Resource Sharing)
The Flask-CORS extension is used to handle Cross-Origin Resource Sharing headers. This is essential to allow the JavaScript running in your web browser (which typically considers localhost and 127.0.0.1 as different origins) to make POST requests to your Flask API endpoint.

Eager Loading
The RAG components (ChromaDB vector store loading/building, LLM initialization, embedding model setup) are performed once at the Flask application's startup. This "eager loading" strategy ensures that the application is fully ready to serve requests as soon as it begins listening on the specified port, eliminating "cold start" delays for the first API call.

8. Project Structure
ai-workshop/
└── RAGs/
    └── HOA/
        ├── data/
        │   ├── <your-hoa-document-1.pdf>
        │   └── <your-hoa-document-N.pdf>
        ├── chroma_db/                 <-- Auto-generated by application (stores vector embeddings)
        │   └── <chroma_db_files>
        ├── templates/
        │   └── index.html             <-- HTML for the web UI
        ├── .env                       <-- (Optional) For environment variables
        ├── hoa.py                     <-- Main Flask application
        └── requirements.txt           <-- Python dependencies
9. Troubleshooting
"Error: Could not connect to the assistant." (in browser UI):
Ensure your hoa.py server is running.
Verify Flask-CORS is installed (pip install Flask-CORS) and CORS(app) is enabled in hoa.py.
Check your browser's developer console (F12, then "Console" or "Network" tab) for CORS errors.
Ensure the JavaScript Workspace URL in index.html (currently http://127.0.0.1:3010/api/bot) correctly matches your Flask app's address and port.
"Error listing models" or LLM-related errors (in Flask console):
Double-check that your GOOGLE_API_KEY environment variable is correctly set in the terminal before you run python hoa.py.
Generate a new API key from Google AI Studio and try again.
Ensure your machine has internet access to reach Google's API.
"Loaded 0 document(s)." or RAG-related errors:
Verify that your PDF documents are placed in the ./data directory relative to where hoa.py is run.
Check file permissions for the data directory and the PDF files.
Slow first response: If this happens after you've implemented eager loading, ensure rm -rf chroma_db was run before testing to force a rebuild, or check if the ChromaDB loading phase is indeed slow.
10. Future Enhancements
Streamlined Document Updates: Implement an API endpoint to upload new documents and dynamically update the vector store without restarting the application.
User Authentication/Authorization: Add a login system to restrict access to the chat assistant.
Error Handling and Logging: More robust logging and error handling for production environments.
Advanced RAG Techniques: Explore more sophisticated retrieval methods (e.g., HyDE, Cohere Rerank) or multi-modal RAG.
Deployment Automation: Scripts or configurations for automated deployment to cloud platforms (AWS, GCP, Azure).
