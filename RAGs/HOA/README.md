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
    * [Testing the API (using curl)](#testing-the-api-using-curl)
7.  [Technical Details](#7-technical-details)
    * [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    * [Embedding Model](#embedding-model)
    * [Vector Database (ChromaDB)](#vector-database-chromadb)
    * [Large Language Model (LLM)](#large-language-model-llm)
    * [Conversational Memory](#conversational-memory)
    * [Hybrid RAG & Persona Prompting](#hybrid-rag--persona-prompting)
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
* **Web API:** Exposes a simple HTTP API endpoint (`/chat`) for integration with a web frontend or other applications.
* **Extensible:** Easily add more PDF documents to expand its knowledge base.
* **Open-Source Technologies:** Built using popular Python libraries like LangChain, Flask, and ChromaDB.

## 3. Architecture

### Phase 1 (Current Implementation)

The current architecture is a simple two-tier setup (though Flask acts as a lightweight application server).

* **Client (e.g., `curl`, browser with JavaScript):** Initiates HTTP POST requests to the Flask application. It sends the `question` and a `session_id` (a unique identifier for the conversation).
* **Python Application (Flask `hoa.py`):**
    * Runs as a web server, listening on port `3010`.
    * Initializes and manages the RAG components (Embedding Model, Vector Database, LLM) once per application startup.
    * Maintains an in-memory dictionary (`qa_chain_configs`) to store individual `ConversationalRetrievalChain` instances (each with its own `ConversationBufferMemory`) for each active `session_id`.
    * When a request comes in, it retrieves or creates the appropriate session's memory and RAG chain.
    * Processes the question using the RAG pipeline.
    * Returns the answer as a JSON HTTP response, along with the `session_id`.
* **ChromaDB Vector Store (`./chroma_db`):** Stores the vectorized (embedded) chunks of your HOA documents locally on the filesystem. This acts as the knowledge base for the RAG system.

+----------------+       HTTP POST (question, session_id)       +-------------------+
|   Client       | <------------------------------------------> |  Python App (Flask) |
| (curl, Browser) |                                              | - hoa.py            |
+----------------+       HTTP JSON Response (answer, session_id) | - Listens on 3010   |
| - Manages in-memory |
|   session_id -> RAG Chain (with Memory) |
+-------------------+
|
v
+-------------------+
|   ChromaDB        |
|  (chroma_db folder)|
+-------------------+

### Future Phases (Conceptual)

As discussed, future enhancements could involve:

* **Web Tier:** A dedicated web frontend (HTML/CSS/JavaScript) hosted by a web server (e.g., Nginx) that interacts with this Flask API.
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

export GOOGLE_API_KEY="YOUR_API_KEY_HERE"

ai-workshop/
└── RAGs/
    └── HOA/
        ├── data/                  <-- Place your PDFs here
        │   ├── bylaws.pdf
        │   ├── Declaration-incl-Supplements.pdf
        │   └── pool-rules.pdf
        ├── hoa.py
        └── requirements.txt

rm -rf chroma_db
pip install -r requirements.txt

python hoa.py

You should see output indicating that the Flask app is starting and listening on port 3010:
Starting HOA Assistant Flask app on [http://127.0.0.1:3010](http://127.0.0.1:3010)
Waiting for incoming requests...
* Serving Flask app 'hoa'
* Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
* Running on [http://0.0.0.0:3010](http://0.0.0.0:3010)
Press CTRL+C to quit

Health Check:
Reqest: curl [http://127.0.0.1:3010/health](http://127.0.0.1:3010/health)
Response: {"message": "HOA Assistant is running", "status": "healthy"}

First Request:
Request: curl -X POST -H "Content-Type: application/json" -d "{\"question\": \"What is the annual HOA due amount?\"}" [http://127.0.0.1:3010/chat](http://127.0.0.1:3010/chat)
Response: {"answer": "The annual HOA due amount is $X,000, payable on [date]...", "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"}

IMPORTANT: Copy the session_id value from this response. You'll need it for subsequent requests in the same conversation.

Second Request (in same session):
Reaues: curl -X POST -H "Content-Type: application/json" -d "{\"question\": \"Tell me about the annual meeting.\"}" [http://127.0.0.1:3010/chat](http://127.0.0.1:3010/chat)

```




