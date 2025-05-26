# 🏡 AI Agent - AI-Powered Document Q&A (Flask Web Service)

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Features](#2-features)
3.  [Architecture](#3-architecture)
    * [Phase 1 (Current Implementation)](#phase-1-current-implementation)
    * [Future Phases (Conceptual)](#future-phases-conceptual)
4.  [Prerequisites](#4-prerequisites)
5.  [Setup Instructions (Local Development)](#5-setup-instructions-local-development)
    * [1. Clone the Repository](#1-clone-the-repository)
    * [2. Configure Google Gemini API Key](#2-configure-google-gemini-api-key)
    * [3. Place Your Documents](#3-place-your-documents)
    * [4. Install Dependencies](#4-install-dependencies)
    * [5. Initialize the ChromaDB Vector Store](#5-initialize-the-chromadb-vector-store)
6.  [Running the Application](#6-running-the-application)
    * [Running the Flask Server Locally](#running-the-flask-server-locally)
    * [**Containerized Deployment (Docker)**](#containerized-deployment-docker)
        * [Docker Prerequisites](#docker-prerequisites)
        * [Build the Docker Image](#build-the-docker-image)
        * [Run the Docker Container](#run-the-docker-container)
        * [Inspect the Container (Optional)](#inspect-the-container-optional)
    * [Accessing the Web UI](#accessing-the-web-ui)
    * [Testing the API (using curl)](#testing-the-api-using-curl)
    * [Health Check](#health-check)
7.  [Technical Details](#7-technical-details)
    * [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    * [Document Handling (Unstructured)](#document-handling-unstructured)
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

The  Assistant is an AI-powered question-answering system designed to help  residents, board members, or property managers quickly find information within their community's documents. Leveraging Retrieval-Augmented Generation (RAG), the assistant can answer questions based on specific  bylaws, declarations, financial reports, and pool rules.

This project implements the core RAG logic as a Flask web service, listening for HTTP requests and providing conversational, context-aware answers.

## 2. Features

* **Multi-Document Type Support:** Processes a wide range of document formats, including PDFs, DOCX, XLSX, CSV, PPTX, and plain text files (`.txt`), ensuring a comprehensive knowledge base from your  documents.
* **Document-Grounded Answers:** Provides responses directly from your uploaded  documents.
* **Hybrid RAG:** Seamlessly falls back to the Large Language Model's (LLM) general knowledge if information is not found in the provided documents, without explicit user notification.
* **Conversational Memory:** Remembers the context of previous questions within a single user session for more natural interactions.
* **Web API (`/api/bot`):** Exposes a dedicated HTTP API endpoint for integration with a web frontend or other backend applications.
* **Simple Web UI (`/`):** Provides a basic browser-based chat interface for easy local testing and interaction.
* **Eager Loading:** RAG components (ChromaDB, LLM, Embedding Model) are initialized at application startup, eliminating "cold start" delays for the first request.
* **Extensible:** Easily add more documents to expand its knowledge base.
* **Containerized:** Provided Dockerfile for consistent and portable deployment across different environments.
* **Open-Source Technologies:** Built using popular Python libraries like LangChain, Flask, and ChromaDB.

## 3. Architecture

### Phase 1 (Current Implementation)

The current architecture provides a clear separation between the UI and the API layer.

* **Client (Web Browser):** Loads the `index.html` UI from the Flask app's root (`/`). Its JavaScript code then makes HTTP POST requests to the `/api/bot` endpoint.
* **Client (External System/`curl`):** Initiates HTTP POST requests directly to the `/api/bot` endpoint.
* **Python Application (Flask `agent.py`):**
    * Runs as a web server, listening on port `3010`.
    * **Serves the Web UI:** The root endpoint (`/`) serves the `templates/index.html` file.
    * **API Endpoint:** The `/api/bot` endpoint handles all JSON-based question-answering requests.
    * **Eager Initialization:** Initializes and manages the RAG components (Embedding Model, Vector Database, LLM) once per application startup, before listening for requests.
    * **Session Management:** Maintains an in-memory dictionary (`qa_chain_configs`) to store individual `ConversationalRetrievalChain` instances (each with its own `ConversationBufferMemory`) for each active `session_id`.
    * When a request comes in, it retrieves or creates the appropriate session's memory and RAG chain.
    * Processes the question using the RAG pipeline.
    * Returns the answer as a JSON HTTP response, along with the `session_id`.
* **ChromaDB Vector Store (`./chroma_db`):** Stores the vectorized (embedded) chunks of your  documents locally on the filesystem. This acts as the knowledge base for the RAG system.


+----------------+       HTTP GET (HTML/JS)          +-------------------+
|                | &lt;-------------------------------> |  Python App (Flask) |
|   Web Browser  |                                   | - agent.py          |
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
* **Containerization (Docker):** Packaging the Flask app into a Docker container for portable and consistent deployment. (This is now implemented as part of this project's setup!)
* **Orchestration (Kubernetes):** Managing Docker containers in a Kubernetes cluster for scalability, high availability, and automated deployments.
* **Externalized Session Memory:** Migrating `ConversationBufferMemory` to a shared, external store like Redis to support horizontal scaling of the Flask application (where multiple instances can share session state).
* **Externalized Vector Database:** For very large datasets or high concurrency, using a standalone, network-accessible vector database service instead of a local filesystem-based ChromaDB.

## 4. Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
* **`pip` (Python package installer)**: Usually comes with Python.
* **Google Gemini API Key**: You'll need an API key from Google AI Studio. You can get one here: [Google AI Studio](https://aistudio.google.com/app/apikey). This key should be set as an environment variable.
* **Your  documents** in various formats (PDF, DOCX, XLSX, etc.).
* **Docker Desktop (for Windows/macOS) or Docker Engine (for Linux servers)** if you plan to run the application in a container.

## 5. Setup Instructions (Local Development)

These instructions are for setting up and running the application directly on your host machine without Docker. If you plan to use Docker, skip to the [Containerized Deployment](#containerized-deployment-docker) section.

### 1. Clone the Repository

If this project is part of a larger Git repository, ensure you have it cloned. Assuming your project root is `ai-workshop` and this specific project is under `RAGs/`:

```bash
git clone <your-repository-url>
cd ai-workshop/RAGs/
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
IMPORTANT: This setting is usually temporary for the current terminal session. For persistent setting, refer to your OS documentation or consider using a .env file and python-dotenv (not implemented by default in this project).

3. Place Your  Documents
Create a directory named data within your  project folder if it doesn't exist. Place all your  documents (.pdf, .docx, .xlsx, .csv, .pptx, .txt, etc.) inside this data folder.

ai-workshop/
└── RAGs/
    └── /
        ├── data/                  <-- Place your documents here
        │   ├── bylaws.pdf
        │   ├── Timesheet-November-2024.docx
        │   ├── Financial-Summary-Q4-2024.xlsx
        │   └── rules.txt
        ├── templates/             <-- Contains index.html for Web UI
        │   └── index.html
        ├── agent.py               <-- Main Flask application
        ├── config.py              <-- Configuration file
        ├── requirements.txt       <-- Python dependencies
        └── Dockerfile             <-- Docker build instructions
4. Install Dependencies
It's recommended to use a Python virtual environment.

Bash

python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
requirements.txt content (as of latest successful build):

Flask==3.1.1
Flask-CORS==6.0.0
langchain==0.3.25
langchain_community==0.3.24
langchain_google_genai # No version, let pip resolve if possible
langchain-huggingface==0.2.0
langchain-chroma==0.2.4
chromadb==1.0.10
pypdf==4.2.0
sentence-transformers==2.7.0
transformers==4.41.2
accelerate==0.30.1
bitsandbytes
unstructured[pdf,docx,csv,xlsx,pptx]==0.14.7
python-magic==0.4.27
python-docx>=1.1.2
openpyxl==3.1.2
google-generativeai # No version, let pip resolve if possible
pysqlite3-binary
pdfminer.six==20221105
5. Initialize the ChromaDB Vector Store
The first time you run agent.py, it will automatically create the chroma_db directory and build the vector store from your documents. If you update your documents or want to rebuild the vector store, you can delete the chroma_db folder before running the application again.

Bash

rm -rf chroma_db # (Use 'rmdir /s /q chroma_db' on Windows)
6. Running the Application
Running the Flask Server Locally
Navigate to your  directory in the terminal and run:

Bash

python agent.py
You should see output similar to this, indicating that the Flask app is starting and listening on port 3010:

Starting Document Assistant Flask app on [http://127.0.0.1:3010](http://127.0.0.1:3010)
Waiting for incoming requests...
Loading documents from: /app/data (supporting multiple file types via UnstructuredFileLoader)
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
Containerized Deployment (Docker)
This is the recommended way to run the application for consistency and to avoid local dependency conflicts.

Docker Prerequisites
Ensure Docker Desktop (Windows/macOS) or Docker Engine (Linux) is installed and running.
On Windows, ensure your Docker Desktop File Sharing settings include the drive where your  project directory resides, and that the data and chroma_db folders have appropriate permissions for Docker to access.
Build the Docker Image
Navigate to your  project directory (where Dockerfile is located) in your terminal and run:

Bash

docker build -t venkat5ai/ai-agent-doc:1.0 -t venkat5ai/ai-agent-doc:latest .
This command builds the Docker image. It will install all system dependencies (Poppler, Tesseract, NLTK data, etc.) and Python packages inside the image. This process might take some time, especially on the first build.

Run the Docker Container
Once the image is built, you can run the container. Ensure you replace MY_GOOGLE_API_KEY with your actual Google Gemini API key. The -v flags mount your local data and chroma_db directories into the container, allowing the app to read your documents and persist the vector store.

Bash

docker run --name ai-agent-doc --rm -p 3010:3010 \
           -v "<span class="math-inline">\(pwd\)/data\:/app/data" \\
\-v "</span>(pwd)/chroma_db:/app/chroma_db" \
           -e GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY" \
           venkat5ai/ai-agent-doc:latest cloud
--name ai-agent-doc: Assigns a readable name to your container.
--rm: Automatically removes the container when it exits.
-p 3010:3010: Maps port 3010 on your host to port 3010 inside the container.
-v "$(pwd)/data:/app/data": Mounts your local data directory (containing your  documents) into /app/data inside the container.
-v "$(pwd)/chroma_db:/app/chroma_db": Mounts your local chroma_db directory (where the vector store will be persisted) into /app/chroma_db inside the container. If ./chroma_db doesn't exist on the host, Docker will create it automatically.
-e GOOGLE_API_KEY="MY_GOOGLE_API_KEY": Passes your Google Gemini API key as an environment variable to the container.
venkat5ai/ai-agent-doc:latest: Specifies the Docker image to run.
cloud: This argument is passed to your entrypoint.sh script.
Inspect the Container (Optional)
If you need to access the container's shell to debug or inspect files (e.g., to verify if chroma_db files are created inside /app/chroma_db), open a new terminal window (while the container is running) and use:

Bash

docker run --name ai-agent-doc --rm -it -p 3010:3010 -v "./data:/app/data" -e GOOGLE_API_KEY="%GOOGLE_API_KEY%" --entrypoint /bin/bash venkat5ai/ai-agent-doc:latest

docker exec -it ai-agent-doc sh
You can then cd /app/chroma_db and ls -la to see the generated files.

Accessing the Web UI
Once the server is running (either locally or via Docker), open your web browser and navigate to:

http://127.0.0.1:3010/

You will see the chat interface. You can type your questions and interact with the  Assistant. The UI will automatically manage the session ID.

Testing the API (using curl)
You can directly interact with the API endpoint using curl commands.

First Request (to start a new session):
A session_id will be generated if not provided. Copy this session_id from the response.

Bash

curl -X POST -H "Content-Type: application/json" -d "{\"question\": \"What is the annual  due amount?\"}" [http://127.0.0.1:3010/api/bot](http://127.0.0.1:3010/api/bot)
Example Response:

JSON

{"answer": "The annual  due amount is $X,000, payable on [date]...", "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"}
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

{"message": " Assistant is running", "status": "healthy"}
7. Technical Details
Retrieval-Augmented Generation (RAG)
The core of the system is RAG, which combines the power of a large language model with a retrieval mechanism. When a question is asked, the system first retrieves relevant chunks of information from the  documents (based on semantic similarity) and then uses the LLM to generate an answer grounded in that retrieved context.

Document Handling (Unstructured)
The unstructured library is used to intelligently parse and extract text from a wide variety of document types, including .pdf, .docx, .xlsx, .csv, .pptx, and .txt. It handles complex layouts and leverages tools like Poppler and Tesseract OCR for text extraction from images within documents.

Embedding Model
The all-MiniLM-L6-v2 Sentence Transformer model is used to convert text (document chunks and user questions) into numerical vector embeddings. These embeddings allow for efficient semantic search within the vector database.

Vector Database (ChromaDB)
ChromaDB is used as a lightweight, local vector store. It persists the document embeddings on the filesystem in the ./chroma_db directory, allowing for fast retrieval of relevant document chunks during the RAG process without re-embedding documents on every application start (unless the chroma_db folder is deleted).

Large Language Model (LLM)
The system utilizes the models/gemini-2.5-flash-preview-05-20 model from Google Gemini through the langchain_google_genai integration. This powerful LLM is responsible for understanding the question, synthesizing information from retrieved documents, and generating coherent, conversational answers.

Conversational Memory
ConversationBufferMemory from LangChain is used to store the history of the current conversation within each user session. This allows the LLM to consider previous turns in the dialogue when generating responses, leading to more natural and contextually aware interactions. Each session_id (managed by the client, e.g., the web UI's JavaScript or explicitly by curl) is tied to its own memory instance.

Hybrid RAG & Persona Prompting
The system is designed to provide answers primarily from the  documents. However, if the retrieved context is insufficient, the LLM's general knowledge will be leveraged, providing a seamless fallback without explicitly informing the user that the answer came from outside the documents. The LLM is also prompted to act as a "diligent and accurate  assistant."

CORS (Cross-Origin Resource Sharing)
The Flask-CORS extension is used to handle Cross-Origin Resource Sharing headers. This is essential to allow the JavaScript running in your web browser (which typically considers localhost and 127.0.0.1 as different origins) to make POST requests to your Flask API endpoint.

Eager Loading
The RAG components (ChromaDB vector store loading/building, LLM initialization, embedding model setup) are performed once at the Flask application's startup. This "eager loading" strategy ensures that the application is fully ready to serve requests as soon as it begins listening on the specified port, eliminating "cold start" delays for the first API call.

8. Project Structure
ai-workshop/
└── RAGs/
    └── /
        ├── data/
        │   ├── <your--document-1.pdf>
        │   ├── <your--document-2.docx>
        │   ├── <your--document-3.xlsx>
        │   └── <your--document-N.txt>
        ├── chroma_db/                 <-- Auto-generated by application (stores vector embeddings)
        │   └── <chroma_db_files>
        ├── templates/
        │   └── index.html             <-- HTML for the web UI
        ├── .env                       <-- (Optional) For environment variables
        ├── agent.py                   <-- Main Flask application
        ├── config.py                  <-- Configuration settings
        ├── requirements.txt           <-- Python dependencies
        └── Dockerfile                 <-- Docker build instructions
        └── entrypoint.sh              <-- Docker container entrypoint script
9. Troubleshooting
"Error: Could not connect to the assistant." (in browser UI):

Ensure your Flask server is running (either python agent.py or via docker run).
Verify Flask-CORS is installed (pip install Flask-CORS) and CORS(app) is enabled in agent.py.
Check your browser's developer console (F12, then "Console" or "Network" tab) for CORS errors.
Ensure the JavaScript API endpoint in index.html (currently http://127.0.0.1:3010/api/bot) correctly matches your Flask app's address and port.
"Error listing models" or LLM-related errors (in Flask console):

Double-check that your GOOGLE_API_KEY environment variable is correctly set in the terminal before you run the application (or in your Docker run command).
Generate a new API key from Google AI Studio and try again.
Ensure your machine has internet access to reach Google's API.
"Loaded 0 document(s)." or RAG-related errors:

Verify that your documents are placed in the ./data directory relative to where the agent.py script (or Docker container) is run.
Check file permissions for the data directory and the document files.
If running in Docker, ensure your docker run -v "$(pwd)/data:/app/data" command is correct and the host data directory is properly mounted and accessible.
ImportError: libGL.so.1: cannot open shared object file: No such file or directory:

This indicates missing OpenGL libraries required by unstructured's image processing (OpenCV).
Fix: Ensure your Dockerfile includes libgl1, libglib2.0-0, libxrender1, libxext6 in the apt-get install command. Rebuild your Docker image after making this change.
ImportError: cannot import name 'PSSyntaxError' from 'pdfminer.pdfparser':

This is a version incompatibility with pdfminer.six.
Fix: Ensure pdfminer.six==20221105 is specified in your requirements.txt. Rebuild your Docker image if you change requirements.txt.
LookupError: Resource <nltk_resource_name> not found. (e.g., punkt, punkt_tab, averaged_perceptron_tagger_eng, maxent_ne_chunker, words):

This means NLTK (used by unstructured for text processing) is missing required data packages.
Fix: Ensure your Dockerfile includes all necessary nltk.download() commands. For example:
Dockerfile

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
After modifying the Dockerfile, force a rebuild of the image to ensure the downloads occur (e.g., by adding a comment to the line to invalidate Docker's cache).
FileNotFoundError: [Errno 2] No such file or directory: 'tesseract':

This indicates that the Tesseract OCR engine, required by unstructured for image-based text extraction (e.g., from scanned PDFs), is not installed or not in the container's PATH.
Fix: Ensure your Dockerfile includes tesseract-ocr and tesseract-ocr-eng in the apt-get install command. Rebuild your Docker image after making this change.
chroma_db folder not appearing on host (when running with Docker on Windows):

Even if the chroma_db directory is created inside the container, Windows Docker Desktop might have permission/file sharing issues preventing it from appearing or being accessible on the host.
Fix:
In Docker Desktop settings (Settings > Resources > File Sharing), ensure the drive where your project is located is shared.
Manually create the chroma_db folder on your host machine before running the docker run command, and ensure your Windows user has full read/write permissions to it.
10. Future Enhancements
Streamlined Document Updates: Implement an API endpoint to upload new documents and dynamically update the vector store without restarting the application.
User Authentication/Authorization: Add a login system to restrict access to the chat assistant.
Error Handling and Logging: More robust logging and error handling for production environments.
Advanced RAG Techniques: Explore more sophisticated retrieval methods (e.g., HyDE, Cohere Rerank) or multi-modal RAG.
Deployment Automation: Scripts or configurations for automated deployment to cloud platforms (AWS, GCP, Azure).