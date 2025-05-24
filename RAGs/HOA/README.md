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