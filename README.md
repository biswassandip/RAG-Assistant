# 🚀 RAG-Backend: Local LLM-Powered Retrieval-Augmented Generation System

![GitHub Repo Size](https://img.shields.io/github/repo-size/your-username/rag-backend)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 📖 About RAG-Backend
**RAG-Backend** is an **open-source, local LLM-powered Retrieval-Augmented Generation (RAG) system** that allows users to **upload multiple file types** (PDFs, DOCX, Excel, Images, XML, etc.), **index them into a vector database**, and **perform intelligent searches** to get accurate responses.  

### 🛠 How It Works (Layman's Terms)
- Think of this as your **personal AI-powered knowledge base**.  
- You **upload documents**, and the system **remembers** what’s inside.  
- Whenever you **ask a question**, it **retrieves** the most relevant content and generates a smart answer using a **local LLM (like Llama2, Mistral, TinyLlama)**.  
- **No internet dependency** – everything runs **locally on your machine**.  

### ⚙️ Technical Overview
- **Retrieval-Augmented Generation (RAG):** Combines **vector-based retrieval (FAISS)** + **BM25 keyword-based retrieval**.  
- **Local LLM-based AI Chat:** Uses **Llama.cpp**, allowing users to interact with an **AI model offline**.  
- **Hybrid Search:** Both **semantic similarity search (FAISS)** and **keyword-based ranking (BM25)**.  
- **Multi-format Document Processing:** Supports **PDFs, DOCX, Excel, Images, XML, Text, etc.**.  
- **LangChain-powered Query Expansion:** Improves search relevance.  
- **Streaming Response:** Answers are streamed **in real-time** via WebSocket.  
- **Efficient Indexing:** Handles large document uploads with **adaptive chunking**.  

---

## 🌟 Why Use RAG-Backend?
### 🏆 Benefits (Layman's Perspective)
✅ **Your Personal AI Research Assistant:** Search through thousands of documents and get **instant, relevant answers**.  
✅ **Works Offline:** No internet required, ensuring **data privacy**.  
✅ **Handles Multiple File Types:** PDFs, Word Docs, Excel, Images, etc.  
✅ **Faster Response with Streaming:** No long waits – answers **stream in real-time**.  
✅ **Open-Source & Free:** Use it **without** API costs or cloud dependencies.  

### 📊 Benefits (Technical Perspective)
✅ **Hybrid Retrieval (Semantic + BM25):** Best of both **vector** and **keyword search**.  
✅ **Supports Local LLMs:** Runs on **TinyLlama, Llama2, Mistral-7B** models via `llama.cpp`.  
✅ **Optimized Query Expansion:** Uses **LangChain** to **expand queries** dynamically.  
✅ **Efficient Document Chunking:** Uses **RecursiveCharacterTextSplitter** for **optimal embedding**.  
✅ **Fully Asynchronous WebSockets:** Ensures **low-latency real-time AI responses**.  

### 🏢 Who is this for?
- 📚 **Researchers & Academics**: Search through **research papers, books, and notes** instantly.  
- 🏢 **Businesses & Enterprises**: Use it as an **internal document search assistant**.  
- 🧑‍💻 **Developers & AI Enthusiasts**: Experiment with **RAG & LLM models locally**.  
- 🔒 **Privacy-Conscious Users**: No cloud API means **your data stays with you**.  

---

## 🏗 Architecture Diagram
Below is the architecture of how **RAG-Backend** processes data and generates responses.

```mermaid
graph TD;
    User -->|Uploads| FileProcessor;
    FileProcessor -->|Extracts text| Chunking;
    Chunking -->|Embeds| FAISSIndex;
    FAISSIndex -->|Stores & Retrieves| VectorDB;
    User -->|Asks a Question| QueryProcessor;
    QueryProcessor -->|Retrieves Docs| VectorDB;
    QueryProcessor -->|Enhances Query| LangChain;
    LangChain -->|Passes Context| LocalLLM;
    LocalLLM -->|Generates Response| ResponseStreamer;
    ResponseStreamer -->|Streams Answer| User;
