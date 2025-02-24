# RAG-Assistant
fastapi → API framework.
uvicorn → FastAPI server.
langchain → RAG implementation.
faiss-cpu → Vector database.
sqlite3 → Local database.
streamlit → Admin panel.
jwt → Authentication.


backend/
│── .env                      # Configuration settings
│── requirements.txt          # Dependencies
│── config.py                 # Load environment variables
│── main.py                   # FastAPI server
│── retriever.py              # Retrieval logic
│── generator.py              # LLM response generation
│── vectorstore.py            # FAISS/Pinecone vector storage
│── data_loader.py            # Upload & process documents
│── utils.py                  # Utility functions
│── models/                   # LLM model integrations
│── data/                     # Uploaded files
│── auth.py                   # Authentication (JWT)
│── cache.py                  # Redis-based caching
│── monitor.py                # API Monitoring (Prometheus)
│── admin.py                  # API Key Management
│── admin_ui.py               # Admin Panel (Streamlit)
│── database.py               # SQLite for storing API keys
│── tests/                    # Unit tests
