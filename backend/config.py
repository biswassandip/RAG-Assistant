import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONFIG = {
    # Default to cloud for speed
    "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "cloud"),  # Default: Cloud API
    "DEFAULT_CLOUD_MODEL": os.getenv("DEFAULT_CLOUD_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
    "LLM_MODEL": os.getenv("LLM_MODEL", "llama2"),
    "LLAMA_MODEL_PATH": os.getenv("LLAMA_MODEL_PATH"),
    "MISTRAL_MODEL_PATH": os.getenv("MISTRAL_MODEL_PATH"),
    "CLOUD_LLM_MODEL": os.getenv("CLOUD_LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta"),

    # Default to TinyLlama
    "LOCAL_LLM_MODEL": os.getenv("LOCAL_LLM_MODEL", "tinyllama"),
    "TINYLLAMA_PATH": os.getenv("TINYLLAMA_PATH"),
    "MISTRAL_PATH": os.getenv("MISTRAL_PATH"),
    "LLAMA2_PATH": os.getenv("LLAMA2_PATH"),

    # ==============================
    # ✅ Hybrid Search Configuration
    # ==============================

    # Chunking settings for document splitting
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 500)),
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 100)),

    # ==============================
    # ✅ FAISS & BM25 Hybrid Search Configuration
    # ==============================

    # Maximum number of documents retrieved for further processing
    "TOP_K": int(os.getenv("TOP_K", 7)),

    # Hybrid Search Weighting
    # 0.0 = Only FAISS (semantic search)
    # 1.0 = Only BM25 (keyword search)
    "ALPHA": float(os.getenv("ALPHA", 0.6)),

    # ==============================
    # ✅ FAISS Similarity Scoring & Thresholding
    # ==============================

    # Enable Adaptive Similarity Thresholding
    # 1 = Adaptive threshold (computed dynamically)
    # 0 = Use a fixed threshold (manual tuning)
    "ENABLE_ADAPTIVE_THRESHOLD": int(os.getenv("ENABLE_ADAPTIVE_THRESHOLD", 1)),

    # Fixed Similarity Threshold (Only used if ENABLE_ADAPTIVE_THRESHOLD=0)
    "SIMILARITY_THRESHOLD": float(os.getenv("SIMILARITY_THRESHOLD", 0.3)),

    # ==============================
    # ✅ FAISS Embedding Model
    # ==============================

    # Options for embedding models:
    # - sentence-transformers/all-MiniLM-L6-v2 (Fast, Medium Accuracy)
    # - sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (Optimized for Q&A)
    # - intfloat/multilingual-e5-base (Best for diverse queries)
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),

    # ==============================
    # ✅ LLM Configuration
    # ==============================

    # Maximum number of tokens in generated responses
    "MAX_TOKENS": int(os.getenv("MAX_TOKENS", 1024)),

    # Randomness level in response generation
    "TEMPERATURE": float(os.getenv("TEMPERATURE", 0.5)),

    # Probability mass for controlling diversity of responses
    "TOP_P": float(os.getenv("TOP_P", 0.95)),

    # Context length: How much conversation history is considered
    "N_CTX": int(os.getenv("N_CTX", 2048)),

    # Batch size: Number of tokens processed in parallel
    "N_BATCH": int(os.getenv("N_BATCH", 64)),

}
