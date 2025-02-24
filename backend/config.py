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
    # Chunking Configuration (Ensure these are integers)
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 500)),  # Default 500 chars
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 100)),  # Default 100 chars
    # Default to TinyLlama
    "LOCAL_LLM_MODEL": os.getenv("LOCAL_LLM_MODEL", "tinyllama"),
    "TINYLLAMA_PATH": os.getenv("TINYLLAMA_PATH"),
    "MISTRAL_PATH": os.getenv("MISTRAL_PATH"),
    "LLAMA2_PATH": os.getenv("LLAMA2_PATH"),
}

