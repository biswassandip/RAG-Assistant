import os
import torch
from config import CONFIG
from llama_cpp import Llama  # For local LLMs (Llama.cpp)
from langchain_huggingface import HuggingFaceEndpoint  # For cloud LLMs
import asyncio


# Load local Llama model if needed
if CONFIG["LLM_PROVIDER"] == "local":
    from llama_cpp import Llama

# Load Hugging Face API integration
if CONFIG["LLM_PROVIDER"] == "cloud":
    from langchain_huggingface import HuggingFaceEndpoint


class LLMService:
    def __init__(self):
        """Dynamically select LLM based on configuration."""
        self.provider = CONFIG["LLM_PROVIDER"]

        if self.provider == "local":
            # Load local models (TinyLlama, Mistral 7B, Llama 2 7B)
            model_paths = {
                "tinyllama": CONFIG["TINYLLAMA_PATH"],
                "mistral7b": CONFIG["MISTRAL_PATH"],
                "llama2": CONFIG["LLAMA2_PATH"],
            }
            selected_model = CONFIG["LOCAL_LLM_MODEL"]
            model_path = model_paths.get(selected_model)

            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"LLM Model file not found at {model_path}. Please check your .env configuration.")

            print(f"🟢 Streaming {selected_model} from {model_path}...")

            self.model = Llama(
                model_path=model_path,
                n_ctx=CONFIG["N_CTX"],  # Context length
                n_batch=CONFIG["N_BATCH"],  # Batch size
                n_threads=os.cpu_count(),  # Use all CPU cores
                n_gpu_layers=20 if torch.cuda.is_available() else 0  # Use GPU if available
            )

        elif self.provider == "cloud":
            # Explicitly set parameters directly in HuggingFaceEndpoint
            self.model = HuggingFaceEndpoint(
                repo_id=CONFIG["DEFAULT_CLOUD_MODEL"],
                max_length=CONFIG["MAX_TOKENS"],  # Dynamically set max_tokens
                temperature=CONFIG["TEMPERATURE"],
                top_p=CONFIG["TOP_P"],
                huggingfacehub_api_token=os.getenv("HF_API_TOKEN")  # Load token from .env
            )

        else:
            raise ValueError("Invalid LLM_PROVIDER. Use 'cloud' or 'local'.")

    async def generate_stream(self, prompt):
        """Stream responses token-by-token."""
        if self.provider == "local":
            # Stream tokens from Llama.cpp model
            for output in self.model(prompt, max_tokens=CONFIG["MAX_TOKENS"], temperature=CONFIG["TEMPERATURE"], stream=True):
                yield output["choices"][0]["text"]

        elif self.provider == "cloud":
            # Stream tokens from Hugging Face
            async for token in self.model.astream(prompt):
                yield token


# Initialize LLM service
llm_service = LLMService()
