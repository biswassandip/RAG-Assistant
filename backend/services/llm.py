import os
from config import CONFIG
import torch
from llama_cpp import Llama  # Load Llama.cpp for local inference

# Load local Llama model if needed
if CONFIG["LLM_PROVIDER"] == "local":
    from llama_cpp import Llama

# Load Hugging Face API integration
if CONFIG["LLM_PROVIDER"] == "cloud":
    from langchain_huggingface import HuggingFaceEndpoint


class LLMService:
    def __init__(self):
        """Dynamically select LLM based on configuration."""
        if CONFIG["LLM_PROVIDER"] == "local":
            # Select model path based on configuration
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

            print(f"🟢 Loading {selected_model} from {model_path}...")

            # Optimized settings for faster inference
            self.model = Llama(
                model_path=model_path,
                n_ctx=1024,  # Reduce context size for speed
                n_batch=32,  # Increase batch size
                n_threads=os.cpu_count(),  # Use all CPU cores
                n_gpu_layers=20 if torch.cuda.is_available() else 0  # Use GPU if available
            )

        elif CONFIG["LLM_PROVIDER"] == "cloud":
            # Explicitly set parameters directly in HuggingFaceEndpoint
            self.model = HuggingFaceEndpoint(
                repo_id=CONFIG["DEFAULT_CLOUD_MODEL"],
                max_length=256,
                temperature=0.7,
                huggingfacehub_api_token=os.getenv("HF_API_TOKEN")  # Load token from .env
            )

        else:
            raise ValueError("Invalid LLM_PROVIDER. Use 'cloud' or 'local'.")

    def generate(self, prompt):
        """Generate response using the selected LLM."""
        if CONFIG["LLM_PROVIDER"] == "local":
            output = self.model(prompt, max_tokens=128, temperature=0.7)
            return output["choices"][0]["text"].strip()

        elif CONFIG["LLM_PROVIDER"] == "cloud":
            # Generate response using Hugging Face API
            return self.model.invoke(prompt)


# Initialize LLM service
llm_service = LLMService()
