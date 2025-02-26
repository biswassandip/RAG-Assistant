import os
import torch
from database import SessionLocal, Configuration
from llama_cpp import Llama  # For local LLMs (Llama.cpp)
from langchain_huggingface import HuggingFaceEndpoint  # For cloud LLMs
import asyncio

# Function to get config from database
def get_config_value(key):
    db = SessionLocal()
    config_entry = db.query(Configuration).filter(Configuration.key == key).first()
    db.close()
    print(f"{key} : {config_entry.value}")
    return config_entry.value if config_entry else None

class LLMService:
    def __init__(self):
        """Dynamically select LLM based on configuration from the database."""
        self.provider = get_config_value("llm-provider")

        if self.provider == "local":
            # Load local models (TinyLlama, Mistral 7B, Llama 2 7B)
            # model_paths = {
            #     "tinyllama": get_config_value("model-path"),
            #     "mistral7b": get_config_value("model-path"),
            #     "llama2": get_config_value("model-path"),
            # }
            selected_model = get_config_value("local-llm-model")
            model_path = get_config_value("model-path")

            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"LLM Model file not found at {model_path}. Please check your database configuration.")

            print(f"🟢 Streaming {selected_model} from {model_path}...")

            self.model = Llama(
                model_path=model_path,
                n_ctx=int(get_config_value("n-ctx")),  # Context length
                n_batch=int(get_config_value("n-batch")),  # Batch size
                n_threads=os.cpu_count(),  # Use all CPU cores
                n_gpu_layers=20 if torch.cuda.is_available() else 0  # Use GPU if available
            )

        elif self.provider == "cloud":
            # Explicitly set parameters directly in HuggingFaceEndpoint
            self.model = HuggingFaceEndpoint(
                repo_id=get_config_value("cloud-llm-model"),
                max_length=int(get_config_value("max-tokens")),
                temperature=float(get_config_value("temperature")),
                top_p=float(get_config_value("top-p")),
                huggingfacehub_api_token=get_config_value("api-token")
            )

        else:
            raise ValueError("Invalid LLM_PROVIDER. Use 'cloud' or 'local'.")

    def generate(self, prompt):
        """Generate a response from the LLM."""
        if self.provider == "local":
            output = self.model(prompt, max_tokens=int(get_config_value("max-tokens")), temperature=float(get_config_value("temperature")))
            return output["choices"][0]["text"].strip()

        elif self.provider == "cloud":
            return self.model.invoke(prompt)

    async def generate_stream(self, prompt):
        """Stream responses token-by-token."""
        if self.provider == "local":
            for output in self.model(prompt, max_tokens=int(get_config_value("max-tokens")), temperature=float(get_config_value("temperature")), stream=True):
                yield output["choices"][0]["text"]

        elif self.provider == "cloud":
            async for token in self.model.astream(prompt):
                yield token

# Initialize LLM service
llm_service = LLMService()
