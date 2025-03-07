import os
import torch
from database import SessionLocal, Configuration
from llama_cpp import Llama  # For local LLMs (Llama.cpp)
import asyncio
from fastapi.responses import StreamingResponse


# constants used for the config
LOCAL_LLM_MODEL = "local-llm-model"
MODEL_PATH = "model-path"
CONTEXT_WINDOW = "n-ctx"
MAX_TOKENS = "max-tokens"
TEMPERATURE = "temperature"
TOP_P = "top-p"
REPEAT_PENALTY = "repeat-penalty"
N_BATCH = "n-batch"
TINYLLAMA = "tinyllama"
LLAMA2 = "llama2"
MISTRAL = "mistral"


# fetches config values from the db
def get_config_value(key):
    """Fetches the latest configuration value from the database."""
    db = SessionLocal()
    config_entry = db.query(Configuration).filter(
        Configuration.key == key).first()
    db.close()
    return config_entry.value if config_entry else None


# main llm service
class LLMService:
    def __init__(self):
        """Initialize LLMService without storing static config values."""
        self.model = None  # Model is loaded dynamically
        self.config = None  # Config is stored in cache

    def get_model(self):
        """Dynamically fetch the latest model based on updated config."""

        # Fetch dynamic config
        config = {
            CONTEXT_WINDOW: int(get_config_value(CONTEXT_WINDOW)),
            MAX_TOKENS: int(get_config_value(MAX_TOKENS)),
            TEMPERATURE: float(get_config_value(TEMPERATURE)),
            TOP_P: float(get_config_value(TOP_P)),
            REPEAT_PENALTY: 1.1,
            MODEL_PATH: get_config_value(MODEL_PATH),
            LOCAL_LLM_MODEL: get_config_value(LOCAL_LLM_MODEL),
            N_BATCH: int(get_config_value(N_BATCH)),
        }

        if not config[MODEL_PATH] or not os.path.exists(config[MODEL_PATH]):
            raise FileNotFoundError(
                f"LLM Model file not found at {config[MODEL_PATH]}. Please check your database configuration."
            )

        # model specific tweaks to handle combinations
        # TO-DO: make it more dynamic
        if TINYLLAMA in config[LOCAL_LLM_MODEL].lower():
            config.update(
                {TEMPERATURE: 0.7, TOP_P: 0.9, REPEAT_PENALTY: 1.2})
        elif LLAMA2 in config[LOCAL_LLM_MODEL].lower():
            config.update(
                {TEMPERATURE: 0.5, TOP_P: 0.8, REPEAT_PENALTY: 1.1})
        elif MISTRAL in config[LOCAL_LLM_MODEL].lower():
            config.update(
                {TEMPERATURE: 0.7, TOP_P: 0.95, REPEAT_PENALTY: 1.05})

        # if the config has not changed then return the model and the config
        if self.config == config:
            return

        # load the model only if there is a change in the config.
        # assuming this will be reloaded other wise it would have been handled above
        print("🔄 Reloading LLM model due to configuration change...")
        return Llama(
            model_path=config[MODEL_PATH],
            n_ctx=config[CONTEXT_WINDOW],
            n_batch=config[N_BATCH],
            n_threads=os.cpu_count(),
            n_gpu_layers=20 if torch.cuda.is_available() else 0
        ), config[CONTEXT_WINDOW], config[MAX_TOKENS], config[TEMPERATURE], config[TOP_P], config[REPEAT_PENALTY]

    def truncate_prompt(self, prompt, context_window):
        """Ensure the prompt does not exceed the model's context window."""
        model, *_ = self.get_model()
        prompt_tokens = model.tokenize(prompt.encode("utf-8"))
        max_tokens = context_window - 100  # Leave space for response

        if len(prompt_tokens) > max_tokens:
            print(
                f"⚠️ Truncating prompt: {len(prompt_tokens)} → {max_tokens} tokens"
            )
            prompt_tokens = prompt_tokens[:max_tokens]

        return model.detokenize(prompt_tokens).decode("utf-8")

    def generate(self, prompt):
        """Generate a response from the LLM."""
        model, context_window, max_tokens, temperature, top_p, repeat_penalty = self.get_model()
        prompt = self.truncate_prompt(prompt, context_window)

        output = model(
            prompt,
            max_tokens=min(max_tokens, context_window // 2),
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty
        )
        return output["choices"][0]["text"].strip()

    async def generate_stream(self, prompt):
        """Stream responses token-by-token for WebSocket clients."""
        model, context_window, max_tokens, temperature, top_p, repeat_penalty = self.get_model()
        prompt = self.truncate_prompt(prompt, context_window)

        async for token in self.local_stream(
            model, prompt, max_tokens, temperature, top_p, repeat_penalty
        ):
            yield token

    async def local_stream(self, model, prompt, max_tokens, temperature, top_p, repeat_penalty):
        """Helper function for local model streaming."""
        for output in model(
            prompt,
            max_tokens=min(max_tokens, 512),  # Avoid excessive length
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stream=True
        ):
            yield output["choices"][0]["text"]
            await asyncio.sleep(0)  # Allow async execution

    def get_selected_model(self):
        """Fetch updated model details for UI."""
        return {
            "model": get_config_value("local-llm-model"),
            "path": get_config_value("model-path"),
        }


# Initialize LLM service
llm_service = LLMService()
