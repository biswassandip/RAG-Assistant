import os
import torch
import asyncio
from database import SessionLocal, Configuration
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import TokenTextSplitter
from services.vectorstore import vector_store

# Constants for database config keys
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


def get_config_value(key):
    """Fetch the latest configuration value from the database."""
    db = SessionLocal()
    config_entry = db.query(Configuration).filter(
        Configuration.key == key).first()
    db.close()
    return config_entry.value if config_entry else None


class LLMService:
    def __init__(self):
        """Initialize LLMService with LangChain and conversation memory."""
        self.model = None  # Lazy-loaded model
        self.config = None  # Cached configuration
        self.memory = ConversationBufferMemory(
            memory_key="history", return_messages=True)

    def get_model(self):
        """Fetch the latest model and configurations dynamically."""
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
                f"LLM Model file not found at {config[MODEL_PATH]}.")

        # Apply model-specific optimizations
        if TINYLLAMA in config[LOCAL_LLM_MODEL].lower():
            config.update({TEMPERATURE: 0.7, TOP_P: 0.9, REPEAT_PENALTY: 1.2})
        elif LLAMA2 in config[LOCAL_LLM_MODEL].lower():
            config.update({TEMPERATURE: 0.5, TOP_P: 0.8, REPEAT_PENALTY: 1.1})
        elif MISTRAL in config[LOCAL_LLM_MODEL].lower():
            config.update(
                {TEMPERATURE: 0.7, TOP_P: 0.95, REPEAT_PENALTY: 1.05})

        if self.config == config:
            return self.model, config  # Return cached model

        print("🔄 Reloading LLM model due to configuration change...")

        self.model = LlamaCpp(
            model_path=config[MODEL_PATH],
            n_ctx=config[CONTEXT_WINDOW],
            n_batch=config[N_BATCH],
            n_threads=os.cpu_count(),
            n_gpu_layers=20 if torch.cuda.is_available() else 0,
            temperature=config[TEMPERATURE],
            top_p=config[TOP_P],
            repeat_penalty=config[REPEAT_PENALTY],
        )

        self.config = config  # Cache latest configuration
        return self.model, config

    def retrieve_context(self, query):
        """Retrieve relevant context using FAISS vector store."""
        return vector_store.retrieve(query)

    def truncate_prompt(self, prompt):
        """Ensure prompt fits within the context window using LangChain’s `TokenTextSplitter`."""
        model, config = self.get_model()
        context_window = config[CONTEXT_WINDOW]

        splitter = TokenTextSplitter(
            chunk_size=context_window - 200, chunk_overlap=20
        )
        truncated_text = splitter.split_text(prompt)

        if not truncated_text:
            raise ValueError("⚠️ Prompt could not be tokenized properly.")

        return truncated_text[0]  # Use first valid chunk

    def generate(self, prompt):
        """Generate response using LangChain's `RunnableSequence`."""
        model, config = self.get_model()
        prompt = self.truncate_prompt(prompt)

        # Define prompt template for LangChain
        template = PromptTemplate(
            input_variables=["input"], template="{input}")

        chain = template | model
        return chain.invoke({"input": prompt}).strip()

    async def generate_stream(self, prompt):
        """Stream responses token-by-token for WebSocket clients."""
        model, config = self.get_model()
        prompt = self.truncate_prompt(prompt)

        async for token in self.local_stream(model, prompt, config[MAX_TOKENS]):
            yield token

    async def local_stream(self, model, prompt, max_tokens):
        """Helper function for local model streaming."""
        used_tokens = 0

        async for output in model.astream(prompt, max_tokens=min(max_tokens, 512)):
            if used_tokens >= max_tokens:
                break

            yield output
            used_tokens += len(output.split())
            await asyncio.sleep(0)

    def get_selected_model(self):
        """Fetch updated model details for UI."""
        return {
            "model": get_config_value(LOCAL_LLM_MODEL),
            "path": get_config_value(MODEL_PATH),
        }


# Initialize LLM service
llm_service = LLMService()
