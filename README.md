# RAG-Assistant
fastapi → API framework.
uvicorn → FastAPI server.
langchain → RAG implementation.
faiss-cpu → Vector database.
sqlite3 → Local database.
streamlit → Admin panel.
jwt → Authentication.

models availabilitt

Download the Larger Models
Llama 2 GGUF
wget -P https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

Mistral 7B GGUF
wget -P https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

Download the Smaller Models
Download TinyLlama 1.1B (FASTEST)
wget -P https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf
Download Mistral 7B (Quantized)
wget -P https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
Download Llama 2 7B (Quantized)
wget -P https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

brew install python@3.11
python3 --version
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
python3 --version

python3 -m venv rag-backend
source rag-backend/bin/activate
deactivate
