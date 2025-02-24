from fastapi import APIRouter
from services.vectorstore import vector_store
from services.llm import llm_service

router = APIRouter()


@router.get("/query0")
def query_rag(question: str):
    """Retrieve documents from FAISS and generate a response using LLM."""
    retrieved_docs = vector_store.retrieve(question)

    # Combine retrieved chunks as context
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Format prompt for LLM
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    # Generate response using Hugging Face API or local model
    response = llm_service.generate(prompt)

    return {"response": response}


@router.get("/query")
def query_rag(question: str):
    """Retrieve documents using Hybrid Search and generate a response."""
    retrieved_docs = vector_store.retrieve(question)

    return {"retrieved_documents": retrieved_docs}
