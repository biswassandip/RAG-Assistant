from fastapi import APIRouter, WebSocket
from services.vectorstore import vector_store
from services.llm import llm_service

router = APIRouter()


@router.websocket("/chat")
async def chat_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming FAISS and LLM responses."""
    await websocket.accept()

    while True:
        # Receive question from client
        question = await websocket.receive_text()
        print(f"Received question: {question}")

        # Retrieve documents from FAISS
        retrieved_docs = vector_store.retrieve(question)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Send FAISS response first
        faiss_response = {"type": "faiss", "data": context}
        await websocket.send_json(faiss_response)

        # Send streaming response from LLM
        response_text = ""
        async for token in llm_service.generate_stream(f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"):
            response_text += token
            await websocket.send_json({"type": "llm", "data": token})

        # Signal completion
        await websocket.send_json({"type": "end", "data": "RAG Response Complete"})
