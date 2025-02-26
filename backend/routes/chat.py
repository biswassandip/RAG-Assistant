from fastapi import APIRouter, WebSocket
from datetime import datetime
from services.vectorstore import vector_store
from services.llm import llm_service
from routes.upload import uploaded_files_metadata  # Import uploaded file metadata

router = APIRouter()
chat_history = []

@router.websocket("/chat")
async def chat_stream(websocket: WebSocket):
    """WebSocket endpoint for chat with history tracking."""
    await websocket.accept()

    while True:
        question = await websocket.receive_text()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Retrieve documents and identify relevant files
        relevant_files = []
        results = vector_store.retrieve(question)
        if results:
            relevant_files = results["retrieved_files"]

        # Improved summarization prompt
        summary_prompt = f"""
        You are a highly intelligent assistant. Given the retrieved documents below, generate a concise and relevant summary that directly answers the user's question.

        - **Only include relevant facts**
        - **Avoid generic or repetitive information**
        - **If multiple documents are retrieved, merge overlapping details**
        - **Do not include unnecessary introductions or disclaimers**
        - **Structure the summary logically**

        **User Question:** {question}

        **Retrieved Documents:** 
        {results}

        **Generate a factual summary:**
        """

        # Generate a summary using the improved prompt
        summary_response = llm_service.generate(summary_prompt)

        # Send summarized retrieval response first
        await websocket.send_json({"type": "faiss", "data": f"🔹 Summary:\n{summary_response}\n\n📌 Referenced Files:"})
        # for file in relevant_files:
        #     await websocket.send_json({"type": "faiss", "data": f"📄 {file['file_name']} ({file['file_type']}, Uploaded: {file['uploaded_date']})"})

        # Stream LLM response
        final_response = ""
        async for token in llm_service.generate_stream(f"Context:\n{summary_response}\n\nUser Question:\n{question}\n\nFinal Answer:"):
            final_response += token
            await websocket.send_json({"type": "llm", "data": token})

        # Save chat history
        chat_history.append({
            "timestamp": timestamp,
            "question": question,
            "retrieved_files": relevant_files,
            "retrieved_summary": summary_response,
            "final_answer": final_response
        })

        await websocket.send_json({"type": "end", "data": "✅ RAG Response Complete"})


@router.get("/chat/history")
def get_chat_history():
    """Retrieve chat history for the dashboard."""
    return chat_history
