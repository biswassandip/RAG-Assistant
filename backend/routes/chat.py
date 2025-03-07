from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from datetime import datetime
from services.vectorstore import vector_store
from services.llm import llm_service
from routes.upload import uploaded_files_metadata  # Import uploaded file metadata

router = APIRouter()
chat_history = {}  # Dictionary to store chat history per client


@router.websocket("/chat")
async def chat_stream(websocket: WebSocket):
    """WebSocket endpoint for chat with history tracking."""
    await websocket.accept()

    try:
        # Receive client_id from frontend
        client_data = await websocket.receive_json()
        client_id = client_data.get("client_id", websocket.client.host)

        if client_id not in chat_history:
            chat_history[client_id] = []

        while True:
            question = await websocket.receive_text()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Retrieve documents and identify relevant files
            relevant_files = []
            results = vector_store.retrieve(question)

            if results:
                # Ensure unique files using a set with tuple representation
                seen_files = set()
                unique_files = []

                for file in results["retrieved_files"]:
                    file_tuple = (file["file_name"],
                                  file["file_type"], file["uploaded_date"])
                    if file_tuple not in seen_files:
                        seen_files.add(file_tuple)
                        unique_files.append(file)

                relevant_files = unique_files  # Assign unique files

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

            # Send only unique retrieved files
            for file in relevant_files:
                await websocket.send_json({"type": "faiss", "data": f"📄 {file['file_name']} ({file['file_type']}, Uploaded: {file['uploaded_date']})"})

            # Stream LLM response
            final_response = ""
            async for token in llm_service.generate_stream(f"Context:\n{summary_response}\n\nUser Question:\n{question}\n\nFinal Answer:"):
                final_response += token
                await websocket.send_json({"type": "llm", "data": token})

            # Save chat history per client
            chat_history[client_id].append({
                "timestamp": timestamp,
                "question": question,
                "retrieved_files": relevant_files,
                "retrieved_summary": summary_response,
                "final_answer": final_response
            })

            await websocket.send_json({"type": "end", "data": "✅ RAG Response Complete"})

    except WebSocketDisconnect:
        print(f"🔴 Client {client_id} disconnected. Cleaning up session.")

    except Exception as e:
        print(f"⚠️ Unexpected error in WebSocket: {str(e)}")
        await websocket.close()


@router.get("/chat/history")
def get_chat_history(client_id: str = Query(None, description="Client ID to retrieve specific chat history")):
    """Retrieve chat history for a specific client. If no client_id is provided, return all history."""
    if client_id:
        return chat_history.get(client_id, [])
    return chat_history
