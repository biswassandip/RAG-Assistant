from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from datetime import datetime
# from services.vectorstore import vector_store
from services.llm import llm_service
from routes.upload import uploaded_files_metadata  # Import uploaded file metadata
from fastapi import APIRouter
from database import SessionLocal, Configuration
from constants import URL_CHAT, URL_CHAT_HISTORY

router = APIRouter()
chat_history = {}  # Dictionary to store chat history per client


def get_config_value(key):
    db = SessionLocal()
    config_entry = db.query(Configuration).filter(
        Configuration.key == key).first()
    db.close()
    return config_entry.value if config_entry else "N/A"


@router.websocket(URL_CHAT)
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
            retrieved_files = []
            retrieved_docs = []
            results = llm_service.retrieve_context(question)
            if results:
                retrieved_docs = results.get("retrieved_docs", [])
                retrieved_files = results.get("retrieved_files", [])

            # get the selected models and path used
            selected_models = llm_service.get_selected_model()


            # Improved summarization prompt
            summary_prompt = f"""
            Context from retrieved documents:
            {retrieved_docs}

            User's Question: {question}

            Provide a concise and well-structured answer:
            """

            # Generate a summary using the improved prompt
            summary_response = ""
            # ✅ Properly awaiting async generator
            async for token in llm_service.generate_stream(summary_prompt):
                summary_response += token
                # Streaming summary
                await websocket.send_json({"type": "summary", "data": token})

            # ✅ Send Retrieved Files Info
            await websocket.send_json({"type": "faiss", "data": f"🔹 Summary:\n{summary_response}"})
            # for file in retrieved_files:
            #     await websocket.send_json({"type": "faiss", "data": f"📄 {file['file_name']} ({file['file_type']}, Uploaded: {file['uploaded_date']})"})

            final_answer_prompt = f"""
            You are an expert assistant providing detailed and well-structured answers.
            Use the provided summary to enhance your response, but do NOT copy it verbatim. Instead:

            1. Reinterpret the summary using your own words.
            2. Expand on key points with additional insights.
            3. If needed, add any missing details based on the retrieved documents.
            4. Ensure a well-structured, logically flowing answer.

            **Summary of Retrieved Information:**
            {summary_response}

            **User's Original Question:**
            {question}

            **Provide a detailed and insightful answer:**
            """

            # ✅ Now Stream LLM Response
            final_response = ""
            async for token in llm_service.generate_stream(final_answer_prompt):
                final_response += token
                # ✅ Streaming final response
                await websocket.send_json({"type": "llm", "data": token})

            # Save chat history per client
            chat_history[client_id].append({
                "timestamp": timestamp,
                "question": question,
                "retrieved_summary": summary_response,
                "final_answer": final_response,
                "selected_models": selected_models
            })

            await websocket.send_json({"type": "end", "data": "✅ RAG Response Complete"})

    except WebSocketDisconnect:
        print(f"🔴 Client {client_id} disconnected. Cleaning up session.")

    except Exception as e:
        print(f"⚠️ Unexpected error in WebSocket: {str(e)}")
        await websocket.close()


@router.get(URL_CHAT_HISTORY)
def get_chat_history(client_id: str = Query(None, description="Client ID to retrieve specific chat history")):
    """Retrieve chat history for a specific client. If no client_id is provided, return all history."""
    if client_id:
        return chat_history.get(client_id, [])
    return chat_history
