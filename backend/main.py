from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.upload import router as upload_router
from routes.query import router as query_router
from routes.chat import router as chat_router  # Import WebSocket route

app = FastAPI(title="RAG API with Document Upload",
              description="A RAG backend supporting document upload & chat streaming.")

# Serve static HTML for the chat UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(upload_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="127.0.0.1", port=8000)
