from fastapi import FastAPI
from routes.upload import router as upload_router
from routes.query import router as query_router

app = FastAPI(title="RAG API", description="A simple RAG backend using FAISS.")

# Include API routes
app.include_router(upload_router, prefix="/api")
app.include_router(query_router, prefix="/api")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="127.0.0.1", port=8000)
