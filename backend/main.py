from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.upload import router as upload_router
from routes.query import router as query_router
from routes.chat import router as chat_router  # Import WebSocket route
from routes.admin import router as admin_router
from constants import APP_TITLE, APP_DESCRIPTION, APP_HOST, APP_PORT
import database  # Ensure database initializes on startup

app = FastAPI(title=APP_TITLE,
              description=APP_DESCRIPTION)

# Serve static HTML for the chat UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(upload_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(admin_router, prefix="/api")


# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host=APP_HOST, port=APP_PORT)
