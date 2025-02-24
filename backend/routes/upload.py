import os
from fastapi import APIRouter, UploadFile, File
from services.vectorstore import vector_store

router = APIRouter()

UPLOAD_DIR = "data/"


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file, process it, and store chunks in FAISS."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file locally
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Read file contents & add chunked text to FAISS
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        vector_store.add_documents([text])  # Now chunked before storage

    return {"message": f"File '{file.filename}' uploaded, chunked, and indexed."}
