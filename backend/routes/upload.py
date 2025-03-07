from fastapi import APIRouter, UploadFile, File
import os
import shutil
from datetime import datetime
from services.vectorstore import vector_store
from constants import URL_UPLOAD, URL_GET_FILES

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_files_metadata = []


def extract_text(file_path):
    """Extract text from PDF, DOCX, or TXT."""
    import docx
    from PyPDF2 import PdfReader

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages])

    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    return ""



@router.post(URL_UPLOAD)
async def upload_files(files: list[UploadFile] = File(...)):
    """Handle file uploads and store metadata."""
    global uploaded_files_metadata

    texts = []
    metadata_list = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ensure this function extracts text correctly
        text = extract_text(file_path)
        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        status = "Success" if text else "Failed"

        metadata = {
            "uploaded_date": upload_date,
            "file_name": file.filename,
            "file_type": file.content_type or "Unknown",
            "status": status,
            "file_about": text[:100] if text else "Could not extract text"
        }

        uploaded_files_metadata.append(metadata)

        if text:
            texts.append(text)
            metadata_list.append(metadata)

    # Ensure both texts and metadata are passed
    if texts:
        vector_store.add_documents(texts, metadata_list)

    return {"message": "Files processed successfully!", "metadata": uploaded_files_metadata}


@router.get(URL_GET_FILES)
def get_uploaded_files():
    """Retrieve uploaded file metadata."""
    return uploaded_files_metadata
