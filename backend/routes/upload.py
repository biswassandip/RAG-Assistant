from fastapi import APIRouter, UploadFile, File, Form
import os
import shutil
from PyPDF2 import PdfReader
import docx
from services.vectorstore import vector_store

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_text_from_file(file_path):
    """Extract text from PDF, DOCX, or TXT files."""
    _, ext = os.path.splitext(file_path)

    if ext.lower() == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    elif ext.lower() == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    return ""


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Handle document uploads and add to FAISS."""
    texts = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = extract_text_from_file(file_path)
        if text:
            texts.append(text)

    if texts:
        vector_store.add_documents(texts)
        return {"message": "Files uploaded and added to FAISS successfully!"}

    return {"message": "No valid text extracted from the uploaded files."}
