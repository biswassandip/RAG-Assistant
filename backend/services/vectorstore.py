import numpy as np
import re
import pytesseract
from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LC_Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from database import SessionLocal, Configuration


# Load Configuration
def get_config_value(key, default=None):
    """Fetch configuration value from the database."""
    db = SessionLocal()
    config_entry = db.query(Configuration).filter(
        Configuration.key == key).first()
    db.close()
    return config_entry.value if config_entry else default


class FAISSIndexManager:
    """Manages FAISS indexing separately, improving retrieval efficiency."""

    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize FAISS with the correct dimensions."""
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.index = None  # Index will be built when documents are added
        self.docstore = InMemoryDocstore()
        self.index_to_docstore_id = {}

    def add_documents(self, documents):
        """Add documents to FAISS index."""
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)

        # Initialize FAISS index if not already built
        if not self.index:
            self.index = FAISS.from_texts(texts, self.embeddings)

        self.index.add_texts(texts, metadatas)

    def search(self, query, k=5):
        """Perform FAISS search with similarity scoring."""
        if not self.index:
            return []

        return self.index.similarity_search_with_score(query, k=k)


class HybridVectorStore:
    def __init__(self):
        """Initialize FAISS for dense retrieval and BM25 for sparse retrieval."""
        embedding_model = get_config_value(
            "embedding-model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.faiss_manager = FAISSIndexManager(embedding_model)
        self.text_corpus = []
        self.bm25_corpus = []
        self.bm25_model = None

        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=int(get_config_value("chunk-size", 500)),
            chunk_overlap=int(get_config_value("chunk-overlap", 100)),
        )

        self.SIMILARITY_THRESHOLD = float(
            get_config_value("similarity-threshold", 0.2))

    def clean_text(self, text):
        """Pre-process text for indexing."""
        return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

    def process_image(self, image_path):
        """Extract text from images using OCR."""
        try:
            return pytesseract.image_to_string(Image.open(image_path)).strip()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return ""

    def process_xml(self, xml_path):
        """Extract text from XML files."""
        try:
            tree = ET.parse(xml_path)
            return " ".join([elem.text for elem in tree.iter() if elem.text])
        except Exception as e:
            print(f"Error processing XML {xml_path}: {e}")
            return ""

    def process_excel(self, excel_path):
        """Extract text from Excel files."""
        try:
            df = pd.read_excel(excel_path, sheet_name=None)
            return " ".join([" ".join(map(str, df[sheet].values.flatten())) for sheet in df])
        except Exception as e:
            print(f"Error processing Excel {excel_path}: {e}")
            return ""

    def add_documents(self, documents):
        """Add documents (text, images, xmls, excel) to FAISS and BM25."""
        chunks = []
        bm25_texts = []

        for doc in documents:
            if isinstance(doc, LC_Document):
                text = doc.page_content
                metadata = doc.metadata
            else:
                continue

            file_type = metadata.get("file_type", "text")
            file_path = metadata.get("file_path", "")

            if file_type == "image":
                text = self.process_image(file_path)
            elif file_type == "xml":
                text = self.process_xml(file_path)
            elif file_type == "excel":
                text = self.process_excel(file_path)

            split_texts = self.chunker.split_text(text)
            for chunk in split_texts:
                cleaned_chunk = self.clean_text(chunk)
                chunks.append(LC_Document(
                    page_content=cleaned_chunk, metadata=metadata))
                bm25_texts.append(cleaned_chunk)

        # Update FAISS index
        self.faiss_manager.add_documents(chunks)

        # Update BM25 model
        self.text_corpus.extend(bm25_texts)
        self.bm25_corpus = [doc.split() for doc in self.text_corpus]
        self.bm25_model = BM25Okapi(self.bm25_corpus)

    def expand_query(self, query):
        """Expand query using predefined synonym mappings."""
        synonym_map = {
            "error": ["issue", "bug", "problem"],
            "performance": ["speed", "efficiency"],
            "document": ["file", "report", "pdf"],
        }
        words = query.split()
        expanded_query = []
        for word in words:
            expanded_query.append(word)
            if word in synonym_map:
                expanded_query.extend(synonym_map[word])
        return " ".join(expanded_query)

    def retrieve(self, query):
        """Retrieve top-k documents using FAISS & BM25, with metadata."""
        if not self.text_corpus:
            return {"message": "No documents available in the knowledge base.", "retrieved_files": []}

        # Expand Query
        expanded_query = self.expand_query(query)

        num_docs = len(self.text_corpus)

        # BM25 Retrieval
        bm25_scores = np.zeros(num_docs)
        if self.bm25_model:
            bm25_scores = self.bm25_model.get_scores(expanded_query.split())

        # FAISS Dense Retrieval
        top_k = int(get_config_value("top-k", 7))
        dense_results_with_scores = self.faiss_manager.search(
            expanded_query, k=top_k)

        dense_results = [doc for doc, score in dense_results_with_scores]
        dense_scores = np.array(
            [score for _, score in dense_results_with_scores])

        if not dense_results:
            return {"message": "No relevant documents found.", "retrieved_files": []}

        # Choose Between Fixed & Adaptive Thresholding
        enable_adaptive_threshold = int(
            get_config_value("enable-adaptive-threshold", 1))
        if enable_adaptive_threshold:
            mean_score = np.mean(dense_scores)
            std_dev = np.std(dense_scores)
            similarity_threshold = max(mean_score - std_dev, 0.1)
        else:
            similarity_threshold = self.SIMILARITY_THRESHOLD


        # Extract metadata safely
        filtered_results = []
        retrieved_files = []

        for doc, score in zip(dense_results, dense_scores):
            if score > similarity_threshold:
                metadata = doc.metadata if hasattr(
                    doc, "metadata") else {}  # Ensure metadata exists

                filtered_results.append({
                    "content": doc.page_content,
                    "metadata": metadata
                })
                retrieved_files.append({
                    "file_name": metadata.get("file_name", "Unknown"),
                    "file_type": metadata.get("file_type", "Unknown"),
                    "uploaded_date": metadata.get("uploaded_date", "Unknown")
                })

        return {"retrieved_docs": filtered_results, "retrieved_files": retrieved_files}


# Initialize Hybrid Store
vector_store = HybridVectorStore()
