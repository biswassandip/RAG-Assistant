import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from config import CONFIG


class HybridVectorStore:
    def __init__(self):
        """Initialize FAISS for dense retrieval and BM25 for sparse retrieval."""
        embedding_model = CONFIG.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Define FAISS index
        dimension = 384  # MiniLM produces 384-dim vectors
        self.index = faiss.IndexFlatL2(dimension)

        # Required docstore and index mapping
        self.docstore = InMemoryDocstore()
        self.index_to_docstore_id = {}

        # Initialize FAISS vector store
        self.store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
        )

        # BM25 Storage
        self.text_corpus = []
        self.bm25_corpus = []
        self.bm25_model = None

        # Define configurable chunking strategy
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
        )

        # Similarity threshold (tuneable in `.env`)
        self.SIMILARITY_THRESHOLD = float(CONFIG.get(
            "SIMILARITY_THRESHOLD", 0.2))  # Default: 0.2

    def add_documents(self, texts):
        """Add text documents to FAISS (dense) and BM25 (sparse) retrieval."""
        chunks = []
        bm25_texts = []

        for text in texts:
            split_texts = self.chunker.split_text(text)
            for chunk in split_texts:
                chunks.append(Document(page_content=chunk))
                bm25_texts.append(chunk)

        # Store in FAISS
        self.store.add_documents(chunks)

        # Store in BM25
        self.text_corpus.extend(bm25_texts)
        self.bm25_corpus = [doc.split() for doc in self.text_corpus]
        self.bm25_model = BM25Okapi(self.bm25_corpus)


    def retrieve(self, query):
        """Retrieve top-k documents using FAISS & BM25, with optional adaptive filtering."""
        if not self.text_corpus:
            return ["No documents available in the knowledge base."]

        num_docs = len(self.text_corpus)  # Total number of stored documents

        # BM25 Sparse Retrieval
        bm25_scores = np.zeros(num_docs)
        if self.bm25_model:
            bm25_scores = self.bm25_model.get_scores(query.split())

        # FAISS Dense Retrieval with Similarity Scores
        dense_results_with_scores = self.store.similarity_search_with_score(
            query, k=CONFIG["TOP_K"]) if self.text_corpus else []
        dense_results = [doc for doc, score in dense_results_with_scores]
        dense_scores = np.array([score for _, score in dense_results_with_scores])

        if not dense_results:
            return ["No relevant documents found."]

        # Choose Between Fixed & Adaptive Thresholding
        if CONFIG["ENABLE_ADAPTIVE_THRESHOLD"]:
            mean_score = np.mean(dense_scores)
            std_dev = np.std(dense_scores)
            # Ensures threshold remains reasonable
            similarity_threshold = max(mean_score - std_dev, 0.1)
        else:
            similarity_threshold = CONFIG["SIMILARITY_THRESHOLD"]

        # Filter Results Based on the Selected Threshold
        filtered_results = [
            doc.page_content for doc, score in zip(dense_results, dense_scores) if score > similarity_threshold
        ]

        return filtered_results if filtered_results else ["No relevant documents found."]



# Initialize Hybrid Store
vector_store = HybridVectorStore()
