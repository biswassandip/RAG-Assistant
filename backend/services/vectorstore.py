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
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")

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
            chunk_size=int(CONFIG.get("CHUNK_SIZE", 500)),
            chunk_overlap=int(CONFIG.get("CHUNK_OVERLAP", 100)),
        )

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

    def retrieve(self, query, k=3, alpha=0.5):
        """Retrieve top-k most relevant documents using hybrid search."""
        if not self.text_corpus:
            return ["No documents available in the knowledge base."]

        num_docs = len(self.text_corpus)  # Total number of stored documents

        # BM25 Sparse Retrieval
        bm25_scores = np.zeros(num_docs)
        if self.bm25_model:
            bm25_scores = self.bm25_model.get_scores(query.split())

        # FAISS Dense Retrieval
        dense_results = self.store.similarity_search(
            query, k=k) if self.text_corpus else []

        # Create an empty FAISS score array with the same shape as BM25
        dense_scores = np.zeros(num_docs)

        # Assign scores for only the retrieved FAISS documents
        for i, doc in enumerate(dense_results):
            doc_index = self.text_corpus.index(doc.page_content)
            dense_scores[doc_index] = 1 / (i + 1)  # Higher rank = lower score

        # **Avoid ZeroDivision Error in Normalization**
        if np.max(bm25_scores) > 0:
            bm25_scores /= np.max(bm25_scores)
        if np.max(dense_scores) > 0:
            dense_scores /= np.max(dense_scores)

        # Hybrid Score (Weighted Sum)
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

        # Get Top-k Documents
        top_indices = np.argsort(hybrid_scores)[::-1][:k]
        retrieved_docs = [self.text_corpus[i] for i in top_indices]

        return retrieved_docs if retrieved_docs else ["No relevant documents found."]


# Initialize Hybrid Store
vector_store = HybridVectorStore()
