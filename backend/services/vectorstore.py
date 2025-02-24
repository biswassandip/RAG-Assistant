import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CONFIG  # Import configuration


class VectorStore:
    def __init__(self):
        """Initialize FAISS vector store with Hugging Face embeddings and configurable chunking."""
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
            embedding_function=self.embeddings,  # Correct embedding function
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
        )

        # Get chunking values from config with error handling
        chunk_size = CONFIG.get("CHUNK_SIZE", 500)  # Default: 500 characters
        # Default: 100 characters
        chunk_overlap = CONFIG.get("CHUNK_OVERLAP", 100)

        # Define configurable chunking strategy
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def add_documents(self, texts):
        """Add text documents to FAISS vector store with chunking."""
        chunks = []
        for text in texts:
            split_texts = self.chunker.split_text(text)
            for chunk in split_texts:
                chunks.append(Document(page_content=chunk))

        self.store.add_documents(chunks)  # Store chunked text

    def retrieve(self, query, k=3):
        """Retrieve top-k most relevant chunks."""
        return self.store.similarity_search(query, k=k)


# Initialize FAISS store
vector_store = VectorStore()
