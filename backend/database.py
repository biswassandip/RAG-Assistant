from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./config.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Configuration(Base):
    __tablename__ = "configurations"
    key = Column(String, primary_key=True, index=True)
    value = Column(String, nullable=False)

def init_db():
    """Initialize database and insert default values."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    default_configs = {
        "llm-provider": "local",
        "llm-model": "llama2",
        "local-llm-model": "mistral7b",
        "model-path": "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "cloud-llm-model": "HuggingFaceH4/zephyr-7b-beta",
        "api-token": "",
        "chunk-size": "500",
        "chunk-overlap": "100",
        "alpha": "0.6",
        "top-k": "7",
        "max-tokens": "1024",
        "temperature": "0.5",
        "top-p": "0.95",
        "n-ctx": "2048",
        "n-batch": "64",
        "similarity-threshold": "0.2",
        "embedding-model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "enable-adaptive-threshold": "1",
        "default-response-length": "medium"
    }

    for key, value in default_configs.items():
        if not db.query(Configuration).filter(Configuration.key == key).first():
            db.add(Configuration(key=key, value=value))
    db.commit()
    db.close()

init_db()
