
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Settings:
    """Configuration settings"""

    MAX_RETRIEVED_DOCS: int = 5 
    SIMILARITY_THRESHOLD: float = 0.3
    CONTEXT_WINDOW_SIZE: int = 4000
    
    # Embedding Service
    EMBEDDING_SERVICE: str = os.getenv("EMBEDDING_SERVICE", "sentence_transformers")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Vector Database
    VECTOR_DB: str = os.getenv("VECTOR_DB", "chromadb")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    
    # Data Paths
    DATA_PATH: str = os.getenv("DATA_PATH", "data/sample_deals.json")
    PROCESSED_DATA_PATH: str = os.getenv("PROCESSED_DATA_PATH", "data/processed/")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "data/vector_db/")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Model Settings
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    HUGGINGFACE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Database Settings
    CHROMADB_COLLECTION_NAME: str = "sales_deal_patterns"
    PINECONE_INDEX_NAME: str = "sales-sentiment-analysis"
    VECTOR_DIMENSION: int = 384  # for all-MiniLM-L6-v2
    
    def __init__(self):
        # Create directories if they don't exist
        Path(self.PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
    
    def validate(self):
        """Validate configuration"""
        if self.EMBEDDING_SERVICE == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI embedding service")
        
        if self.VECTOR_DB == "pinecone" and (not self.PINECONE_API_KEY or not self.PINECONE_ENVIRONMENT):
            raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT are required when using Pinecone")
        
        if not Path(self.DATA_PATH).exists():
            raise FileNotFoundError(f"Data file not found: {self.DATA_PATH}")

# Global settings instance
settings = Settings()