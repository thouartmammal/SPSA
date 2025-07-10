import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List

# Load environment variables
load_dotenv()

class Settings:
    """Centralized configuration - all settings controllable from environment"""

    # =================== CORE APPLICATION ===================
    APP_NAME: str = os.getenv("APP_NAME", "Sales Sentiment RAG API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # =================== API CONFIGURATION ===================
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    
    # =================== AUTHENTICATION ===================
    REQUIRE_AUTH: bool = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
    API_KEY: Optional[str] = os.getenv("API_KEY")
    JWT_SECRET: Optional[str] = os.getenv("JWT_SECRET")
    
    # =================== RATE LIMITING ===================
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_CALLS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_CALLS_PER_MINUTE", "60"))
    RATE_LIMIT_CALLS_PER_HOUR: int = int(os.getenv("RATE_LIMIT_CALLS_PER_HOUR", "1000"))
    
    # =================== CACHING ===================
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    LLM_CACHE_TTL: int = int(os.getenv("LLM_CACHE_TTL", "1800"))
    EMBEDDING_CACHE_TTL: int = int(os.getenv("EMBEDDING_CACHE_TTL", "7200"))
    
    # =================== LLM PROVIDERS ===================
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "azure").lower()
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    
    # Azure OpenAI
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    
    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    
    # Groq
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    # =================== EMBEDDING SERVICE ===================
    EMBEDDING_SERVICE: str = os.getenv("EMBEDDING_SERVICE", "sentence_transformers").lower()
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # Sentence Transformers
    SENTENCE_TRANSFORMER_MODEL: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    
    # OpenAI Embeddings
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    
    # =================== VECTOR DATABASE ===================
    VECTOR_DB: str = os.getenv("VECTOR_DB", "chromadb").lower()
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "data/vector_db")
    
    # ChromaDB
    CHROMADB_COLLECTION_NAME: str = os.getenv("CHROMADB_COLLECTION_NAME", "sales_sentiment_deals")
    
    # Pinecone
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "sales-sentiment-rag")
    
    # =================== DATA PATHS ===================
    DATA_PATH: str = os.getenv("DATA_PATH", "data/final_deal_details.json")
    PROCESSED_DATA_PATH: str = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    LOGS_PATH: str = os.getenv("LOGS_PATH", "logs")
    
    # =================== RAG CONFIGURATION ===================
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "10"))
    RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.1"))
    RAG_CONTEXT_WINDOW: int = int(os.getenv("RAG_CONTEXT_WINDOW", "4000"))
    
    # =================== CONTEXT BUILDER ===================
    CONTEXT_INCLUDE_SIMILAR_DEALS: bool = os.getenv("CONTEXT_INCLUDE_SIMILAR_DEALS", "true").lower() == "true"
    CONTEXT_INCLUDE_SENTIMENT_PATTERNS: bool = os.getenv("CONTEXT_INCLUDE_SENTIMENT_PATTERNS", "true").lower() == "true"
    CONTEXT_INCLUDE_LANGUAGE_TONE: bool = os.getenv("CONTEXT_INCLUDE_LANGUAGE_TONE", "true").lower() == "true"
    CONTEXT_INCLUDE_DEAL_PROGRESSION: bool = os.getenv("CONTEXT_INCLUDE_DEAL_PROGRESSION", "true").lower() == "true"
    CONTEXT_INCLUDE_CLIENT_BEHAVIOR: bool = os.getenv("CONTEXT_INCLUDE_CLIENT_BEHAVIOR", "true").lower() == "true"
    
    # =================== LOGGING ===================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "structured")  # structured or simple
    
    # =================== KNOWLEDGE BASE ===================
    KB_REBUILD_ON_STARTUP: bool = os.getenv("KB_REBUILD_ON_STARTUP", "false").lower() == "true"
    KB_BATCH_SIZE: int = int(os.getenv("KB_BATCH_SIZE", "50"))
    KB_ENABLE_INCREMENTAL: bool = os.getenv("KB_ENABLE_INCREMENTAL", "true").lower() == "true"
    
    def __init__(self):
        """Initialize settings and create required directories"""
        self._create_directories()
        self._validate_settings()
    
    def _create_directories(self):
        """Create required directories"""
        directories = [
            self.PROCESSED_DATA_PATH,
            self.VECTOR_DB_PATH,
            self.LOGS_PATH,
            os.path.dirname(self.DATA_PATH)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_settings(self):
        """Validate required settings based on configuration"""
        errors = []
        
        # Validate LLM provider settings
        if self.LLM_PROVIDER == "azure":
            if not all([self.AZURE_OPENAI_API_KEY, self.AZURE_OPENAI_ENDPOINT, 
                       self.AZURE_OPENAI_DEPLOYMENT_NAME]):
                errors.append("Azure OpenAI configuration incomplete")
        elif self.LLM_PROVIDER == "openai":
            if not self.OPENAI_API_KEY:
                errors.append("OpenAI API key required")
        elif self.LLM_PROVIDER == "anthropic":
            if not self.ANTHROPIC_API_KEY:
                errors.append("Anthropic API key required")
        elif self.LLM_PROVIDER == "groq":
            if not self.GROQ_API_KEY:
                errors.append("Groq API key required")
        
        # Validate embedding service
        if self.EMBEDDING_SERVICE == "openai" and not self.OPENAI_API_KEY:
            errors.append("OpenAI API key required for embeddings")
        
        # Validate vector database
        if self.VECTOR_DB == "pinecone":
            if not all([self.PINECONE_API_KEY, self.PINECONE_ENVIRONMENT]):
                errors.append("Pinecone configuration incomplete")
        
        # Validate data path
        if not Path(self.DATA_PATH).exists():
            errors.append(f"Data file not found: {self.DATA_PATH}")
        
        # Validate authentication
        if self.REQUIRE_AUTH and not self.API_KEY:
            errors.append("API key required when authentication is enabled")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration for current provider"""
        configs = {
            "azure": {
                "api_key": self.AZURE_OPENAI_API_KEY,
                "endpoint": self.AZURE_OPENAI_ENDPOINT,
                "deployment_name": self.AZURE_OPENAI_DEPLOYMENT_NAME,
                "api_version": self.AZURE_OPENAI_API_VERSION
            },
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "model": self.ANTHROPIC_MODEL
            },
            "groq": {
                "api_key": self.GROQ_API_KEY,
                "model": self.GROQ_MODEL
            }
        }
        
        return configs.get(self.LLM_PROVIDER, {})
    
    def get_embedding_config(self) -> dict:
        """Get embedding service configuration"""
        if self.EMBEDDING_SERVICE == "openai":
            return {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_EMBEDDING_MODEL
            }
        else:
            return {
                "model": self.SENTENCE_TRANSFORMER_MODEL
            }
    
    def get_vector_db_config(self) -> dict:
        """Get vector database configuration"""
        if self.VECTOR_DB == "pinecone":
            return {
                "api_key": self.PINECONE_API_KEY,
                "environment": self.PINECONE_ENVIRONMENT,
                "index_name": self.PINECONE_INDEX_NAME
            }
        else:
            return {
                "path": self.VECTOR_DB_PATH,
                "collection_name": self.CHROMADB_COLLECTION_NAME
            }

# Global settings instance
settings = Settings()